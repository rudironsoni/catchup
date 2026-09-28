package deepseek

import (
	"bytes"
	"context"
	"os"
	"path/filepath"
	"slices"
	"strings"
	"testing"
	"time"

	"github.com/klauspost/compress/zstd"

	"github.com/wilbeibi/catchup/internal/session"
)

// transcript mirrors a real dsh session log: header first, then append-only
// events. The plugin- and skill-catalog-sourced user rows, the packed chunk
// row, tool plumbing, and the compaction replacement are all noise the
// provider must skip.
const transcript = `{"type":"session","version":0,"id":"session-abc","createdAt":1786870675123,"cwd":"/home/u/src/xurl","delegationDepth":0,"agentPreset":"standard"}
{"type":"permission/preset","seq":0,"time":1786870675375,"data":{"preset":"workspace-write"}}
{"type":"turn/start","seq":1,"time":1786871592619,"data":{"turn":1}}
{"type":"step/start","seq":2,"time":1786871592687,"data":{"turn":1,"step":1}}
{"type":"request/header","seq":3,"time":1786871592688,"data":{"header":{"config":{"provider":"deepseek-official","model":"deepseek-v4-flash","reasoningEffort":"high","maxTokens":256000}}}}
{"type":"user/message","seq":4,"time":1786871592688,"data":{"content":[{"type":"text","text":"support dsh"}],"source":{"kind":"user","rpcId":"rpc-1"},"role":"user","id":"u1"},"surfaceOp":"append"}
{"type":"user/message","seq":5,"time":1786871592691,"data":{"content":[{"type":"text","text":"Current runtime context. Sandbox snapshot."}],"source":{"kind":"plugin","plugin":"@deepseek-ai/dsh-system-prompt","form":"snapshot"},"role":"user","id":"u2"},"surfaceOp":"append"}
{"type":"user/message","seq":6,"time":1786871592693,"data":{"content":[{"type":"text","text":"<system-reminder>skills catalog</system-reminder>"}],"source":{"kind":"skill-catalog"},"role":"user","id":"u3"},"surfaceOp":"append"}
{"type":"text-chunks","seq0":7,"time0":1786871595000,"data":{"chunks":[]}}
{"type":"assistant/message","seq":10,"time":1786871595652,"data":{"turn":1,"step":1,"message":{"content":[{"type":"reasoning","text":"hidden"},{"type":"text","text":"I will inspect the format."}],"role":"assistant","id":"a1","source":{"kind":"model","provider":"deepseek-official","model":"deepseek-v4-flash"}}},"surfaceOp":"append"}
{"type":"tool/call","seq":11,"time":1786871595700,"data":{"turn":1,"step":2,"callId":"c1","name":"bash","arguments":"{\"command\":\"ls\"}"}}
{"type":"tool/result","seq":12,"time":1786871595800,"data":{"turn":1,"step":2,"callId":"c1","content":[]}}
{"type":"assistant/message","seq":13,"time":1786871595900,"data":{"turn":1,"step":2,"message":{"content":[{"type":"text","text":"done"}],"role":"assistant","id":"a2","source":{"kind":"model","provider":"deepseek-official","model":"deepseek-v4-flash"}}},"surfaceOp":"append"}
{"type":"step/end","seq":14,"time":1786871595950,"data":{"turn":1,"step":2}}
{"type":"turn/end","seq":15,"time":1786871595951,"data":{"turn":1,"reason":"end_turn"}}
{"type":"session/title","seq":16,"time":1786871596000,"data":{"title":"dsh support","messageSeqs":[4],"source":{"kind":"provider","provider":"session-title-first-prompt-llm","model":{"provider":"deepseek-official","model":"deepseek-v4-flash"}}}}
{"type":"compaction/checkpoint","seq":17,"time":1786871600000,"data":{"summary":"summary so far"}}
{"type":"user/message","seq":18,"time":1786871600100,"data":{"content":[{"type":"text","text":"compacted replacement"}],"source":{"kind":"plugin","plugin":"compaction"},"role":"user","id":"u4"},"surfaceOp":{"op":"replace","shadowed":[4]}}
{"type":"user/message","seq":19,"time":1786871600200,"data":{"content":[{"type":"text","text":"finish it"}],"source":{"kind":"user","rpcId":"rpc-2"},"role":"user","id":"u5"},"surfaceOp":"append"}
{"type":"assistant/message","seq":20,"time":1786871600300,"data":{"turn":2,"step":1,"message":{"content":[{"type":"text","text":"first part"},{"type":"text","text":"second part"}],"role":"assistant","id":"a3","source":{"kind":"model","provider":"deepseek-official","model":"deepseek-v4-flash"}}},"surfaceOp":"append"}
`

var wantEntries = []session.Entry{
	{Kind: session.KindMessage, Role: session.RoleUser, Text: "support dsh", Time: time.UnixMilli(1786871592688)},
	{Kind: session.KindMessage, Role: session.RoleAssistant, Text: "I will inspect the format.", Time: time.UnixMilli(1786871595652)},
	{Kind: session.KindMessage, Role: session.RoleAssistant, Text: "done", Time: time.UnixMilli(1786871595900)},
	{Kind: session.KindCompact, Text: "summary so far", Time: time.UnixMilli(1786871600000)},
	{Kind: session.KindMessage, Role: session.RoleUser, Text: "finish it", Time: time.UnixMilli(1786871600200)},
	{Kind: session.KindMessage, Role: session.RoleAssistant, Text: "first part\nsecond part", Time: time.UnixMilli(1786871600300)},
}

// transcriptV3 is session-abc as dsh 0.1.5-rc.3 writes it at format version 3,
// sanitized from a real native v3 log with the tail a reopen appends to a
// migrated one. The system prompt travels as system/message rows, one replacing
// another; none of it, nor the reopen's end-seed and model selection, is
// conversation.
const transcriptV3 = `{"type":"session","version":3,"id":"session-abc","createdAt":1786870675123,"cwd":"/home/u/src/xurl","isSeeded":false,"delegationDepth":0,"agentPreset":"standard"}
{"type":"permission/preset","seq":0,"time":1786870675375,"data":{"preset":"workspace-write"}}
{"type":"turn/start","seq":1,"time":1786871592619,"data":{"turn":1}}
{"type":"step/start","seq":2,"time":1786871592687,"data":{"turn":1,"step":1}}
{"type":"system/message","seq":3,"time":1786871592688,"data":{"turn":1,"step":1,"message":{"id":"s1","role":"system","source":{"kind":"plugin","plugin":"@deepseek-ai/dsh-system-prompt"},"content":[]}},"surfaceOp":"append"}
{"type":"user/message","seq":4,"time":1786871592689,"data":{"content":[{"type":"text","text":"support dsh v3"}],"source":{"kind":"user","rpcId":"rpc-1","clientTimeZone":"America/Los_Angeles"},"role":"user","id":"u1"},"surfaceOp":"append"}
{"type":"user/message","seq":5,"time":1786871592690,"data":{"content":[{"type":"text","text":"<system-reminder>AGENTS.md</system-reminder>"}],"source":{"kind":"agent-instructions","form":"instructions","baseline":true},"role":"user","id":"u2"},"surfaceOp":"append"}
{"type":"system/message","seq":6,"time":1786871592691,"data":{"turn":1,"step":1,"message":{"id":"s2","role":"system","source":{"kind":"plugin","plugin":"@deepseek-ai/dsh-system-prompt"},"content":[{"type":"text","text":"You are an AI agent powered by DeepSeek Harness."}]}},"surfaceOp":{"op":"replace","startSeq":3,"endSeq":3},"sourceEventSeqs":[3]}
{"type":"request/header","seq":7,"time":1786871592692,"data":{"header":{"config":{"provider":"deepseek-official","model":"deepseek-flash","reasoningEffort":"high","maxTokens":256000}}}}
{"type":"session/title","seq":8,"time":1786871592693,"data":{"title":"dsh support v3","messageSeqs":[4],"source":{"kind":"fallback"}}}
{"type":"assistant/message","seq":9,"time":1786871595652,"data":{"turn":1,"step":1,"message":{"role":"assistant","content":[{"type":"reasoning","text":"hidden"},{"type":"text","text":"From v3."}],"source":{"kind":"model","provider":"deepseek-official","model":"deepseek-flash"},"id":"a1"},"usage":{"inputTokens":10,"outputTokens":5,"totalTokens":15},"stream":[{"type":"chunk","time":1786871595000,"chunk":{"type":"block-start","index":0,"blockType":"reasoning"}}]},"surfaceOp":"append"}
{"type":"step/end","seq":10,"time":1786871595653,"data":{"turn":1,"step":1}}
{"type":"turn/end","seq":11,"time":1786871595654,"data":{"turn":1,"reason":{"kind":"completed"}}}
{"type":"session/end-seed","seq":12,"time":1786957995654,"data":{}}
{"type":"model/selection","seq":13,"time":1786957996000,"data":{"provider":"deepseek-official","model":"deepseek-flash","reasoningEffort":"high"}}
`

// transcriptV4 is a new session as dsh 0.1.7-rc.2 writes it at format version
// 4, sanitized from a real headless run. The prompt's inbox copy, the system
// prompt, the runtime-context and skill-catalog rows, the title request and the
// session-log delivery watermarks are not conversation.
const transcriptV4 = `{"type":"session","version":4,"id":"session-new","createdAt":1790566681407,"cwd":"/home/u/src/other","isSeeded":false,"delegationDepth":0}
{"type":"permission/preset","seq":0,"time":1790566681421,"data":{"preset":"workspace-write"}}
{"type":"sandbox/mode","seq":1,"time":1790566681423,"data":{"mode":"workspace-write"}}
{"type":"approval/policy","seq":2,"time":1790566681423,"data":{"policy":"ask"}}
{"type":"agent/inbox/spliced","seq":3,"time":1790566681427,"data":{"target":"next-turn","start":0,"inserted":[{"content":[{"type":"text","text":"say pong"}],"source":{"kind":"user"},"role":"user","id":"u1"}]}}
{"type":"turn/start","seq":4,"time":1790566681428,"data":{"turn":1}}
{"type":"agent/inbox/spliced","seq":5,"time":1790566681429,"data":{"target":"next-turn","start":0,"removedCount":1,"inserted":[]}}
{"type":"step/start","seq":6,"time":1790566681535,"data":{"turn":1,"step":1}}
{"type":"system/message","seq":7,"time":1790566681537,"data":{"turn":1,"step":1,"message":{"role":"system","content":[{"type":"text","text":"You are an AI agent powered by DeepSeek Harness."}],"source":{"kind":"system-prompt"},"id":"s1"}},"surfaceOp":"append"}
{"type":"user/message","seq":8,"time":1790566681538,"data":{"content":[{"type":"text","text":"say pong"}],"source":{"kind":"user"},"role":"user","id":"u1"},"surfaceOp":"append"}
{"type":"user/message","seq":9,"time":1790566681540,"data":{"content":[{"type":"text","text":"Current runtime context.\n\nApproval policy: ask."}],"source":{"kind":"runtime-context","form":"snapshot","sections":[{"name":"approval:policy","text":"Approval policy: ask."}]},"role":"user","id":"u2"},"surfaceOp":"append"}
{"type":"user/message","seq":10,"time":1790566681541,"data":{"content":[{"type":"text","text":"<system-reminder>skills</system-reminder>"}],"source":{"kind":"skill-catalog","form":"catalog","entries":[{"name":"catchup","description":"Recovers a previous session."}]},"role":"user","id":"u3"},"surfaceOp":"append"}
{"type":"request/header","seq":11,"time":1790566681543,"data":{"header":{"config":{"provider":"deepseek-official","model":"deepseek-flash","maxTokens":256000,"reasoningEffort":"high"},"adapterDefaults":{"reasoningEffort":true,"maxTokens":true},"tools":[{"name":"bash","description":"Execute a bash command.","parameters":{"type":"object","properties":{"command":{"type":"string"}},"required":["command"]}}]},"reason":"initial"}}
{"type":"request/context","seq":12,"time":1790566681545,"data":{"provider":"deepseek-official","model":"deepseek-flash","contextWindow":1000000,"systemPromptUpdate":"in-history"}}
{"type":"session/title","seq":13,"time":1790566681550,"data":{"title":"say pong","messageSeqs":[8],"source":{"kind":"fallback"}}}
{"type":"session/title-llm-request","seq":14,"time":1790566681552,"data":{"titleProvider":"session-title-first-prompt-llm","messageSeqs":[8],"route":{"provider":"deepseek-official","model":"deepseek-flash"},"system":"Create a concise title.","messages":[{"content":[{"type":"text","text":"[{\"seq\":8,\"text\":\"say pong\"}]"}],"source":{"kind":"dsh-session-title-llm"},"role":"user","id":"t1"}],"maxTokens":64}}
{"type":"session-log-deepseek/delivery-accepted","seq":15,"time":1790566682154,"data":{"sessionId":"session-new","sessionFormatVersion":4,"throughSeq":14}}
{"type":"session-log-deepseek/delivery-accepted","seq":16,"time":1790566682182,"data":{"sessionId":"session-new","sessionFormatVersion":4,"throughSeq":14}}
{"type":"session/title","seq":17,"time":1790566682481,"data":{"title":"pong reply request","messageSeqs":[8],"source":{"kind":"provider","provider":"session-title-first-prompt-llm","model":{"provider":"deepseek-official","model":"deepseek-flash"}}}}
{"type":"assistant/message","seq":18,"time":1790566682620,"data":{"turn":1,"step":1,"message":{"role":"assistant","content":[{"type":"reasoning","text":""},{"type":"text","text":"pong"}],"source":{"kind":"model","provider":"deepseek-official","model":"deepseek-flash","replayState":{"response":{"kind":"deepseek-messages","version":1,"model":"deepseek-flash"},"blocks":[{"type":"reasoning","signature":"sig"},{"type":"text"}]}},"id":"a1"},"usage":{"inputTokens":6980,"outputTokens":3,"totalTokens":6983}},"surfaceOp":"append"}
{"type":"step/end","seq":19,"time":1790566682624,"data":{"turn":1,"step":1}}
{"type":"turn/end","seq":20,"time":1790566682629,"data":{"turn":1,"reason":{"kind":"completed"}}}
`

// subagent is a subagent's child session of session-abc: a directory of its
// own, named by a bare UUID, with the header a real one carries.
var subagent = strings.Replace(renameSession(transcript, subagentID, "/home/u/src/xurl"),
	`"delegationDepth":0,"agentPreset":"standard"`,
	`"parentSession":"session-abc","origin":"subagent","delegationDepth":1,"agentPreset":"code"`, 1)

const subagentID = "0d1e2f3a-4b5c-4d6e-8f70-819203a4b5c6"

// renameSession retargets a transcript copy at another id and cwd so two
// fixtures can live in different munged directories.
func renameSession(body, id, cwd string) string {
	s := strings.Replace(body, "session-abc", id, 1)
	return strings.Replace(s, "/home/u/src/xurl", cwd, 1)
}

// writeSession writes one file into a session directory, zstd-compressed when
// its name says so.
func writeSession(t *testing.T, root, munged, id, name, body string, mod time.Time) {
	t.Helper()
	dir := filepath.Join(root, "sessions", munged, id)
	if err := os.MkdirAll(dir, 0o755); err != nil {
		t.Fatal(err)
	}
	var buf []byte
	if strings.HasSuffix(name, ".zstd") {
		var z bytes.Buffer
		w, err := zstd.NewWriter(&z)
		if err != nil {
			t.Fatal(err)
		}
		if _, err := w.Write([]byte(body)); err != nil {
			t.Fatal(err)
		}
		if err := w.Close(); err != nil {
			t.Fatal(err)
		}
		buf = z.Bytes()
	} else {
		buf = []byte(body)
	}
	file := filepath.Join(dir, name)
	if err := os.WriteFile(file, buf, 0o644); err != nil {
		t.Fatal(err)
	}
	if err := os.Chtimes(file, mod, mod); err != nil {
		t.Fatal(err)
	}
}

func checkThread(t *testing.T, t2 session.Thread) {
	t.Helper()
	if t2.Source.Ref.Provider != session.ProviderDeepSeek || t2.Source.Ref.SessionID != "session-abc" {
		t.Fatalf("source ref = %+v", t2.Source.Ref)
	}
	md := t2.Source.Metadata
	if md["title"] != "dsh support" || md["cwd"] != "/home/u/src/xurl" {
		t.Fatalf("metadata = %+v", md)
	}
	if md["model"] != "deepseek-v4-flash" || md["model_provider"] != "deepseek-official" {
		t.Fatalf("model metadata = %+v", md)
	}
	if len(t2.Entries) != len(wantEntries) {
		t.Fatalf("got %d entries: %+v", len(t2.Entries), t2.Entries)
	}
	for i, e := range t2.Entries {
		if e != wantEntries[i] {
			t.Fatalf("entry %d = %+v, want %+v", i, e, wantEntries[i])
		}
	}
}

func TestReadDeepSeekSession(t *testing.T) {
	root := t.TempDir()
	writeSession(t, root, "--home-u-src-xurl--", "session-abc", "session.jsonl", transcript, time.Now())

	p := New()
	src, err := p.Resolve(context.Background(), session.Roots{DeepSeek: root}, "session-abc")
	if err != nil {
		t.Fatal(err)
	}
	thread, err := p.Read(context.Background(), src)
	if err != nil {
		t.Fatal(err)
	}
	checkThread(t, thread)
}

func TestReadDeepSeekSessionZstd(t *testing.T) {
	root := t.TempDir()
	writeSession(t, root, "--home-u-src-xurl--", "session-abc", "session.jsonl.zstd", transcript, time.Now())

	p := New()
	thread, err := p.Read(context.Background(), session.Source{
		Ref:  session.Ref{Provider: session.ProviderDeepSeek, SessionID: "session-abc"},
		Path: filepath.Join(root, "sessions", "--home-u-src-xurl--", "session-abc", "session.jsonl.zstd"),
	})
	if err != nil {
		t.Fatal(err)
	}
	checkThread(t, thread)
}

func TestReadTornTail(t *testing.T) {
	root := t.TempDir()
	// A crashed writer's last line is half-written; the parsed prefix must
	// survive with a warning.
	torn := transcript + `{"type":"turn/sta`
	writeSession(t, root, "--home-u-src-xurl--", "session-abc", "session.jsonl", torn, time.Now())

	p := New()
	thread, err := p.Read(context.Background(), session.Source{
		Ref:  session.Ref{Provider: session.ProviderDeepSeek, SessionID: "session-abc"},
		Path: filepath.Join(root, "sessions", "--home-u-src-xurl--", "session-abc", "session.jsonl"),
	})
	if err != nil {
		t.Fatal(err)
	}
	if len(thread.Warnings) != 1 || !strings.Contains(thread.Warnings[0], "the last record is incomplete") {
		t.Fatalf("warnings = %+v", thread.Warnings)
	}
	if len(thread.Entries) != len(wantEntries) {
		t.Fatalf("got %d entries, want the full parsed prefix: %+v", len(thread.Entries), thread.Entries)
	}
}

func TestResolveNewestAndByID(t *testing.T) {
	root := t.TempDir()
	old := renameSession(transcript, "session-old", "/home/u/src/other")
	writeSession(t, root, "--home-u-src-other--", "session-old", "session.jsonl", old, time.Now().Add(-2*time.Hour))
	writeSession(t, root, "--home-u-src-xurl--", "session-abc", "session.jsonl", transcript, time.Now())
	writeSession(t, root, "--home-u-src-xurl--", subagentID, "session.jsonl", subagent, time.Now().Add(time.Hour))

	p := New()
	src, err := p.Resolve(context.Background(), session.Roots{DeepSeek: root}, "")
	if err != nil {
		t.Fatal(err)
	}
	if src.Ref.SessionID != "session-abc" {
		t.Fatalf("newest = %q, want session-abc, never a subagent", src.Ref.SessionID)
	}

	src, err = p.Resolve(context.Background(), session.Roots{DeepSeek: root}, subagentID)
	if err != nil {
		t.Fatal(err)
	}
	if src.Ref.SessionID != subagentID || src.Metadata["parent"] != "session-abc" {
		t.Fatalf("subagent by id = %q, metadata %+v", src.Ref.SessionID, src.Metadata)
	}

	src, err = p.Resolve(context.Background(), session.Roots{DeepSeek: root}, "session-old")
	if err != nil {
		t.Fatal(err)
	}
	if src.Ref.SessionID != "session-old" {
		t.Fatalf("by id = %q, want session-old", src.Ref.SessionID)
	}

	if _, err := p.Resolve(context.Background(), session.Roots{DeepSeek: root}, "session-nope"); err == nil {
		t.Fatal("unknown id resolved without error")
	}
}

func TestListFilters(t *testing.T) {
	root := t.TempDir()
	writeSession(t, root, "--home-u-src-xurl--", "session-abc", "session.jsonl", transcript, time.Now())
	old := renameSession(transcript, "session-old", "/home/u/src/other")
	writeSession(t, root, "--home-u-src-other--", "session-old", "session.jsonl", old, time.Now().Add(-2*time.Hour))
	writeSession(t, root, "--home-u-src-xurl--", subagentID, "session.jsonl", subagent, time.Now().Add(-time.Hour))

	p := New()
	sums, err := p.List(context.Background(), session.Roots{DeepSeek: root}, session.ListOptions{})
	if err != nil {
		t.Fatal(err)
	}
	if len(sums) != 2 || sums[0].Ref.SessionID != "session-abc" || sums[0].Rank != 1 || sums[1].Rank != 2 {
		t.Fatalf("listing = %+v", sums)
	}
	if sums[0].Title != "dsh support" || sums[0].Cwd != "/home/u/src/xurl" || sums[0].Preview != "support dsh" {
		t.Fatalf("summary[0] = %+v", sums[0])
	}

	sums, err = p.List(context.Background(), session.Roots{DeepSeek: root}, session.ListOptions{Cwd: "/home/u/src/other"})
	if err != nil {
		t.Fatal(err)
	}
	if len(sums) != 1 || sums[0].Ref.SessionID != "session-old" {
		t.Fatalf("cwd-filtered listing = %+v", sums)
	}

	sums, err = p.List(context.Background(), session.Roots{DeepSeek: root}, session.ListOptions{Query: "inspect"})
	if err != nil {
		t.Fatal(err)
	}
	if len(sums) != 2 {
		t.Fatalf("query listing = %+v", sums)
	}

	sums, err = p.List(context.Background(), session.Roots{DeepSeek: root}, session.ListOptions{Query: "no-such-word"})
	if err != nil {
		t.Fatal(err)
	}
	if len(sums) != 0 {
		t.Fatalf("query listing = %+v, want none", sums)
	}
}

// TestMigratedSession reads session directories as dsh leaves them. Migrating
// session-abc wrote its version 3 log beside the untouched version 0 one, next
// to the lease and a manual backup; session-new began at version 4.
func TestMigratedSession(t *testing.T) {
	root := t.TempDir()
	mod := time.Now().Add(-time.Hour).Truncate(time.Second)
	writeSession(t, root, "--home-u-src-xurl--", "session-abc", "session.jsonl.zstd", transcript, mod.Add(-24*time.Hour))
	writeSession(t, root, "--home-u-src-xurl--", "session-abc", "session.v3.jsonl.zstd", transcriptV3, mod)
	writeSession(t, root, "--home-u-src-xurl--", "session-abc", "session.lock", "", time.Now())
	writeSession(t, root, "--home-u-src-xurl--", "session-abc", "session.jsonl.zstd.bak-x", transcript, time.Now())
	newMod := mod.Add(-time.Hour)
	writeSession(t, root, "--home-u-src-other--", "session-new", "session.v4.jsonl.zstd", transcriptV4, newMod)

	p := New()
	src, err := p.Resolve(context.Background(), session.Roots{DeepSeek: root}, "session-abc")
	if err != nil {
		t.Fatal(err)
	}
	if filepath.Base(src.Path) != "session.v3.jsonl.zstd" || !src.UpdatedAt.Equal(mod) {
		t.Fatalf("source = %s at %v, want the version 3 log at %v", src.Path, src.UpdatedAt, mod)
	}
	thread, err := p.Read(context.Background(), src)
	if err != nil {
		t.Fatal(err)
	}
	want := []session.Entry{
		{Kind: session.KindMessage, Role: session.RoleUser, Text: "support dsh v3", Time: time.UnixMilli(1786871592689)},
		{Kind: session.KindMessage, Role: session.RoleAssistant, Text: "From v3.", Time: time.UnixMilli(1786871595652)},
	}
	if !slices.Equal(thread.Entries, want) || len(thread.Warnings) != 0 {
		t.Fatalf("entries = %+v, warnings = %q", thread.Entries, thread.Warnings)
	}

	if src, err = p.Resolve(context.Background(), session.Roots{DeepSeek: root}, "session-new"); err != nil {
		t.Fatal(err)
	}
	if thread, err = p.Read(context.Background(), src); err != nil {
		t.Fatal(err)
	}
	want = []session.Entry{
		{Kind: session.KindMessage, Role: session.RoleUser, Text: "say pong", Time: time.UnixMilli(1790566681538)},
		{Kind: session.KindMessage, Role: session.RoleAssistant, Text: "pong", Time: time.UnixMilli(1790566682620)},
	}
	if !slices.Equal(thread.Entries, want) || len(thread.Warnings) != 0 {
		t.Fatalf("version 4 entries = %+v, warnings = %q", thread.Entries, thread.Warnings)
	}

	sums, err := p.List(context.Background(), session.Roots{DeepSeek: root}, session.ListOptions{})
	if err != nil {
		t.Fatal(err)
	}
	if len(sums) != 2 || sums[0].Ref.SessionID != "session-abc" || sums[1].Ref.SessionID != "session-new" {
		t.Fatalf("listing = %+v, want each session once", sums)
	}
}

func TestEmptyRoot(t *testing.T) {
	p := New()
	if _, err := p.Resolve(context.Background(), session.Roots{DeepSeek: t.TempDir()}, ""); err == nil {
		t.Fatal("resolve on empty root succeeded")
	}
	sums, err := p.List(context.Background(), session.Roots{DeepSeek: t.TempDir()}, session.ListOptions{})
	if err != nil || len(sums) != 0 {
		t.Fatalf("list on empty root = %+v, %v", sums, err)
	}
}

func TestTitleFallsBackToCwdBase(t *testing.T) {
	root := t.TempDir()
	// Drop the session/title line: the fallback is the cwd's base name.
	var lines []string
	for _, l := range strings.Split(strings.TrimSpace(transcript), "\n") {
		if strings.Contains(l, `"type":"session/title"`) {
			continue
		}
		lines = append(lines, l)
	}
	body := strings.Join(lines, "\n") + "\n"
	writeSession(t, root, "--home-u-src-xurl--", "session-abc", "session.jsonl", body, time.Now())

	p := New()
	src, err := p.Resolve(context.Background(), session.Roots{DeepSeek: root}, "")
	if err != nil {
		t.Fatal(err)
	}
	if src.Metadata["title"] != "xurl" {
		t.Fatalf("title = %q, want xurl", src.Metadata["title"])
	}
}
