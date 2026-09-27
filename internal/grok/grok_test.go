package grok

import (
	"context"
	_ "embed"
	"encoding/json"
	"os"
	"path/filepath"
	"slices"
	"strings"
	"testing"
	"time"

	"github.com/wilbeibi/catchup/internal/session"
)

// Sanitized excerpts from session 01a0be48-8eca-7633-ba5c-f2f58d58a384,
// inspected with grok 1.0.41 (4220f3b224a6) on 2026-09-27. Updates source
// lines: 4,5,6,8,14,20,27,29,33,957,2027,7164,7890,14971,34313,34314.
// The checkpoint is the one referenced on line 2027. The last line is the
// /compact host-turn echo from a grok 1.0.41 headless session run 2026-09-27.
// Text, paths and IDs are replaced; timestamps are renumbered and telemetry
// omitted. Update/content shapes and relative ordering are preserved, including
// promptIndex on real turns and both compaction_meta records around a retained
// user turn. Rows added below for corrupt/unknown input are deliberate mutations.
// These excerpts establish the observed format, not compatibility with every
// Grok version. Listing visibility also follows upstream persistence.rs at
// xai-org/grok-build@f0e3be1100ef5252488e3be8bb0e91cf68d8c305.

//go:embed testdata/updates.jsonl
var acpLog string

//go:embed testdata/checkpoint.json
var checkpoint string

const cwd = "/home/u/src/proj"

func writeSession(t *testing.T, root, group, id string, summary map[string]any, updates string, mod time.Time) string {
	t.Helper()
	dir := filepath.Join(root, "sessions", group, id)
	if err := os.MkdirAll(dir, 0o755); err != nil {
		t.Fatal(err)
	}
	files := map[string]string{updatesFile: updates}
	if summary != nil {
		b, err := json.Marshal(summary)
		if err != nil {
			t.Fatal(err)
		}
		files["summary.json"] = string(b)
	}
	for name, body := range files {
		if body == "" {
			continue
		}
		path := filepath.Join(dir, name)
		if err := os.WriteFile(path, []byte(body), 0o644); err != nil {
			t.Fatal(err)
		}
		if err := os.Chtimes(path, mod, mod); err != nil {
			t.Fatal(err)
		}
	}
	return dir
}

func TestRead(t *testing.T) {
	at := func(n int64) time.Time { return time.UnixMilli(n * 1000) }
	// Literal expectations are independent of session.Failure and the parser.
	updates := []session.Entry{
		{Kind: "message", Role: "user", Text: "support grok", Time: at(1), Retained: true},
		{Kind: "message", Role: "assistant", Text: "I will read the log.", Time: at(3)},
		{Kind: "failure", Role: "tool", Tool: "list_dir", Input: `{"target_directory":"redacted"}`, Text: `{"type":"ListDir","NotFound":"redacted"}`, Time: at(7)},
		{Kind: "failure", Role: "tool", Tool: "read_file", Input: `{"target_file":"a.txt","limit":80}`, Text: "read failed", Time: at(8)},
		{Kind: "failure", Role: "tool", Text: "background failed", Time: at(10)},
		{Kind: "compact", Text: "the compaction summary", Time: at(11)},
		{Kind: "message", Role: "user", Text: "finish it", Time: at(13)},
		{Kind: "stop", Reason: "error", Time: at(14)},
		{Kind: "message", Role: "assistant", Text: "all done", Time: at(15)},
	}
	// A second compaction moves the mark to its own last real user turn: the
	// typed /loop, not the interjection after it. Both lines are shaped from
	// grok-build 4247f661 source, not observed in a log: a /loop prompt
	// (xai-grok-shell slash_commands.rs, instruction abbreviated) and a mid-turn
	// interjection (interjection.rs, xai-interjection-core format.rs).
	lines := strings.SplitAfter(acpLog, "\n")
	loop := `{"timestamp":18,"method":"session/update","params":{"sessionId":"fixture-session","update":{"sessionUpdate":"user_message_chunk","content":{"type":"text","text":"# /loop -- schedule a recurring prompt\n\n## Input\n5m check CI","_meta":{"displayText":"/loop 5m check CI"}},"_meta":{"modelId":"grok-build","promptIndex":18}},"_meta":{"agentTimestampMs":18000}}}` + "\n"
	interjection := `{"timestamp":19,"method":"session/update","params":{"sessionId":"fixture-session","update":{"sessionUpdate":"user_message_chunk","content":{"type":"text","text":"The user sent a message while you were working:\n<user_query>\nalso update the docs\n</user_query>\nMake sure to complete any unfinished tasks from previous turns.","_meta":{"displayText":"also update the docs"}},"_meta":{"modelId":"grok-build","interjection":true}},"_meta":{"agentTimestampMs":19000}}}` + "\n"
	again := slices.Clone(updates)
	again[0].Retained = false
	again = append(again,
		session.Entry{Kind: "message", Role: "user", Text: "/loop 5m check CI", Time: at(18), Retained: true},
		session.Entry{Kind: "message", Role: "user", Text: "also update the docs", Time: at(19)},
		session.Entry{Kind: "compact", Text: "the compaction summary", Time: at(11)})
	unknown := `{"params":{"update":{"sessionUpdate":"future_kind"}}}` + "\n"
	// The pre-envelope line upstream still reads (storage/mod.rs tests).
	legacy := `{"sessionId":"s","update":{"sessionUpdate":"user_message_chunk","content":{"type":"text","text":"legacy prompt"}}}`
	for _, tc := range []struct {
		name, updates string
		want          []session.Entry
		warnings      []string
	}{
		{"updates", acpLog, updates, nil},
		{"second compaction", acpLog + loop + interjection + lines[10], again, nil},
		{"timestamp fallback", strings.Replace(acpLog, `"agentTimestampMs":1000`, `"unused":1000`, 1), updates, nil},
		{"duplicate failure", acpLog + lines[7], updates, nil},
		{"torn updates", acpLog + `{"params":`, updates, []string{"malformed record"}},
		{"unknown update", acpLog + unknown, updates, []string{"future_kind"}},
		{"legacy envelope", legacy, []session.Entry{{Kind: "message", Role: "user", Text: "legacy prompt"}}, nil},
		{"no transcript", "", nil, nil},
	} {
		t.Run(tc.name, func(t *testing.T) {
			dir := writeSession(t, t.TempDir(), "group", "id", nil, tc.updates, time.Now())
			cp := filepath.Join(dir, "compaction_checkpoints")
			if err := os.Mkdir(cp, 0o755); err != nil {
				t.Fatal(err)
			}
			if err := os.WriteFile(filepath.Join(cp, "cp1.json"), []byte(checkpoint), 0o644); err != nil {
				t.Fatal(err)
			}
			th, err := New().Read(context.Background(), session.Source{Path: dir})
			if err != nil {
				t.Fatal(err)
			}
			if !slices.Equal(th.Entries, tc.want) {
				t.Fatalf("entries =\n%+v\nwant\n%+v", th.Entries, tc.want)
			}
			if len(tc.warnings) == 0 && len(th.Warnings) != 0 {
				t.Fatalf("unexpected warnings: %v", th.Warnings)
			}
			for _, want := range tc.warnings {
				if !strings.Contains(strings.Join(th.Warnings, "\n"), want) {
					t.Errorf("warnings = %v, want %q", th.Warnings, want)
				}
			}
		})
	}
}

func TestReadSourceErrors(t *testing.T) {
	file := filepath.Join(t.TempDir(), "not-a-directory")
	if err := os.WriteFile(file, nil, 0o644); err != nil {
		t.Fatal(err)
	}
	for _, path := range []string{"", file} {
		if _, err := New().Read(context.Background(), session.Source{Path: path}); err == nil {
			t.Errorf("read %q: expected an error", path)
		}
	}
}

func TestListAndResolve(t *testing.T) {
	root := t.TempDir()
	roots := session.Roots{Grok: root}
	old, fresh := time.Unix(100, 0), time.Unix(200, 0)
	for _, tc := range []struct {
		id, group, dir, title string
		active                time.Time
		mtime                 time.Time
	}{
		{"new", "%2Fhome%2Fu%2Fsrc%2Fproj", cwd, "grok support", fresh, old},
		{"old", "other-abc123", "/other", "other session", old, fresh},
	} {
		summary := map[string]any{
			"info":            map[string]string{"id": tc.id, "cwd": tc.dir},
			"generated_title": tc.title, "session_summary": "longer summary",
			"current_model_id": "grok-build", "last_active_at": tc.active.Format(time.RFC3339),
		}
		writeSession(t, root, tc.group, tc.id, summary, acpLog, tc.mtime)
	}
	p := New()
	for _, tc := range []struct {
		name string
		opts session.ListOptions
		ids  []string
	}{
		{"all", session.ListOptions{}, []string{"new", "old"}},
		{"cwd", session.ListOptions{Cwd: "/other"}, []string{"old"}},
		{"limit", session.ListOptions{Limit: 1}, []string{"new"}},
		{"pre-compaction query", session.ListOptions{Query: "read failed"}, []string{"new", "old"}},
		{"post-compaction query", session.ListOptions{Query: "finish it"}, []string{"new", "old"}},
		{"hidden context", session.ListOptions{Query: "reminder_token"}, nil},
		{"reasoning", session.ListOptions{Query: "reasoning-only-token"}, nil},
		{"successful tool", session.ListOptions{Query: "tool-only-token"}, nil},
	} {
		t.Run(tc.name, func(t *testing.T) {
			rows, err := p.List(context.Background(), roots, tc.opts)
			if err != nil || len(rows) != len(tc.ids) {
				t.Fatalf("rows = %+v, %v; want ids %v", rows, err, tc.ids)
			}
			for i, row := range rows {
				if row.Ref.SessionID != tc.ids[i] || row.Rank != i+1 || row.Preview != "support grok" {
					t.Errorf("row %d = %+v", i, row)
				}
				src, err := p.Resolve(context.Background(), roots, row.Ref.SessionID)
				if err != nil || src.Ref != row.Ref || src.Metadata["cwd"] != row.Cwd {
					t.Errorf("resolve listed id = %+v, %v", src, err)
				}
			}
		})
	}
	src, err := p.Resolve(context.Background(), roots, "")
	if err != nil || src.Ref.SessionID != "new" || src.Metadata["title"] != "grok support" || src.Metadata["model"] != "grok-build" || !src.UpdatedAt.Equal(fresh) {
		t.Fatalf("newest source = %+v, %v", src, err)
	}
	if _, err := p.Resolve(context.Background(), roots, "missing"); err == nil {
		t.Fatal("unknown id resolved")
	}
}

func TestVisibility(t *testing.T) {
	for _, tc := range []struct {
		name   string
		fields map[string]any
		listed bool
	}{
		{"normal", map[string]any{"num_messages": 1}, true},
		{"hidden", map[string]any{"hidden": true, "num_messages": 1}, false},
		{"subagent", map[string]any{"session_kind": "subagent", "num_messages": 1}, false},
		{"subagent resume", map[string]any{"session_kind": "subagent_resume", "num_messages": 1}, false},
		{"explicit visible", map[string]any{"session_kind": "subagent", "hidden": false, "num_messages": 1}, true},
		{"husk", map[string]any{}, false},
		{"titled", map[string]any{"generated_title": "title"}, true},
		{"fork", map[string]any{"session_kind": "fork"}, true},
		{"parent", map[string]any{"parent_session_id": "parent"}, true},
		{"fork timestamp", map[string]any{"forked_at": "2026-07-18T00:00:00Z"}, true},
	} {
		t.Run(tc.name, func(t *testing.T) {
			root := t.TempDir()
			writeSession(t, root, "group", "id", tc.fields, acpLog, time.Now())
			p, roots := New(), session.Roots{Grok: root}
			rows, err := p.List(context.Background(), roots, session.ListOptions{})
			if err != nil || (len(rows) == 1) != tc.listed {
				t.Fatalf("rows = %+v, %v; listed = %v", rows, err, tc.listed)
			}
			if _, err := p.Resolve(context.Background(), roots, "id"); err != nil {
				t.Fatalf("explicit id: %v", err)
			}
			_, err = p.Resolve(context.Background(), roots, "")
			if (err == nil) != tc.listed {
				t.Errorf("newest error = %v; listed = %v", err, tc.listed)
			}
		})
	}
}

func TestIndexFallbacks(t *testing.T) {
	for _, tc := range []struct {
		name, title string
		summary     map[string]any
		updates     string
		resolves    bool
	}{
		{"summary title", "summary", map[string]any{"session_summary": "summary", "updated_at": "1970-01-01T00:01:40Z"}, acpLog, true},
		{"cwd title", "proj", map[string]any{"info": map[string]string{"cwd": cwd}, "num_messages": 1}, acpLog, true},
		{"no transcript", "", map[string]any{"num_messages": 1}, "", true},
		{"no summary", "", nil, acpLog, false},
	} {
		t.Run(tc.name, func(t *testing.T) {
			root := t.TempDir()
			writeSession(t, root, "slug-hash", "id", tc.summary, tc.updates, time.Unix(100, 0))
			p, roots := New(), session.Roots{Grok: root}
			src, err := p.Resolve(context.Background(), roots, "id")
			if (err == nil) != tc.resolves {
				t.Fatalf("resolve = %+v, %v", src, err)
			}
			if tc.resolves && (src.Ref.SessionID != "id" || src.Metadata["title"] != tc.title || !src.UpdatedAt.Equal(time.Unix(100, 0))) {
				t.Errorf("source = %+v", src)
			}
			rows, err := p.List(context.Background(), roots, session.ListOptions{})
			if err != nil || (len(rows) == 1) != (tc.resolves && tc.updates != "") {
				t.Errorf("list = %+v, %v", rows, err)
			}
		})
	}
	p, roots := New(), session.Roots{Grok: t.TempDir()}
	if _, err := p.Resolve(context.Background(), roots, ""); err == nil {
		t.Fatal("empty root resolved")
	}
	if rows, err := p.List(context.Background(), roots, session.ListOptions{}); err != nil || len(rows) != 0 {
		t.Fatalf("empty root = %+v, %v", rows, err)
	}
}
