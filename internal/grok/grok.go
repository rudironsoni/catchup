// Package grok reads Grok Build history under $GROK_HOME (default ~/.grok):
// sessions/<cwd-group>/<id>/{summary.json,updates.jsonl}.
// IDs and cwd come from summary.json, not the percent-encoded or hashed group.
//
// Format evidence: real sessions inspected with grok 1.0.41 (4220f3b224a6),
// 2026-09-27; sanitized excerpts and their provenance are in grok_test.go.
// Listing predicates follow xai-org/grok-build at
// f0e3be1100ef5252488e3be8bb0e91cf68d8c305, crates/codegen/xai-grok-shell/src/
// session/persistence.rs (Summary::is_hidden/is_unused_optimistic_husk).
// Compatibility with other versions has not been established.
//
// updates.jsonl is the authoritative ACP timeline, preserving timestamps,
// failures, stop reasons and history across compactions; chat_history.jsonl is
// a cache derived from it (upstream storage/mod.rs) and is not read. Assistant
// chunks join until a user turn, tool call or turn completion.
// hideFromScrollback excludes injected user chunks; promptIndex alone does not.
// Unknown kinds warn, while known reasoning/tooling/UI updates are explicitly
// ignored below.
//
// Each compaction_checkpoint references a file under compaction_checkpoints/
// whose compaction_meta records, after the environment prefix, form the recap.
// Hidden sessions and unused husks stay out of listings; --id still reads them.
package grok

import (
	"bufio"
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io/fs"
	"os"
	"path/filepath"
	"sort"
	"strings"
	"time"

	"github.com/wilbeibi/catchup/internal/session"
)

// Provider reads Grok session directories. It is stateless; each call re-reads
// the files, so a session Grok is still appending to is never blocked.
type Provider struct{}

// New returns a Grok provider.
func New() *Provider { return &Provider{} }

var _ session.Provider = (*Provider)(nil)

const updatesFile = "updates.jsonl"

// scanLine bounds one updates.jsonl line.
const scanLine = 64 << 20

func (p *Provider) Resolve(ctx context.Context, roots session.Roots, id string) (session.Source, error) {
	infos, err := enumerate(roots.Grok)
	if err != nil {
		return session.Source{}, err
	}
	if id == "" {
		for _, in := range infos {
			if in.visible() {
				return in.source(), nil
			}
		}
		return session.Source{}, fmt.Errorf("grok: no sessions found under %s", roots.Grok)
	}
	for _, in := range infos {
		if in.sum.Info.ID == id || filepath.Base(in.dir) == id {
			return in.source(), nil
		}
	}
	return session.Source{}, fmt.Errorf("grok: no session with id %q", id)
}

func (p *Provider) Read(ctx context.Context, src session.Source) (session.Thread, error) {
	if src.Path == "" {
		return session.Thread{}, errors.New("grok: source has no path")
	}
	info, err := os.Stat(src.Path)
	if err != nil {
		return session.Thread{}, err
	}
	if !info.IsDir() {
		return session.Thread{}, errors.New("grok: source is not a session directory")
	}
	return readUpdates(src, false)
}

func (p *Provider) List(ctx context.Context, roots session.Roots, opts session.ListOptions) ([]session.Summary, error) {
	infos, err := enumerate(roots.Grok)
	if err != nil {
		return nil, err
	}
	limit := opts.EffectiveLimit()
	out := make([]session.Summary, 0, limit)
	for _, in := range infos {
		if len(out) >= limit {
			break
		}
		if !in.visible() {
			continue
		}
		src := in.source()
		if !opts.MatchesCwd(src.Metadata["cwd"]) {
			continue
		}
		// A query must see the whole session, exactly as a read does — a term
		// that only exists before a compaction is still in updates.jsonl. A
		// plain listing needs only a preview, so it stops at the first turn.
		t, err := readUpdates(src, opts.Query == "")
		if err != nil || len(t.Entries) == 0 {
			continue
		}
		if !opts.Matches(t) {
			continue
		}
		out = append(out, opts.Summarize(t))
	}
	for i := range out {
		out[i].Rank = i + 1
	}
	return out, nil
}

// --- session index ----------------------------------------------------------

// grokSummary mirrors the fields of summary.json this provider uses. It is
// deliberately partial: Grok's summary carries many more keys and grows, and an
// unknown key must not fail a parse.
type grokSummary struct {
	Info struct {
		ID  string `json:"id"`
		Cwd string `json:"cwd"`
	} `json:"info"`
	SessionSummary  string `json:"session_summary"`
	GeneratedTitle  string `json:"generated_title"`
	CurrentModelID  string `json:"current_model_id"`
	UpdatedAt       string `json:"updated_at"`
	LastActiveAt    string `json:"last_active_at"`
	SessionKind     string `json:"session_kind"`
	ParentSessionID string `json:"parent_session_id"`
	ForkedAt        string `json:"forked_at"`
	NumMessages     int    `json:"num_messages"`
	Hidden          *bool  `json:"hidden"`
}

// sessionInfo is one located session: its directory, parsed summary, and
// recency.
type sessionInfo struct {
	dir  string
	sum  grokSummary
	when time.Time
}

// visible reports whether the agent's own listing would show this session. It
// mirrors Summary::is_hidden (explicit hidden overrides the subagent default)
// and is_unused_optimistic_husk (no fork provenance, messages or title).
func (in sessionInfo) visible() bool {
	hidden := strings.HasPrefix(in.sum.SessionKind, "subagent")
	if in.sum.Hidden != nil {
		hidden = *in.sum.Hidden
	}
	return !hidden && !in.isHusk()
}

// isHusk mirrors Grok's is_unused_optimistic_husk: a TUI session opened and
// abandoned before it had a title or a message. Fork provenance exempts one.
func (in sessionInfo) isHusk() bool {
	if in.sum.SessionKind == "fork" || in.sum.ParentSessionID != "" || in.sum.ForkedAt != "" {
		return false
	}
	return in.sum.NumMessages == 0 && in.displayTitle() == ""
}

func (in sessionInfo) source() session.Source {
	id := in.sum.Info.ID
	if id == "" {
		id = filepath.Base(in.dir)
	}
	md := map[string]string{}
	if cwd := in.sum.Info.Cwd; cwd != "" {
		md["cwd"] = cwd
	}
	if title := in.title(); title != "" {
		md["title"] = title
	}
	if in.sum.CurrentModelID != "" {
		md["model"] = in.sum.CurrentModelID
	}
	return session.Source{
		Ref:       session.Ref{Provider: session.ProviderGrok, SessionID: id},
		Path:      in.dir,
		UpdatedAt: in.when,
		Metadata:  md,
	}
}

// title prefers the model-generated title, falls back to the session summary,
// then to the directory name — the same fallback every provider's listing uses
// for a session its agent never named.
func (in sessionInfo) title() string {
	if t := in.displayTitle(); t != "" {
		return t
	}
	if cwd := in.sum.Info.Cwd; cwd != "" {
		return filepath.Base(cwd)
	}
	return ""
}

// displayTitle is Grok's Summary::display_title: the generated title, else the
// session summary, with no directory-name fallback.
func (in sessionInfo) displayTitle() string {
	if t := strings.TrimSpace(in.sum.GeneratedTitle); t != "" {
		return t
	}
	return strings.TrimSpace(in.sum.SessionSummary)
}

// enumerate walks every summary.json under <root>/sessions and returns the
// sessions newest-first. A session directory without a readable summary.json is
// skipped: summary.json is the index entry, and Grok itself treats a directory
// without one as an images-only stub rather than a session.
func enumerate(root string) ([]sessionInfo, error) {
	dir := filepath.Join(root, "sessions")
	if _, err := os.Stat(dir); errors.Is(err, fs.ErrNotExist) {
		return nil, nil
	}
	var out []sessionInfo
	err := filepath.WalkDir(dir, func(p string, d fs.DirEntry, err error) error {
		if err != nil || d.IsDir() || d.Name() != "summary.json" {
			return nil
		}
		b, e := os.ReadFile(p)
		if e != nil {
			return nil
		}
		var s grokSummary
		if json.Unmarshal(b, &s) != nil {
			return nil
		}
		out = append(out, sessionInfo{dir: filepath.Dir(p), sum: s, when: summaryTime(s, p)})
		return nil
	})
	if err != nil {
		return nil, err
	}
	// Newest first. last_active_at is Grok's own recency field (it advances only
	// when content is added), so it outranks updated_at, which metadata-only
	// writes also touch; the file mtime is the last resort. The id breaks ties so
	// the order is deterministic.
	sort.SliceStable(out, func(i, j int) bool {
		if !out[i].when.Equal(out[j].when) {
			return out[i].when.After(out[j].when)
		}
		return out[i].sum.Info.ID < out[j].sum.Info.ID
	})
	return out, nil
}

// summaryTime reads a session's recency, falling back from last_active_at to
// updated_at to the summary file's own mtime.
func summaryTime(s grokSummary, summaryPath string) time.Time {
	if t := session.ParseTime(s.LastActiveAt); !t.IsZero() {
		return t
	}
	if t := session.ParseTime(s.UpdatedAt); !t.IsZero() {
		return t
	}
	if info, err := os.Stat(summaryPath); err == nil {
		return info.ModTime()
	}
	return time.Time{}
}

// --- authoritative timeline (updates.jsonl) ---------------------------------

// acpLine is one updates.jsonl record: a JSON-RPC notification whose params
// carry the discriminated update (kept raw and decoded per kind below) and an
// outer _meta with the millisecond wall-clock. Old sessions wrote the params
// object bare, without the method/params envelope (upstream storage/mod.rs).
type acpLine struct {
	Timestamp int64           `json:"timestamp"`
	Update    json.RawMessage `json:"update"`
	Params    struct {
		Update json.RawMessage `json:"update"`
		Meta   struct {
			AgentTimestampMs int64 `json:"agentTimestampMs"`
		} `json:"_meta"`
	} `json:"params"`
}

// acpKind decodes just the discriminator of an update.
type acpKind struct {
	SessionUpdate string `json:"sessionUpdate"`
}

// contentObject is a single content block, the shape a message chunk carries.
type contentObject struct {
	Type string `json:"type"`
	Text string `json:"text"`
}

// The per-kind update shapes. Each is decoded independently from the raw
// update because one field name carries different shapes across kinds (content
// is an object for a message chunk and an array for a tool result), and one
// shared struct would fail a whole line on the first mismatch.
type acpUserChunk struct {
	Content contentObject `json:"content"`
	Meta    struct {
		HideFromScrollback bool `json:"hideFromScrollback"`
		HostTurn           bool `json:"hostTurn"`
	} `json:"_meta"`
}

type acpAssistantChunk struct {
	Content contentObject `json:"content"`
}

type acpToolCall struct {
	ToolCallID string          `json:"toolCallId"`
	Title      string          `json:"title"`
	RawInput   json.RawMessage `json:"rawInput"`
	Meta       struct {
		Tool struct {
			Name string `json:"name"`
		} `json:"x.ai/tool"`
	} `json:"_meta"`
}

type acpToolUpdate struct {
	ToolCallID string          `json:"toolCallId"`
	Status     string          `json:"status"`
	Title      string          `json:"title"`
	Content    json.RawMessage `json:"content"`
	RawOutput  json.RawMessage `json:"rawOutput"`
}

type acpTaskCompleted struct {
	TaskSnapshot struct {
		TaskID   string `json:"task_id"`
		Output   string `json:"output"`
		ExitCode *int   `json:"exit_code"`
	} `json:"task_snapshot"`
}

type acpTurnCompleted struct {
	StopReason string `json:"stop_reason"`
}

type acpCompaction struct {
	CheckpointFile string `json:"checkpoint_file"`
}

// acpIgnored names every updates.jsonl kind that is tooling or UI state rather
// than conversation, so a kind Grok adds later lands in the unknown warning
// instead of being silently dropped.
var acpIgnored = map[string]bool{
	"agent_thought_chunk": true, "hook_execution": true, "hook_annotation": true,
	"plan": true, "goal_updated": true, "retry_state": true,
	"background_tasks": true, "task_backgrounded": true,
	"subagent_spawned": true, "subagent_finished": true,
	"session_recap": true, "auto_compact_started": true, "auto_compact_completed": true,
	"current_mode_update": true, "memory_dream_queued": true, "memory_dream_started": true,
	"memory_dream_completed": true, "image_compressed": true, "model_changed": true,
	"available_commands_update": true, "pending_interaction": true, "interaction_resolved": true,
	"session_summary_generated": true, "tool_call_delta_chunk": true,
}

// readUpdates parses updates.jsonl into the visible timeline, carrying a
// timestamp on every entry. stopAfterUser returns as soon as the first real user
// turn is read, which is all a listing preview needs; a full read, and a queried
// listing, see the whole session. A missing transcript is not an error.
func readUpdates(src session.Source, stopAfterUser bool) (session.Thread, error) {
	f, err := os.Open(filepath.Join(src.Path, updatesFile))
	if errors.Is(err, fs.ErrNotExist) {
		return session.Thread{Source: src}, nil
	}
	if err != nil {
		return session.Thread{}, err
	}
	defer f.Close()

	var entries []session.Entry
	var warnings []string
	var unknown session.UnknownTypes
	calls := map[string]toolCall{}
	answered := map[string]bool{}
	var abuf strings.Builder
	var atime time.Time
	gotUser := false

	flush := func() {
		if abuf.Len() == 0 {
			return
		}
		entries = append(entries, session.Entry{Kind: session.KindMessage, Role: session.RoleAssistant, Text: abuf.String(), Time: atime})
		abuf.Reset()
	}

	sc := bufio.NewScanner(f)
	sc.Buffer(make([]byte, 0, 64*1024), scanLine)
	for sc.Scan() {
		line := bytes.TrimSpace(sc.Bytes())
		if len(line) == 0 {
			continue
		}
		var rec acpLine
		if err := json.Unmarshal(line, &rec); err != nil {
			warnings = append(warnings, session.ReadStopWarning(err))
			break
		}
		raw := rec.Params.Update
		if raw == nil {
			raw = rec.Update
		}
		var kind acpKind
		if json.Unmarshal(raw, &kind) != nil {
			continue
		}
		ts := lineTime(rec.Params.Meta.AgentTimestampMs, rec.Timestamp)

		switch kind.SessionUpdate {
		case "user_message_chunk":
			var u acpUserChunk
			if json.Unmarshal(raw, &u) != nil || u.Content.Type != "text" {
				continue
			}
			// Grok marks injected context it hides from the user's own
			// scrollback (monitor events, system reminders) with
			// hideFromScrollback, and echoes a slash command such as /compact
			// as a hostTurn chunk. Everything else is a real turn: a typed
			// prompt, or a mid-turn interjection whose text is wrapped in
			// <user_query>, which extractUserQuery unwraps.
			if u.Meta.HideFromScrollback || u.Meta.HostTurn {
				continue
			}
			q := extractUserQuery(u.Content.Text)
			if q == "" {
				continue
			}
			flush()
			entries = append(entries, session.Entry{Kind: session.KindMessage, Role: session.RoleUser, Text: q, Time: ts})
			gotUser = true
		case "agent_message_chunk":
			var a acpAssistantChunk
			if json.Unmarshal(raw, &a) != nil || a.Content.Type != "text" || a.Content.Text == "" {
				continue
			}
			if abuf.Len() == 0 {
				atime = ts
			}
			abuf.WriteString(a.Content.Text)
		case "tool_call":
			var tc acpToolCall
			if json.Unmarshal(raw, &tc) != nil || tc.ToolCallID == "" {
				continue
			}
			flush()
			name := tc.Title
			if name == "" {
				name = tc.Meta.Tool.Name
			}
			calls[tc.ToolCallID] = toolCall{name: name, input: tc.RawInput}
		case "tool_call_update":
			var tu acpToolUpdate
			if json.Unmarshal(raw, &tu) != nil || tu.Status != "failed" || tu.ToolCallID == "" || answered[tu.ToolCallID] {
				continue
			}
			answered[tu.ToolCallID] = true
			flush()
			c := calls[tu.ToolCallID]
			name := c.name
			if name == "" {
				name = tu.Title
			}
			entries = append(entries, session.Failure(name, c.input, toolFailureText(tu.Content, tu.RawOutput), ts))
		case "task_completed":
			var t acpTaskCompleted
			if json.Unmarshal(raw, &t) != nil {
				continue
			}
			snap := t.TaskSnapshot
			if snap.ExitCode == nil || *snap.ExitCode == 0 || answered[snap.TaskID] {
				continue
			}
			answered[snap.TaskID] = true
			flush()
			c := calls[snap.TaskID]
			entries = append(entries, session.Failure(c.name, c.input, snap.Output, ts))
		case "turn_completed":
			var t acpTurnCompleted
			if json.Unmarshal(raw, &t) != nil {
				continue
			}
			flush()
			if t.StopReason == "error" {
				entries = append(entries, session.Entry{Kind: session.KindStop, Reason: "error", Time: ts})
			}
		case "compaction_checkpoint":
			var c acpCompaction
			if json.Unmarshal(raw, &c) != nil {
				continue
			}
			flush()
			markRetained(entries)
			entries = append(entries, session.Entry{Kind: session.KindCompact, Text: checkpointSummary(src.Path, c.CheckpointFile), Time: ts})
		default:
			if acpIgnored[kind.SessionUpdate] {
				continue
			}
			unknown.Add(kind.SessionUpdate)
		}
		if stopAfterUser && gotUser {
			break
		}
	}
	if err := sc.Err(); err != nil {
		warnings = append(warnings, session.ReadStopWarning(err))
	}
	flush()
	return session.Thread{Source: src, Entries: entries, Warnings: unknown.AppendTo(warnings)}, nil
}

// lineTime picks the millisecond wall-clock when present, else the enclosing
// line's epoch-second timestamp.
func lineTime(ms, secs int64) time.Time {
	if ms != 0 {
		return time.UnixMilli(ms)
	}
	if secs != 0 {
		return time.Unix(secs, 0)
	}
	return time.Time{}
}

// textArray joins the text of a content block array, the shape tool_call_update
// carries: blocks are {"type":"text","text"} or a {"type":"content",
// "content":{"type":"text","text"}} wrapper.
func textArray(raw json.RawMessage) string {
	if len(raw) == 0 || raw[0] != '[' {
		return ""
	}
	var blocks []struct {
		Type    string `json:"type"`
		Text    string `json:"text"`
		Content *struct {
			Type string `json:"type"`
			Text string `json:"text"`
		} `json:"content"`
	}
	if json.Unmarshal(raw, &blocks) != nil {
		return ""
	}
	var parts []string
	for _, b := range blocks {
		if b.Content != nil && b.Content.Text != "" {
			parts = append(parts, b.Content.Text)
		} else if b.Text != "" {
			parts = append(parts, b.Text)
		}
	}
	return strings.Join(parts, "\n")
}

// toolFailureText reads a failed call's output: the content blocks when there
// are any, else rawOutput, Grok's serialized ToolOutput (e.g.
// {"type":"ListDir","NotFound":...}), kept as JSON.
func toolFailureText(content, rawOutput json.RawMessage) string {
	if s := textArray(content); s != "" {
		return s
	}
	if len(bytes.TrimSpace(rawOutput)) > 0 {
		return string(rawOutput)
	}
	return "tool call failed"
}

// checkpointSummary reads the compaction summary Grok saved for a seam, so
// --since-compact and a bare read show what replaced the context rather than an
// empty marker. It is a best effort: a missing or reshaped file leaves the
// marker bare.
func checkpointSummary(sessionDir, file string) string {
	if file == "" {
		return ""
	}
	b, err := os.ReadFile(filepath.Join(sessionDir, file))
	if err != nil {
		return ""
	}
	var c struct {
		CompactedHistory []struct {
			Type            string          `json:"type"`
			Content         json.RawMessage `json:"content"`
			SyntheticReason *string         `json:"synthetic_reason"`
		} `json:"compacted_history"`
	}
	if json.Unmarshal(b, &c) != nil {
		return ""
	}
	// The first compaction_meta record is the environment prefix (cwd, date)
	// that opens every rebuilt context; the rest are the summary and any
	// image-path note (xai-chat-state build_compacted_history).
	var parts []string
	prefix := true
	for _, it := range c.CompactedHistory {
		if it.Type != "user" || it.SyntheticReason == nil || *it.SyntheticReason != "compaction_meta" {
			continue
		}
		if !prefix {
			parts = append(parts, textArray(it.Content))
		}
		prefix = false
	}
	return strings.Join(parts, "\n")
}

// toolCall is the name and structured input of an assistant tool call.
type toolCall struct {
	name  string
	input json.RawMessage
}

const (
	userQueryOpen  = "<user_query>"
	userQueryClose = "</user_query>"
)

// extractUserQuery returns a real user turn's words. Grok wraps a prompt with
// injected context, the actual request inside <user_query>…</user_query>; when
// the tags are absent the whole text is returned.
func extractUserQuery(text string) string {
	i := strings.Index(text, userQueryOpen)
	if i < 0 {
		return strings.TrimSpace(text)
	}
	rest := text[i+len(userQueryOpen):]
	if j := strings.Index(rest, userQueryClose); j >= 0 {
		return strings.TrimSpace(rest[:j])
	}
	return strings.TrimSpace(rest)
}

// markRetained flags the one message the compaction about to be appended kept
// verbatim. Grok rebuilds the context from its summary and the last real user
// turn, dropping the assistant and tool tail (xai-grok-shell compaction.rs,
// for_compaction). Every other mark is cleared, so only the last compaction
// counts. Unverified: in goal mode Grok keeps the goal objective instead.
func markRetained(entries []session.Entry) {
	kept := false
	for i := len(entries) - 1; i >= 0; i-- {
		user := entries[i].Kind == session.KindMessage && entries[i].Role == session.RoleUser
		entries[i].Retained = user && !kept
		kept = kept || user
	}
}
