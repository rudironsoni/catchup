// Package deepseek implements session.Provider over DeepSeek Harness (dsh)
// history: per-session JSONL logs under $DSH_HOME/sessions (default
// ~/.dsh/sessions).
//
// Format reference, derived from the dsh session docs, the dsh 0.1.7-rc.2
// packages that write and migrate the logs (@deepseek-ai/dsh-session-format,
// dsh-session-persistence-jsonl, dsh-session-format-v2-to-v3 and -v3-to-v4,
// dsh-subagent), and live installs:
//
//	sessions/--<project>--/<id>/ holds one session; the project directory is a
//	lossy encoding of the working directory, so the header's cwd is used
//	instead. The log is zstd-compressed JSONL by default, one frame per append
//	batch; installs that configure compression off write plain JSONL instead.
//	Both are read here.
//
//	Each format version has its own file: session.jsonl[.zstd] is version 0,
//	session.vN.jsonl[.zstd] version N. Writing to an older session first
//	migrates it to the current version (4 as of 0.1.7-rc.2) in a new file and
//	leaves the old one untouched, so the highest version is the live log. The
//	session.lock lease, migration staging files and manual backups beside it
//	are not logs.
//
// The first line is a session header {"type":"session","version","id","cwd",
// "createdAt","agentPreset"}; every later line is one append-only event
// {type, seq, time, data} with epoch-ms times and contiguous seq. Versions 0
// and 1 store packed chunk runs as single rows (text-chunks/reasoning-chunks
// with seq0 instead of seq); they carry no timeline content and are skipped
// along with every other non-message event. A crashed writer can leave a torn
// final line; reading stops there and keeps the parsed prefix.
//
// A subagent's child session is a session of its own whose header adds origin
// "subagent" and its parentSession. dsh's session list hides it, and so do List
// and Resolve without an id; Resolve by id still finds it. A user's fork also
// records a parentSession, but no origin, so it stays listed.
//
// Visible on the timeline: user/message events whose data.source.kind is
// "user" (other kinds are injected context, not conversation: runtime-context
// sandbox snapshots, skill catalogs, agent instructions, tool modes, and before
// version 4 anything a plugin sourced; a live session is dominated by them)
// and assistant/message text blocks (reasoning blocks are skipped). Tool
// results are tool/result events, never user/message. Compaction seam events
// (type prefixed "compaction/") become compaction markers. Metadata: the
// header's id and cwd; session/title events, last writer wins;
// request/header's data.header.config {provider,model} plus each assistant
// message's data.message.source {provider,model}.
//
// Unverified: no local install had run compaction, so the compaction/* shapes
// and the replacement user/message rows the compaction plugins append (a
// non-"append" surfaceOp) come from the docs alone; replacements are skipped
// rather than guessed at, leaving the pre-compaction history visible.
package deepseek

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"io/fs"
	"os"
	"path/filepath"
	"regexp"
	"sort"
	"strconv"
	"strings"
	"time"

	"github.com/klauspost/compress/zstd"

	"github.com/wilbeibi/catchup/internal/session"
)

// Provider reads dsh session logs. It is stateless; every call re-reads the
// log files, so a concurrently writing dsh is never blocked.
type Provider struct{}

// New returns a DeepSeek Harness provider.
func New() *Provider { return &Provider{} }

var _ session.Provider = (*Provider)(nil)

func (p *Provider) Resolve(ctx context.Context, roots session.Roots, id string) (session.Source, error) {
	files, err := sessionFiles(roots.DeepSeek)
	if err != nil {
		return session.Source{}, err
	}
	if len(files) == 0 {
		return session.Source{}, fmt.Errorf("deepseek: no sessions found under %s", roots.DeepSeek)
	}
	if id != "" {
		for _, fi := range files {
			if fi.id == id {
				return readMeta(fi)
			}
		}
		return session.Source{}, fmt.Errorf("deepseek: no session with id %q", id)
	}
	for _, fi := range files {
		if src, err := readMeta(fi); err != nil || !isSubagent(src) {
			return src, err
		}
	}
	return session.Source{}, fmt.Errorf("deepseek: no sessions found under %s", roots.DeepSeek)
}

func (p *Provider) Read(ctx context.Context, src session.Source) (session.Thread, error) {
	if src.Path == "" {
		return session.Thread{}, errors.New("deepseek: source has no path")
	}
	info, err := os.Stat(src.Path)
	if err != nil {
		return session.Thread{}, err
	}
	return readThread(fileInfo{path: src.Path, mod: info.ModTime(), id: src.Ref.SessionID})
}

func (p *Provider) List(ctx context.Context, roots session.Roots, opts session.ListOptions) ([]session.Summary, error) {
	files, err := sessionFiles(roots.DeepSeek)
	if err != nil {
		return nil, err
	}
	limit := opts.EffectiveLimit()
	out := make([]session.Summary, 0, limit)
	for _, fi := range files {
		if len(out) >= limit {
			break
		}
		t, err := readThread(fi)
		if err != nil || len(t.Entries) == 0 || isSubagent(t.Source) {
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

// --- file enumeration -------------------------------------------------------

type fileInfo struct {
	path    string
	mod     time.Time
	id      string
	version int
}

// logName is dsh's canonical generation name (CANONICAL_LOG_FILENAME) under
// either compression suffix: session.jsonl is version 0, session.vN.jsonl
// version N. Lock files, migration staging and manual backups never match.
var logName = regexp.MustCompile(`^session(?:\.v([1-9][0-9]*))?\.jsonl(?:\.zstd)?$`)

// sessionFiles returns one log per dsh session under <root>/sessions, newest
// first. A migration leaves the older generation beside the new one, so each
// session directory yields only its highest version, the one dsh reads and
// appends to. The walk matches file names rather than hard-coding the two
// directory levels, so a future nesting change costs nothing. The session id is
// its directory's base name.
func sessionFiles(root string) ([]fileInfo, error) {
	dir := filepath.Join(root, "sessions")
	if _, err := os.Stat(dir); errors.Is(err, fs.ErrNotExist) {
		return nil, nil
	}
	var files []fileInfo
	seen := map[string]int{} // session directory -> its index in files
	err := filepath.WalkDir(dir, func(p string, d fs.DirEntry, err error) error {
		if err != nil || d.IsDir() {
			return nil
		}
		m := logName.FindStringSubmatch(d.Name())
		if m == nil {
			return nil
		}
		version, _ := strconv.Atoi(m[1]) // v0's name has no number: 0
		sessionDir := filepath.Dir(p)
		i, ok := seen[sessionDir]
		if ok && files[i].version >= version {
			return nil
		}
		info, e := d.Info()
		if e != nil {
			return nil
		}
		fi := fileInfo{path: p, mod: info.ModTime(), id: filepath.Base(sessionDir), version: version}
		if ok {
			files[i] = fi
		} else {
			seen[sessionDir] = len(files)
			files = append(files, fi)
		}
		return nil
	})
	sort.Slice(files, func(i, j int) bool { return files[i].mod.After(files[j].mod) })
	return files, err
}

// isSubagent reports a subagent's child session, which dsh's own session list
// hides: its header's origin is "subagent". A parentSession alone does not
// mark one, since a user's fork of a session records its parent too.
func isSubagent(src session.Source) bool {
	return src.Metadata["relationship"] == "child"
}

// readMeta delegates instead of doing a metadata-only scan: dsh keeps the
// title and model in mid-file events, so a "lighter" read would parse the
// whole file anyway. Logs are small (a long session stays under a few MB).
func readMeta(fi fileInfo) (session.Source, error) {
	t, err := readThread(fi)
	return t.Source, err
}

// --- parsing ----------------------------------------------------------------

// The envelope decodes only fields present on (nearly) every line; data is
// kept raw and handed to per-type shapes in applyLine. A fat struct covering
// every event's fields cannot work here: unrelated events reuse field names
// with different shapes (session/title's source.model is an object describing
// the title model, while a message source's model is a string), and one
// mismatch would fail the whole line's decode.
type dshLine struct {
	Type      string          `json:"type"`
	ID        string          `json:"id"`  // session header only
	Cwd       string          `json:"cwd"` // session header only
	Time      int64           `json:"time"`
	SurfaceOp json.RawMessage `json:"surfaceOp"`
	Data      json.RawMessage `json:"data"`

	// Session header only, set on a subagent's child session.
	Origin        string `json:"origin"`
	ParentSession string `json:"parentSession"`
}

// Per-event data shapes; decode tolerance is applyLine's contract.
type dshUserMessage struct {
	Content []dshBlock `json:"content"`
	Source  struct {
		Kind string `json:"kind"`
	} `json:"source"`
}

type dshAssistantMessage struct {
	Message struct {
		Content []dshBlock `json:"content"`
		Source  struct {
			Provider string `json:"provider"`
			Model    string `json:"model"`
		} `json:"source"`
	} `json:"message"`
}

type dshBlock struct {
	Type string `json:"type"`
	Text string `json:"text"`
}

func readThread(fi fileInfo) (session.Thread, error) {
	f, err := os.Open(fi.path)
	if err != nil {
		return session.Thread{}, err
	}
	defer f.Close()

	var r io.Reader = f
	if strings.HasSuffix(fi.path, ".zstd") {
		zr, err := zstd.NewReader(f)
		if err != nil {
			return session.Thread{}, fmt.Errorf("deepseek: open %s: %w", fi.path, err)
		}
		defer zr.Close()
		r = zr
	}

	src := session.Source{
		Ref:       session.Ref{Provider: session.ProviderDeepSeek, SessionID: fi.id},
		Path:      fi.path,
		UpdatedAt: fi.mod,
		Metadata:  map[string]string{},
	}

	var entries []session.Entry
	var warnings []string
	var unknown session.UnknownTypes
	dec := json.NewDecoder(r)
	first := true
	for dec.More() {
		var line dshLine
		if err := dec.Decode(&line); err != nil {
			warnings = append(warnings, session.ReadStopWarning(err))
			break
		}
		if first {
			first = false
			if line.Type == "session" {
				if line.ID != "" {
					src.Ref.SessionID = line.ID
				}
				if line.Cwd != "" {
					src.Metadata["cwd"] = line.Cwd
				}
				if line.Origin == "subagent" {
					src.Metadata["parent"], src.Metadata["relationship"] = line.ParentSession, "child"
				}
				continue
			}
		}
		applyLine(&src, &entries, &unknown, line)
	}
	if src.Metadata["title"] == "" {
		if cwd := src.Metadata["cwd"]; cwd != "" {
			src.Metadata["title"] = filepath.Base(cwd)
		}
	}
	return session.Thread{Source: src, Entries: entries, Warnings: unknown.AppendTo(warnings)}, nil
}

// applyLine folds one event into the source metadata or the timeline. Events
// are processed in file order, so "last writer wins" fields (title, model)
// land on the final value naturally. Each data shape is decoded
// independently; an unparseable or absent shape makes that event contribute
// nothing rather than fail the line.
func applyLine(src *session.Source, entries *[]session.Entry, unknown *session.UnknownTypes, line dshLine) {
	switch {
	case line.Type == "user/message":
		// Human turns only; injected context and unverified compaction
		// replacements are skipped (see package doc).
		var d dshUserMessage
		if json.Unmarshal(line.Data, &d) != nil {
			return
		}
		if d.Source.Kind != "user" || !isAppend(line.SurfaceOp) {
			return
		}
		if txt := blocksText(d.Content); txt != "" {
			*entries = append(*entries, session.Entry{
				Kind: session.KindMessage, Role: session.RoleUser,
				Text: txt, Time: msToTime(line.Time),
			})
		}
	case line.Type == "assistant/message":
		var d dshAssistantMessage
		if json.Unmarshal(line.Data, &d) != nil || !isAppend(line.SurfaceOp) {
			return
		}
		if txt := blocksText(d.Message.Content); txt != "" {
			*entries = append(*entries, session.Entry{
				Kind: session.KindMessage, Role: session.RoleAssistant,
				Text: txt, Time: msToTime(line.Time),
			})
		}
		if s := d.Message.Source; s.Model != "" || s.Provider != "" {
			setModel(src, s.Model, s.Provider)
		}
	case line.Type == "session/title":
		var d struct {
			Title string `json:"title"`
		}
		if json.Unmarshal(line.Data, &d) == nil && d.Title != "" {
			src.Metadata["title"] = d.Title
		}
	case line.Type == "request/header":
		var d struct {
			Header struct {
				Config struct {
					Provider string `json:"provider"`
					Model    string `json:"model"`
				} `json:"config"`
			} `json:"header"`
		}
		if json.Unmarshal(line.Data, &d) == nil {
			if c := d.Header.Config; c.Model != "" || c.Provider != "" {
				setModel(src, c.Model, c.Provider)
			}
		}
	case strings.HasPrefix(line.Type, "compaction/"):
		var d struct {
			Summary string `json:"summary"`
		}
		json.Unmarshal(line.Data, &d)
		*entries = append(*entries, session.Entry{
			Kind: session.KindCompact, Text: d.Summary, Time: msToTime(line.Time),
		})
	case dshIgnored[line.Type]:
	default:
		unknown.Add(line.Type)
	}
}

// dshIgnored names every other event dsh writes: the 0.1.7-rc.2 vocabulary
// (KNOWN_SESSION_EVENT_TYPES) and the rows only older logs hold. None of it is
// conversation: the system prompt and developer notices, the streamed halves
// of the messages already read, the tools, hooks and commands the agent ran
// and the files they touched, the turn and step frames around them, the
// session's own settings, approvals, feedback and plugin state, and the model
// requests and log uploads the CLI makes for itself. Naming them is what lets
// an event dsh grows later announce itself.
var dshIgnored = map[string]bool{
	"assistant/chunk": true, "text-chunks": true, "reasoning-chunks": true, "tool-call-chunks": true,
	"assistant/attempt": true, "system/message": true, "developer/message": true,

	"tool/call": true, "tool/result": true, "tool/code-dispatch": true,
	"tool/code-dispatch-start": true, "tool/ptc-dispatch": true, "tool/ptc-dispatch-start": true,
	"tool-workflow/run-start": true, "tool-workflow/run-end": true,
	"tool-workflow/agent-start": true, "tool-workflow/agent-end": true,
	"command/run": true, "command/done": true, "todo/write": true, "hook/invoked": true, "hook/result": true,
	"image/offload": true, "workspace/changes": true, "deliverables/presented": true,

	"turn/start": true, "turn/end": true, "step/start": true, "step/end": true,

	"session": true, "session/end-seed": true, "sandbox/mode": true, "approval/policy": true,
	"approval/asked": true, "approval/decided": true, "permission/preset": true, "model/selection": true,
	"agent-preset/selected": true, "subagent/descriptor": true, "agent/inbox/spliced": true,
	"subagent/catalog": true, "subagent/model-selection-policy": true, "plan/mode": true,
	"goal/change": true, "schedule/change": true, "team/member": true, "team/task": true,
	"team/message/queued": true, "team/message/delivered": true,
	"feedback/record": true, "feedback/message-put": true, "feedback/message-delete": true,

	"request/context": true, "llm/retry": true, "llm/retry-started": true,
	"session/title-llm-request": true, "web/deepseek-search-llm-request": true,
	"session-log-deepseek/delivery-accepted": true,
}

// isAppend reports whether an event joined the surface as a plain append.
// Absent counts as append: the badge is required on surface events, so its
// absence means an older or synthetic row, not a replacement.
func isAppend(op json.RawMessage) bool {
	return len(op) == 0 || string(op) == `"append"`
}

// setModel keeps the last writer. request/header (per model request) and the
// assistant message sources agree in practice; when they do not, the later
// event is the model that actually answered the later turns.
func setModel(src *session.Source, model, provider string) {
	if model != "" {
		src.Metadata["model"] = model
	}
	if provider != "" {
		src.Metadata["model_provider"] = provider
	}
}

// blocksText joins the text blocks of a content list.
func blocksText(blocks []dshBlock) string {
	var parts []string
	for _, b := range blocks {
		if b.Type == "text" && b.Text != "" {
			parts = append(parts, b.Text)
		}
	}
	return strings.Join(parts, "\n")
}

func msToTime(ms int64) time.Time {
	if ms == 0 {
		return time.Time{}
	}
	return time.UnixMilli(ms)
}
