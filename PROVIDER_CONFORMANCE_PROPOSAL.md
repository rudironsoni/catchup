# Proposal: shared provider behavior checks

Status: draft for maintainer discussion. This proposal adds no framework,
provider changes, or CI gates. Provider-local tests remain the default.

## Problem

A provider can pass tests built around a lossy transcript while omitting
failures or pre-compaction history available in another file. Grok support in
[PR #23](https://github.com/wilbeibi/catchup/pull/23) exposed a second example: a checkpoint fixture with one summary record
missed the real format's two `compaction_meta` records, so the parser returned
injected context without the actual summary. A generic non-empty-entry check
would not have caught that defect either.

The framework removed from #23 added 517 lines of test support and provider
tests, plus a 25-line repository-wide PR template. Its gates searched Go source
for helper calls and package comments for phrases. Those checks did not prove
that a provider read the right file or preserved its format's meaning.

## Options

| Approach | Benefit | Maintenance cost |
| --- | --- | --- |
| Extend existing provider tests | Direct expected timelines from real formats; no new shared mechanism | Repeats a few list/resolve assertions |
| Small opt-in behavior helper | Reuses repeated list/resolve/read assertions where fixtures already exist | Adds a shared test dependency and requires evidence it catches distinct defects |
| Mandatory framework and source-scanning gates | Enforces a uniform convention | More fixtures and registries; helper-call or comment checks can pass without protecting behavior |

Recommend the first approach now. Consider the second only after identifying
repeated, unprotected behavior that cannot be covered economically by existing
tables. Do not restore the third approach.

## Acceptance criteria for a later implementation

- Exercise `session.Provider` through List, Resolve and Read. Never inspect
  source syntax or require wording in comments.
- Reuse existing fixtures. Pin expected session IDs and observable outcomes
  independently of parser output; do not derive a query oracle from that output.
- Keep format-specific timestamps, failure details, compaction and visibility
  expectations in provider tests. Missing fixture coverage is not evidence that
  an agent lacks a capability.
- Give each contract one test owner. A shared check must replace equivalent
  assertions or catch a demonstrated omission; it must not add another layer
  over the same input and expectation.
- Demonstrate each check failing under the relevant behavior change. Report
  added and removed lines separately for test support, fixtures and docs.
- Keep runtime providers unchanged and obtain maintainer agreement on the
  measured benefit before expanding the helper across the repository.

## Decision requested

Is there a recurring List/Resolve/Read contract worth sharing after the focused
Grok tests land? If so, select that contract and two existing fixtures for a
small follow-up. The Grok provider does not depend on accepting this proposal.
