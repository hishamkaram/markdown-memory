# Step B adoption trial: simulated variant (#43)

Fixed on 2026-10-09, **before** any trial session ran. Git history is the evidence that the task
sets, metrics and decision rules below were not adjusted after results arrived. Changing anything
here means a commit that says what changed and why, dated before the run it governs.

## Question

When `markdown-memory` is only *registered* (no navigation rule, no hook), do coding agents pick
it up on their own for documentation questions, and what happens to their context exposure and
answers when they do or do not?

## Deviations from #43

This is a simulated variant. It does not replace a real-use trial; it bounds what one would find.

| #43 asks for | This run does | Why |
| --- | --- | --- |
| Real sessions over a fixed duration | 20 authored tasks per repository, each run once per agent as a one-shot headless session | Reproducible, and it runs today; real-use sampling stays open |
| Eligibility defined up front, then sampled | Every task is documentation-seeking by construction, so eligibility is 100% | No classifier error, at the cost of task realism |
| ~100 blind labels with task context | Answerability labelled by the task author **before** any run, verified by `git grep` over the repository's Markdown, never by the system under test | A pool built from the tool's own results would flatter it |

## Frozen setup

| Item | Value |
| --- | --- |
| Server | `markdown-memory` 0.7.6 (`28db21b`), default `embeddinggemma` preset, auto-index on |
| `SERVER_INSTRUCTIONS` | Unchanged from 0.7.6 for the whole run |
| Agents | Claude Code 2.1.295 headless (`--print`, plan/read-only mode, provider default model); Codex CLI 0.160.1 (`exec`, read-only sandbox, provider default model) |
| Dispatch | Delegation Layer 0.1.25, read-only permission, 10-minute budget per session |
| Registration | User-scope MCP registration in both agents; no navigation rule, no hook, no prompt mention of the server |
| Pre-indexing | Each repository is indexed once before the run, so no session pays the first full embed |

Both agents run with the operator's ordinary user-level configuration (other MCP servers,
plugins, hooks). That is a covariate, not a control; it is recorded, and the competing tools
present are listed in the results.

**Repositories**

| Repository | Commit | Tracked `.md` | Task set |
| --- | --- | --- | --- |
| astral-sh/uv | `f3e56e1` | 206 | [`tasks-uv.json`](tasks-uv.json) |
| vitejs/vite | `7a89794` | 84 | [`tasks-vite.json`](tasks-vite.json) |
| Private TypeScript monorepo | (withheld) | 116 | withheld; SHA-256 `d87f321ab08a68d69d77f953dd08290e3a2b7ddc4e35e3c0a9bd64395b378be1` |

The private repository's own agent instructions route code discovery to a separate code-graph MCP
server and text search. That is the competing-tool condition, and its results are reported
separately and in aggregate only.

Each set holds 16 answerable tasks and 4 the repository's Markdown cannot answer (two plausible
but non-existent identifiers, two out-of-scope how-tos; 20%, above the 5-10% of the retrieval
protocol, because abstention is a primary concern here and 2 per repository would say nothing).

## Session brief

Identical for every task except the question; the server is never named:

```text
You are the assigned worker; do not delegate this assignment and do not modify any files.

A developer working in this repository asks: "<question>"

Answer from this repository's own documentation. Cite the file path and section for each claim.
If the documentation does not cover it, say so plainly instead of guessing. Keep the answer under
150 words.
```

Order: tasks are shuffled with seed `43` per repository; the two agents run the same order,
at most 4 sessions concurrently.

## Measures

Extracted with [`scripts/usage_from_transcripts.py`](../../scripts/usage_from_transcripts.py)
at 0.7.6. Claude's `stream-json` output is read as is; Codex's `exec --json` events are first
converted to the same shape (shell commands become `Bash` calls, MCP calls become
`mcp__<server>__<tool>`). All sizes are the script's characters/4 estimates of context exposure,
not billed tokens.

| Measure | Definition |
| --- | --- |
| **Adoption** (primary) | Share of sessions with at least one retrieval call (`search_docs`, `read_section`, `get_document_outline`, `list_documents`) |
| Calls per adopting session | Retrieval calls / adopting sessions |
| Same-file fallback | A whole-file read of a file a search returned, within the script's follow-up window |
| Reformulation | The script's reformulation chains |
| Documentation exposure | Estimated tokens of every documentation-reading result per session: retrieval results and Markdown `Read` (from the script), plus the output of shell commands whose text names `.md` or a docs directory (counted alongside it with the same characters/4 rule, because the script does not size `Bash` results) |
| Answer accuracy | Answerable tasks: correct / partial / wrong against the labelled source section |
| Abstention | No-answer tasks: says the docs do not cover it (correct) vs. answers as if documented (wrong) |

Answers are graded from the final answer text before the grader looks at that session's tool
calls.

## Decision rules

Per agent, pooled over the two public repositories; the private repository is reported beside
them, not pooled, because its competing-tool rule is a different condition.

- **Adoption >= 50%**: registration alone is enough for that agent; a navigation rule is not
  needed for adoption.
- **Adoption < 20%**: registration alone is insufficient for that agent; the next change is to
  discovery (instructions or tool descriptions), not retrieval quality.
- **20-50%**: partial. Reported per repository and task kind; no conclusion beyond that.
- **Harm guard**: if adopting sessions' answer accuracy on answerable tasks is 15pp or more below
  non-adopting sessions' for the same agent, that is reported as a quality regression whatever
  adoption is.
- **Exposure** is descriptive only: median documentation tokens per session, adopting vs not.
  With n <= 20 per cell no exposure difference is called a saving.

`p` values are not computed; with these cell sizes a 95% interval on a share of 40 is about
+/-15pp, and the rules above are set wider than that on purpose.

## Availability check and aborts

- A Claude session whose `init` event does not list `markdown-memory` as connected is excluded
  and counted.
- Codex does not report MCP startup in its event stream. If no Codex session makes a retrieval
  call, one forced probe (a session told to call `search_docs`) runs **after** the trial to
  separate "not adopted" from "not available". It is reported, never pooled.
- Stop and redesign if more than 10% of sessions fail to publish a result, or if any
  repository's index reports coverage other than `verified` at the start.

## Reporting

Results go to #43 as a comment: per agent x repository cells, the measures above, the excluded
sessions, and the forced probe if it ran. The private repository contributes counts and
proportions only: no paths, identifiers, questions or answer text.
