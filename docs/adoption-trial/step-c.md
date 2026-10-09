# Step C adoption trial: discovery text, A/B (#126)

Fixed on 2026-10-09, **before** any Step C session ran. Like [Step B](protocol.md), git history
is the evidence that the arms, the answer key and the decision rules below were not adjusted
after results arrived.

## Question

Step B found adoption under 20% for both agents (Claude Code 2/40 sessions, Codex 1/40), and
its rule for that outcome says the next change is to discovery. Does leading the server's
instructions and `search_docs` description with *when* to search raise adoption, without
hurting answers?

## Why the text, and nothing else

- Claude Code loads server instructions and tool *names* at session start; tool schemas,
  descriptions included, are deferred to tool search
  ([context window](https://code.claude.com/docs/en/context-window)). Before choosing a tool it
  sees our instructions, not our descriptions.
- Codex reads `instructions` as server-wide guidance and advises keeping "the first 512
  characters self-contained so the most important guidance is available when Codex is deciding
  how to use the server" ([MCP](https://learn.chatgpt.com/docs/extend/mcp)).
- 0.7.6's instructions are 1,473 characters. The first 512 explain indexing; the one sentence
  on when to use the server, "Prefer these tools over reading whole Markdown files.", starts at
  character 1,420. `search_docs`' description opens with its implementation ("Hybrid search
  (BM25 keywords + semantic vectors, fused with RRF)").
- MCP tool annotations are hints a client must not base tool choice on from an untrusted server,
  so they are not a discovery lever here.

## Arms

| | A (control) | B (treatment) |
| --- | --- | --- |
| Source | `main` at the merge of this protocol's PR; `src/` equals 0.7.6 (`git diff v0.7.6 -- src/` is empty) | `feat/126-discovery-text` (#131) at `ac32e1d`: A's `src/` plus the text change only |
| Instructions | 0.7.6, unchanged | A 463-character lead first (below); the closing "Prefer these tools over reading whole Markdown files." removed |
| `search_docs` description | 0.7.6, unchanged | First sentence: "Search this project's Markdown documentation for the best-matching section - call it first for any question the docs might answer, before grep or reading .md files." The rest unchanged |

B's lead:

> Search this project's Markdown documentation before grepping or reading .md files:
> search_docs(query) returns the best-matching section - or an excerpt of the passage that
> matched - with pointers to the next best, and read_section fetches any one of them by heading.
> Use it for every question the documentation might answer, with exact identifiers (flags, env
> vars, config keys) or plain words; for an identifier, it also says when no indexed section
> contains it.

B is one treatment, the lead and the description together; a result cannot be attributed to
either half alone. B is not merged to `main` before the trial; it merges only if the ship rule
below passes.

## Frozen setup

| Item | Value |
| --- | --- |
| Server | Each arm from its own worktree: `uv run --directory <worktree> markdown-memory`, default `embeddinggemma` preset |
| Index | One database per arm and repository (`MARKDOWN_MEMORY_DB`), built before the run; coverage must read `verified`. `MARKDOWN_MEMORY_AUTO_INDEX=0` in every session, so none starts an index run |
| Recorded | Each worktree's commit and `uv.lock` hash, the model files' SHA-256, each agent's version and resolved model |
| Agents | Claude Code 2.1.294, Codex CLI 0.161.0, each with its provider default model (Step B ran 2.1.295 and 0.160.1) |
| Claude | `claude -p --output-format stream-json --verbose --setting-sources project --strict-mcp-config --mcp-config <arm> --permission-mode default`; allowed: the four retrieval tools, `Read`, `Grep`, `Glob`, and `Bash` for `rg`, `grep`, `sed -n`, `head`, `cat`, `wc`, `ls`, `find`; denied: `index_directory`, `Edit`, `Write`, `NotebookEdit` |
| Codex | `codex exec --json --ignore-user-config -s read-only`, the arm's server through `-c` with `required = true` (a server that fails to start fails the session) and `approval_mode = "approve"` for the four retrieval tools only |
| Repositories | astral-sh/uv `f3e56e1` ([tasks](tasks-uv.json)), vitejs/vite `7a89794` ([tasks](tasks-vite.json)); the private repository is not used |
| Tasks and brief | Step B's 40 public tasks and its session brief, which never names the server |
| Order | Shuffled with seed `43` per repository. Each task runs in both arms back to back, with both agents; the arm that runs first alternates from one task to the next. At most 4 sessions at once |
| Budget | 10 minutes per session |
| Harness | [`scripts/adoption_trial.py`](../../scripts/adoption_trial.py) |

160 sessions: 40 tasks x 2 agents x 2 arms, which is 80 pairs (task x agent).

Covariates the same in both arms and not controlled: `--ignore-user-config` skips only Codex's
`config.toml`, so the operator's global `~/.codex/AGENTS.md` still loads (it names other MCP
servers, none of them mounted); uv's checkout carries its own `AGENTS.md` and `.codex/`
hooks; `--setting-sources project` still loads a repository's own Claude settings (neither
repository has any).

## Checks

- **Probe:** before the run, one session per agent x arm in a neutral one-file repository, told
  to call `search_docs`. Each must succeed, and its transcript must yield a paired
  `search_docs` call with the session's id and directory.
- **Availability:** a Claude session whose `init` does not show `markdown-memory` connected as
  the only MCP server, with no plugin, has failed. Codex fails the session itself when the
  server does not start (`required = true`).
- **Failed session:** a non-zero exit, no final answer, a failed availability check, or the
  budget exceeded. The whole pair (both arms) runs once more; if either arm fails again, the
  pair is dropped from both arms, so the arms always share a denominator. Dropped pairs are
  counted. More than 10% of pairs dropped stops the analysis: redesign.
- **Completeness:** the report refuses a trial in which any of the 80 pairs of the frozen task
  sets never ran, and reads each task's answerability from those files, never from the grades.
- **Edit check:** every tool is read-only, so after each session
  `git status --porcelain --untracked-files=no` on the clone must be empty; any tracked change
  stops the whole run for investigation.

## Measures

Per agent x arm, over the kept pairs:

| Measure | Definition |
| --- | --- |
| **Adoption** (primary) | Share of sessions with at least one retrieval call (`search_docs`, `read_section`, `get_document_outline`, `list_documents`) |
| Calls per adopting session | Retrieval calls / adopting sessions (a mean) |
| Same-file fallback | Searches followed by a read of a file the search returned / searches |
| Reformulation | The number of reformulation chains |
| Documentation exposure | Median estimated tokens per documentation-seeking session, and the shell channel (#129) beside it |
| Answer accuracy | Answerable tasks: correct / partial / wrong against the key |
| Abstention | No-answer tasks: abstained (correct) / answered as if documented |

Transcripts are read by [`usage_from_transcripts.py`](../../scripts/usage_from_transcripts.py):
Claude's stream-json with the session id and the `init` directory added to each record (the
parser reads `sessionId` and a per-record `cwd`), Codex's events converted (shell commands
become `Bash` calls with their output, MCP calls become `mcp__<server>__<tool>` with their
result). Sizes are characters/4 estimates of context exposure, not billed tokens.

## Answer key and grading

- The key, [`key-uv.json`](key-uv.json) and [`key-vite.json`](key-vite.json), is fixed in the
  same commit as this protocol. It was built from each repository's Markdown at the frozen
  commit by `git grep` and reading, never through `markdown-memory`. Each answerable task has
  the section(s) that answer it and a one-sentence reference answer; each no-answer task has
  the `git grep` commands that came back empty (or with unrelated hits, quoted).
- The final answer is Claude's last `result` and Codex's last `agent_message`.
- Answers are graded under shuffled ids, without the arm or the tool calls, against the key:
  **correct** (states the key's point), **partial** (relevant, but misses or hedges it),
  **wrong** (contradicts it, or says the docs do not cover it); for no-answer tasks,
  **abstained** (says the docs do not cover it) or **answered** (presents it as documented).
- Codex and Gemini (agy CLI) grade independently. A disagreement takes the lower grade unless
  the key's cited text settles it; every disagreement and its resolution is listed.

## Decision rules

Per agent, pooled over uv and vite, over the kept pairs:

- **Effect:** an exact one-sided sign test (McNemar) on the discordant pairs: b pairs adopted
  in B only, c in A only, n = b + c; b, c, n and p are reported. Two agents are tested, so the
  effect holds only at p < 0.025 (Bonferroni), keeping the chance of a false release at or below
  5%. Exactly: n = 6 needs b = 6; n = 10 needs b >= 9; n = 20 needs b >= 15; n <= 5 never holds.
- **Sufficiency** (descriptive): B >= 50% means the instructions are enough for that agent;
  B < 20% means they are not, and the next lever is a navigation rule in the repository's own
  agent files (a new issue).
- **Harm guard:** only `correct` counts as correct, never `partial`. It fires for an agent if
  B's share of correct answers on the kept answerable pairs is 15 points or more below A's, or
  if B abstains correctly on 2 or more fewer of the kept no-answer pairs than A.
- **Ship rule:** B merges and 0.7.7 is released only if the effect holds for at least one agent
  and the harm guard fires for neither. Otherwise B is closed unmerged, and the results say
  which rule stopped it.
- **Exposure** is descriptive only; no difference is called a saving.
- **Step B** numbers are set beside B's for reference, with the agent-version drift stated; the
  A/B decides.

## Reporting

Results go to #126 as a comment: per agent, both arms' measures, the sign test, the harm guard
and the verdict; exclusions and dropped pairs; the probes; grader agreement and every
disagreement.
