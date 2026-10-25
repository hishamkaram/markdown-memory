# Step D adoption trial: a navigation rule in AGENTS.md, A/B (#134)

Fixed on 2026-10-25, **before** any Step D session ran. As with [Step B](protocol.md) and
[Step C](step-c.md), git history is the evidence that the arms, the rule, the checks and the
decision rules below were not adjusted after results arrived. A change after this commit is an
amendment, dated and committed before the run it governs.

## Question

With the server registered and its current (0.7.11) discovery text, does one line in the
repository's own `AGENTS.md` that names `search_docs` raise a default Codex's use of the server on
documentation questions, without hurting answers?

## Why this lever, and what Step C showed

Step C held everything but the server's text constant. Claude Code went from 32/40 to 40/40, and
Codex stayed at 0/40 in both arms. Codex's own session records (its rollouts under
`$CODEX_HOME/sessions`, which hold the messages the model was given and every tool call and
output) show why. These facts were measured over all 80 Step C Codex rollouts, picked out as the
sessions whose working directory was a trial clone and whose input held the trial brief:

- **Model and mode:** every session ran `gpt-6.1-sol` in code mode. Each tool call is one
  JavaScript `exec`. MCP tools are nested inside it, and the model finds them only by reading
  `ALL_TOOLS`, a list of `{name, description}` entries. Each markdown-memory entry's description
  begins with the server's instructions.
- **Source:** the code-mode `exec` description names helpers, not tools
  (`codex-rs/code-mode-protocol/src/description.rs`). At rust-v0.161.0, code mode leaves out the
  `tool_search` tool (`codex-rs/tools/src/code_mode.rs`; openai/codex#32101).
- **The operator's instructions decided the search.** The operator's global `~/.codex/AGENTS.md`
  was in context in **80/80** sessions. **76/80** sessions began by filtering `ALL_TOOLS`, and
  all 76 filtered for the codebase-memory-mcp names that file lists. That server was not mounted.
- **Our tools barely surfaced.** **2/80** sessions got markdown-memory tools in a tool output,
  both in arm A, and neither called one. **No arm-B session's messages or tool output holds
  arm B's text.** As far as the rollouts show, Step C never delivered its treatment to Codex.
  The tool spec sent on the wire is not logged, so that is the strongest claim the records
  support.
- **Step C's Codex was not isolated.** `--ignore-user-config` skips only `config.toml`
  ([non-interactive mode](https://learn.chatgpt.com/docs/non-interactive-mode)). The model also
  saw 53 skills from the operator's skill directories, and 14 `mcp__codex_apps__*` account tools
  appear in one session's listing.

So in this Codex, the model's instructions decide what it looks for among its tools. A
repository rule that names `search_docs` is the plausible lever. Seeing the tools is not enough:
both sessions that saw them did not use them.

Codex reads `$CODEX_HOME/AGENTS.md`, then one `AGENTS.md` per directory from the git root down to
the working directory, concatenated up to 32 KiB
([AGENTS.md](https://learn.chatgpt.com/docs/agent-configuration/agents-md)).

## Arms

| | A (control) | B (treatment) |
| --- | --- | --- |
| Clone | astral-sh/uv `f3e56e1`, vitejs/vite `7a89794`, as they are | a second clone of each, with one commit on top that adds the rule |
| uv `AGENTS.md` (3,505 bytes, a flat bullet list) | unchanged | the rule appended as its final line |
| vite `AGENTS.md` | absent | created, holding only the rule |
| Index | built from A's clone | built from B's clone |

Both arms use the same server (a worktree of `v0.7.11`), the same agent builds, the same brief and
the same Codex home. The rule's own line is indexed in arm B: it is part of the treatment, as it
would be in a repository that adopted it. Effects on answers therefore cannot be attributed to the
instructions alone.

**The rule** (frozen; the harness holds it as `RULE` and checks it verbatim):

> - For questions about this repository's documentation, call the `search_docs` tool (markdown-memory MCP server) before grepping or reading Markdown files; use `read_section` only when the text it returns is not enough.

It names the tool, because the facts above say the name is what the model searches for. It says
*when* to call it, and it matches the server's own advice on `read_section`.

## Codex in a home of its own

The maintainer chose this setup over the operator's own configuration. It is the same in both arms.

- Codex runs with `CODEX_HOME` and `HOME` both set to a dedicated trial home, outside every
  repository. Setting `HOME` keeps `$HOME/.agents/skills` from loading. The trial home holds only
  its own `auth.json`, from a separate
  `CODEX_HOME=<trial home> codex login --device-auth -c cli_auth_credentials_store="file"`.
  The operator's `~/.codex` and its login are never read, copied or touched.
- Every session also gets `-c features.apps=false -c features.plugins=false
  -c features.hooks=false -c cli_auth_credentials_store="file"`, and keeps
  `--ignore-user-config`.
- The model is pinned with `-c model=...`, and the effort too when it is not the default.
- The MCP server's own environment gets the real `HOME` back, because uv and the model cache live
  there.
- The environment is built for each session alone. The harness never changes its own.
- Claude runs as in Step C, with `--setting-sources project --strict-mcp-config`, plus
  `DISABLE_AUTOUPDATER=1`.

## Checks

A rollout cannot show that a hook ran, or that a tool was loaded and never called. Codex
rust-v0.163.0 does not persist `HookStarted` or `HookCompleted` (`codex-rs/rollout/src/policy.rs`).
So isolation is certified by construction and checked before the run, and the rollouts are read
only for what they can show.

**Before the run.** Each result is recorded in the setup record.
1. **Trial home listing.** The trial home holds only `auth.json` and what Codex writes itself:
   `sessions/`, logs, caches and its own `skills/.system`. There is no `config.toml`,
   `AGENTS.md*`, `hooks.json` or `plugins/`. The listing is checked again after the run.
2. **Effective features.** `codex features list`, under the same environment and overrides,
   reports `apps`, `plugins` and `hooks` as false.
3. **Inventory probes**, one per arm and repository (4), each in that arm's clone:
   `adoption_trial.py probe --inventory`.
   - Codex is told to run exactly `text(JSON.stringify(ALL_TOOLS.map(t => t.name)))` in one
     `exec`.
   - The names are read from that call's own paired output in the rollout. The output must have
     completed.
   - The MCP names must be exactly the server's five tools, with no `codex_apps` and no other
     server.
   - The first probe that passes writes the full list to `--baseline`. The other three must
     list exactly the same tools, native ones included.
   - The claim is scoped to what this proves: no other *enabled* nested tool.
4. **Forced probes**, Step C's, but run in the real clones: 2 agents x 2 arms x 2 repositories.
   Each must make a successful paired `search_docs` call. Every Codex probe must also pass the
   delivery check below.

**In every Codex session: the delivery check.** This is Codex's availability check. A session
that fails it has failed, and its pair runs once more as in Step C. Missing or malformed
evidence fails the check.
- The `thread.started` id maps to exactly one `<trial home>/sessions/**/rollout-*-<id>.jsonl`.
  Every record of it parses, and its one `session_meta` has that id and the arm's clone as its
  working directory.
- Every `turn_context` names the pinned model and reports the pinned effort. A missing `effort`
  field fails. The first inventory probe runs without `--codex-effort`; if its `turn_context`
  reports a concrete effort, that effort is pinned with `--codex-effort` before any other
  session, and the probe runs again.
- The startup records are the base instructions, plus the messages before the one user message
  that equals the brief. Exactly one must equal it.
  - **B:** an `AGENTS.md` message is present and contains the rule verbatim.
  - **A:** no `AGENTS.md` message contains the rule. vite's arm A has no `AGENTS.md` message at
    all.
  - **Both arms:**
    - the startup records hold none of `codebase-memory`, `context7` or `Personal rules`;
    - every skill root listed there, resolved with `expanduser` and then `realpath`, is the
      trial home's `skills/.system` or lies inside the arm's clone.
- What a session read after the brief is not checked: it can quote anything.

Claude's `init` check and the edit check, on each arm's own clone, are Step C's.

## Frozen setup

| Item | Value |
| --- | --- |
| Server | `uv run --directory <v0.7.11 worktree> markdown-memory`, default `embeddinggemma` preset, one database per arm and repository, built and `verified` before the run, `MARKDOWN_MEMORY_AUTO_INDEX=0` |
| Agents | Codex CLI 0.163.0, Claude Code 2.1.301; Codex's model and effort pinned from the first inventory probe, Claude's provider default |
| Tasks, brief, key | Step B's 40 public tasks, its brief, and Step C's answer key ([`key-uv.json`](key-uv.json), [`key-vite.json`](key-vite.json)) |
| Order | Shuffled with seed `43` per repository; both arms back to back, both agents, the arm that runs first alternating; at most 4 sessions at once; 10 minutes per session |
| Failed session | Step C's: rerun the pair once, then drop it from both arms; more than 10% dropped stops the analysis |
| Harness | [`scripts/adoption_trial.py`](../../scripts/adoption_trial.py) `--protocol step-d` |
| Data | Outside the repository, in a directory that survives a reboot. Step C's run directories lived under `/tmp` and were lost; its Codex rollouts survive in the operator's `~/.codex/sessions` |

Recorded in the setup record, with the run aborted on any drift:
- both agents' binaries, versions and resolved models;
- Codex's pinned effort;
- the server commit and `uv.lock`;
- the model files' SHA-256;
- both clones' HEAD;
- the four databases' SHA-256.

160 sessions: 40 tasks x 2 agents x 2 arms, which makes 80 pairs.

## Measures

Per agent x arm, over the kept pairs:

| Measure | Definition |
| --- | --- |
| **Adoption** (primary) | Share of sessions with at least one retrieval call (`search_docs`, `read_section`, `get_document_outline`, `list_documents`) that was answered and did not fail. An empty search still counts. Step C's count, attempts included, is reported beside it |
| Interval | Wilson 95% interval of the adoption share, overall and per repository; the paired difference (B-only minus A-only pairs), overall and per repository |
| Searched first | Sessions whose `search_docs` came before any hand read of Markdown |
| Calls per adopting session | Retrieval calls in the adopting sessions / adopting sessions (Step C divided every session's calls) |
| Fallback, chains, exposure | As in Step C |
| Answer accuracy, abstention | As in Step C |

Described but not judged: whether a Codex session's first `exec` searched `ALL_TOOLS`, and for
what.

## Grading

As in Step C. Answers are graded blind, under shuffled ids, without the arm, the agent or the tool
calls, against the frozen key. Codex and Gemini (agy CLI) grade independently. A disagreement
takes the lower grade unless the key's cited text settles it, and every disagreement is listed.

## Decision rules

Per agent, pooled over uv and vite, over the kept pairs:

- **Effect, for Codex only:** an exact one-sided sign test on the discordant pairs. It holds at
  p < 0.05. One agent is tested, so the family's 5% is not split. Claude is expected at its
  ceiling (40/40 in Step C's arm B), so it gets no effect test. Exactly: n <= 4 never holds,
  n = 5 needs b = 5, n = 10 needs b >= 9, n = 20 needs b >= 15.
- **Harm guard, for both agents:** Step C's. Only `correct` counts as correct. It fires when B's
  share of correct answers on the kept answerable pairs is 15 points or more below A's, or when B
  abstains correctly on 2 or more fewer of the kept no-answer pairs than A.
- **README recommendation:** the rule is recommended for Codex only if Codex's effect holds and the
  harm guard fires for neither agent. The README then states the measured adoption and its
  interval. That the guard did not fire is not written up as "quality unchanged".
- **Sufficiency (descriptive):** Codex at B >= 50% means the rule is enough. Below 20%, a new
  issue is filed for the next lever.
- **Scope of the claim:** a default Codex, on these authored tasks, in these two repositories. It
  is not a measure of real use.

## Reporting

Results go to #134 as a comment:
- per agent, both arms' measures, the sign test, the harm guard and the verdict;
- exclusions and dropped pairs;
- the setup record and all four checks;
- grader agreement and every disagreement;
- Step C's corrected Codex finding, that no arm-B record holds its treatment.

## Amendment 1 (2026-10-25, before any trial session)

Made after the four checks and before the first trial session. Nothing here was chosen from a
trial result: none exists yet.

**Agent versions.** Both agents moved between the freeze and the checks. They are frozen at:

| Agent | Frozen above | Frozen now |
| --- | --- | --- |
| Codex CLI | 0.163.0 | 0.164.0 |
| Claude Code | 2.1.301 | 2.1.310 |

The reason for checking isolation by construction still holds at 0.164.0: rust-v0.164.0
`codex-rs/rollout/src/policy.rs` still lists `HookStarted` and `HookCompleted` among the events it
does not persist.

**Codex's effort.** A session started in the trial home without an effort override writes
`turn_context` records with no `effort` field. The delivery check fails such a session, as it
should, since the field is the evidence. The effort is therefore pinned with `--codex-effort low`.
`low` is `gpt-6.1-sol`'s `default_reasoning_level` in the trial home's `models_cache.json`, so the
pin keeps the default effort and makes Codex record it. This was found with one session in an
empty git repository outside the clones, not with the first inventory probe as frozen above. All
four inventory probes then ran with the pin and passed the delivery check.

**Codex's default features.** With the overrides, `codex features list` reports `apps`, `plugins`,
`hooks` and `memories` false, and `multi_agent` true. `multi_agent` is left at its default. It adds
two developer messages to every session's startup. That is the same in both arms.

**Check results.**
1. Trial home: `auth.json` and Codex's own state (`sessions/`, `log/`, `tmp/`, SQLite state,
   `models_cache.json`, `installation_id`, `shell_snapshots/`), with only `.system` in `skills/`.
   There is no `config.toml`, `AGENTS.md*`, `hooks.json` or `plugins/`.
2. Features: `apps`, `plugins` and `hooks` false (each defaults to true without the overrides).
3. Inventory: all four probes passed and list the same 18 tools: 13 native and the server's five.
   There is no `codex_apps` tool and no other server.
4. Forced probes: 8/8 made one successful `search_docs` call in their arm's clone. The four Codex
   probes passed the delivery check.

**Drift.** The setup record (agent binaries and versions, the model and effort, the server commit
and `uv.lock`, the model files, the four clones' HEAD and the four databases, all by SHA-256) is
taken again before and after each repository's run. Any difference aborts the run. The harness
does not check this itself.
