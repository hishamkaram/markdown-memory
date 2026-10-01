# markdown-memory

**Your coding agent reads whole Markdown files to answer one question. This returns the
section that answers it.**

Ask an agent a question about your docs and it opens the files that might answer it, whole.
Most of what lands in its context is about something else, and the part you wanted competes
with it. markdown-memory indexes your documentation by heading, so the same question comes
back as the section that answers it, quoted verbatim, with pointers to the next few.

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="https://raw.githubusercontent.com/hishamkaram/markdown-memory/main/docs/assets/how-it-works-dark.svg">
  <source srcset="https://raw.githubusercontent.com/hishamkaram/markdown-memory/main/docs/assets/how-it-works-light.svg">
  <img src="https://raw.githubusercontent.com/hishamkaram/markdown-memory/main/docs/assets/how-it-works-light.png" width="100%"
       alt="One question asked of four documentation files. Reading them whole costs 16,270
            tokens. markdown-memory splits them at every heading, ranks by keywords and by
            vectors, fuses the two, and returns the section that answers in full, 407
            tokens, with four pointers to the rest.">
</picture>

Measured on this repository's own documentation - `README.md`, `CLAUDE.md`, `AGENTS.md` and
`docs/evaluation-protocol.md`, 16,270 tokens in all:

```
search_docs("where does the embedding model get downloaded")

  407 tok  README.md  markdown-memory > The embedding model > What downloads, when, and where
  712 tok  README.md  markdown-memory > The embedding model > Pre-download it, or install offline
  666 tok  README.md  markdown-memory
  738 tok  CLAUDE.md  markdown-memory > Commands
  383 tok  README.md  markdown-memory > The embedding model > What is checked before the model is loaded
```

**One section in full - 407 tokens - instead of 16,270**: the one that answers, first. The
other four come back as pointers - where each section is, what reading it costs, and the
passage that matched - so when the first is not the answer, one `read_section` fetches the
one that is.

It is a local [Model Context Protocol](https://modelcontextprotocol.io) server - MCP is the
protocol agents use to call tools - and it runs entirely on your machine: parsing with
`markdown-it-py`, embeddings with
[EmbeddingGemma-300m](https://huggingface.co/onnx-community/embeddinggemma-300m-ONNX)
(4-bit ONNX on CPU, 768 dimensions), storage in SQLite - one database per documentation
root, with a keyword index and two vector indexes over it. No API key, no network after the
first model download, nothing leaves the machine.

## Install

### Prerequisites

- **Python 3.11 or newer.** The project develops on 3.12 and CI runs 3.11 through 3.14 on
  x86-64 Linux.
- **[uv](https://docs.astral.sh/uv/)**, which manages the interpreter and the dependencies:
  ```bash
  curl -LsSf https://astral.sh/uv/install.sh | sh     # macOS / Linux
  ```
- **Roughly 1 GB of disk**: ~218 MB for the embedding model, the rest for the index.
- Linux or macOS, x86-64 or arm64. Everything runs on CPU; there is no GPU path and no API
  key. CI covers arm64 Linux on every change and Apple Silicon on `main`, because the 4-bit
  graph picks its `MatMulNBits` kernel from what the CPU offers rather than from the file -
  see [What downloads, when, and where](#what-downloads-when-and-where). Those kernels do
  not all return the same numbers: the gate measures four CPU families and the same text
  embeds up to 1.6e-3 cosine apart between x86-64 and arm64. Copying an index between two
  machines therefore searches vectors from one kernel with queries from another, which
  nothing detects - and which was measured, on the labelled set and through the real
  ranking, to move no top result and lose no answer. It reshuffles positions two to five.

### Get it

```bash
uv tool install markdown-memory        # or: pipx install markdown-memory
markdown-memory --download-model       # optional: fetch the ~218 MB model now
```

`uv tool install` puts the `markdown-memory` command in `~/.local/bin`; if your shell cannot
find it, `uv tool update-shell` adds that directory to your PATH. Fetching the model up
front is optional - the first search does it otherwise - but it keeps a 218 MB download
out of your first question.

### Register it with your client

One registration serves every project. The server indexes the project the client was
started in - Claude Code tells it (`CLAUDE_PROJECT_DIR`), Codex starts it there - and each
project gets its own index.

```bash
claude mcp add --scope user markdown-memory -- markdown-memory    # Claude Code
codex mcp add markdown-memory -- markdown-memory                  # Codex
```

`claude mcp list` and `codex mcp list` should show it. Upgrade with
`uv tool upgrade markdown-memory` (or `pipx upgrade markdown-memory`); remove it with
`claude mcp remove --scope user markdown-memory`, `codex mcp remove markdown-memory` and
`uv tool uninstall markdown-memory` (or `pipx uninstall markdown-memory`).

Desktop apps (Claude Desktop, Cursor) usually do not see `~/.local/bin`, so give them the
absolute path that `command -v markdown-memory` prints. Claude Desktop has no project
either, so name the documentation root:

```json
{
  "mcpServers": {
    "markdown-memory": {
      "command": "/home/you/.local/bin/markdown-memory",
      "env": { "MARKDOWN_MEMORY_DOCS_DIR": "/absolute/path/to/your/docs" }
    }
  }
}
```

`uvx markdown-memory` works in place of the installed command too, with one catch: its
first launch downloads about 250 MB of dependencies before the server can answer, longer
than Codex waits by default - set `startup_timeout_sec = 60` under
`[mcp_servers.markdown-memory]` in `~/.codex/config.toml` if you go that way.

### From source

For working on markdown-memory itself:

```bash
git clone https://github.com/hishamkaram/markdown-memory
cd markdown-memory
uv sync
uv run markdown-memory --version
```

The `.mcp.json` in the clone runs that checkout (`uv run markdown-memory`) whenever Claude
Code is opened in it, with no edit.

### Index once, then search

The server keeps its documentation root indexed by itself. The first `search_docs` (or
`list_documents` without a directory) after it starts begins a catch-up run in the
background - whatever changed while no server ran, new files included - and answers from
the index as it stood, with `index_status.indexing` set. Nothing runs before that first
call: loading the model holds the interpreter for seconds, and a client waiting on the
handshake gives up quickly. After that, each of those calls decides whether another run is
due:

- a file it sees edited, deleted or unreadable starts one, at most every 10 seconds;
- a weights mismatch a search has just recorded starts one at once;
- otherwise one walks the tree every 5 minutes while the server is in use, which is what
  finds a file nobody indexed yet.

One run at a time, in one thread, stopped between two documents when the server shuts down;
a stopped run leaves what a killed run leaves, and the next one resumes. While it runs,
`index_status.indexing` is `true` and the message says so instead of asking for
`index_directory`. Nothing watches the filesystem, so a server nobody is using does not
spend anything; `read_section` and `get_document_outline` do not start a run. The first
run of a large tree embeds everything and competes with queries for CPU until it is done;
`--no-auto-index` (or `MARKDOWN_MEMORY_AUTO_INDEX=0`) turns all of this off, and then:

```
index_directory()                  # scans the docs root, embeds what it finds
list_documents()                   # index_status.coverage should read "verified"
```

The first run downloads the embedding model and then indexes, so it is the slow one - see
[The embedding model](#the-embedding-model). After that, indexing is incremental: re-running
it costs milliseconds when nothing changed. If `coverage` reads `"unknown"`, something was
missed and `index_status.message` says what to run.

`index_status` also carries `changed_files`: indexed documents a cheap probe could not
confirm are still what was indexed - their bytes differ, or they are gone, unreadable, or no
longer a regular file. It is best-effort in both directions. Unreadability is noticed only
where a moved timestamp made it read the file at all: a file whose permissions changed and
whose time did not is answered from the time, and it is the next `index_directory` that
records the failure and takes `coverage` to `"unknown"`. A path that cannot even be
`stat`-ed - its parent directory lost its permissions, say - has no timestamp to compare
and is counted straight away. It counts only rows the index
holds, so a file nobody has indexed yet is not among them: finding those means walking the
tree, which is the expensive half of indexing and not something a search should pay for.
And it reads bytes only where the modification time moved, so an edit that restores a file's
own timestamp is missed - indexing is not fooled by that, since it hashes every file it
walks; what is missed is only the hint that running it is worth it. A zero means nothing was
detected, not that every file was re-hashed. `coverage` stays `"verified"` while the count is
non-zero: the walk really did finish and really did read every file it found. What moved on
is the tree, and the message says so.

And it carries `gitignore`: what git said the last time a walk of the root finished -
`applied`, `off` (switched off), `no_repository` (nothing to ignore), `unknown` (no walk has
finished since the index was built), or `unavailable`: the root is in a repository git could
not be asked about - not installed, timed out, refused by `safe.directory` - so what it
ignores was indexed as well. Only `unavailable` gets a message, and only on a tree that is
otherwise whole; the server log says why.

## The embedding model

### What downloads, when, and where

The first search (or `markdown-memory --download-model`) fetches three files from
[`onnx-community/embeddinggemma-300m-ONNX`](https://huggingface.co/onnx-community/embeddinggemma-300m-ONNX)
at a pinned revision: the 4-bit ONNX graph, its external weights, and the tokenizer.
About **218 MB**, once per machine, into:

```
$XDG_CACHE_HOME/markdown-memory/models/embeddinggemma-300m-onnx-5090578d9565/   # ~/.cache/... by default
```

The revision is part of the folder name, so moving the pin fetches the new weights instead
of serving the old ones under a name that claims to be the new ones. One revision can publish
several graphs, though, so the folder name is not the whole answer: a `.verified` stamp that
does not name exactly the files this version needs - a cache left behind by the int8 graph at
this same revision, say - is rejected and the files are fetched, rather than half-trusted.

The cache is shared by every project on purpose - the weights are identical and read-only,
so copying them per project would be pure waste. Point `MARKDOWN_MEMORY_MODEL_CACHE`
somewhere else to move it.

The graph is `onnx/model_q4.onnx`, published 4-bit: the 262144x768 vocabulary table is
quantized and gathered by a single `GatherBlockQuantized`, and the projections run as
`MatMulNBits`, so the table is never expanded to float32. Nothing is derived or rewritten on
your machine - earlier versions ran the int8 graph and patched it here to get the same
effect. Using Gemma is covered by the
[Gemma Terms of Use](https://ai.google.dev/gemma/terms); this repository distributes no
weights.

### What is checked before the model is loaded

Every file's size and sha256 is pinned at the pinned revision. After a full check, a
`.verified` stamp records each file's size, mtime, ctime, inode and device, so an ordinary
start is a handful of `stat` calls rather than most of a second of hashing. Anything that
differs sends the files back to be hashed against the pins, and a file that does not match
is re-fetched - just that file.

**Guaranteed:** any change to a model file's contents or metadata since it was verified is
caught. `cp -p`, `tar x` and `rsync --inplace` can overwrite a file and restore its mtime,
which is why ctime is in the stamp - nothing in user space can set that back. **Not
guaranteed:** silent disk bit-rot, with no write at all.

A failure to *load* verified files is not treated as damage: it means onnxruntime,
permissions or memory, so the error is raised as it stands and nothing is downloaded.

Verification and loading hold a shared `flock`; downloading and repairing hold it
exclusively, so several servers starting at once download once between them. A model cache
on **NFS or SMB shared between machines is not supported** - `flock` can be local-only
there.

Nothing is ever deleted to reclaim space. After a successful start, one log line on stderr
names any other `embeddinggemma-300m-onnx*` folders and what they cost, and any file sitting
in the current folder that this version does not use - the int8 graph an upgrade left behind
weighs about 310 MB - and removing them is yours to do.

### Pre-download it, or install offline

To fetch the model deliberately rather than on the first query:

```bash
markdown-memory --download-model                          # the default, EmbeddingGemma
markdown-memory --download-model --embedder bge-small     # or the light preset
```

It downloads and loads the configured model, then exits; it touches no index.

For a machine with no network, copy the three files into
`$XDG_CACHE_HOME/markdown-memory/models/embeddinggemma-300m-onnx-5090578d9565/`, keeping
`onnx/model_q4.onnx`, `onnx/model_q4.onnx_data` and `tokenizer.json` where they are. The
first load hashes them once, writes the `.verified` stamp, and never touches the network.

`bge-small` is downloaded by `fastembed`, which pins no revision: if that cache is deleted,
it can come back with different weights under the same model name. EmbeddingGemma's
weights change on purpose, when a release moves its pin or graph. Either way, every document
records the weights that embedded it, and the index records the one revision that vouches
for all of its vectors:

- **Indexing repairs in place.** A document stamped by other weights is re-embedded like a
  changed file, even when its bytes are the same. Before the first new vector is written the
  index is marked as being re-embedded; nothing is deleted first, so keyword search answers
  throughout, and a run that is killed resumes where it stopped, because each stamp is
  written with its vectors. At the end of a run the revision is restored only once no
  vector-bearing document in the whole database - every root's, and one indexed on its own
  inside a pruned directory such as `.venv` - is stamped by anything else. Until then
  `index_status.coverage` reads `"unknown"` and its message names the directories still to
  be indexed.
- **Searching** compares for itself, after embedding the query, and falls back to keyword
  ranking alone when the answer differs - for the old weights and the new alike while a
  repair is under way. It does not wait to be told: weights can change while no Markdown
  file does, and then there is no indexing run to notice. The search that notices records
  it, and the next `index_directory` loads the model first and re-embeds what it finds.

A model that loads but cannot say which weights it is may not write into an index that
names its weights: its vectors could never be told apart from the ones already stored. An
index whose vectors no revision vouches for - built while the weights could not be read -
is not ranked against a query from weights that can name themselves - from the upgrade to
schema v6 on - and the first run with such weights re-embeds it. A server that loaded
`bge-small` while its revision could not be read keeps it unnamed until it restarts: the
revision is read beside the weights it loads, never after them.

### Presets

Two presets, chosen with `MARKDOWN_MEMORY_EMBEDDER`:

| Preset | Dimensions | Download | Peak RAM | Indexing | Held-out Top-1 / Top-3 / Top-5 |
| --- | --- | --- | --- | --- | --- |
| `embeddinggemma` (default) | 768 | ~218 MB | ~0.65 GB | ~7 vectors/s | 88% / 97% / 97% |
| `bge-small` | 384 | ~65 MB | ~1.1 GB | ~12 vectors/s | 68% / 82% / 88% |

Accuracy is the frozen baseline in `scripts/eval_data/baseline.json`, recorded by
`scripts/eval_retrieval.py` over a 54-section corpus and the 34 **held-out** paraphrase
queries, which were written before any parameter was tuned. The **dev** set - the one
tuning is allowed to look at, and deliberately harder - scores 71% / 85% / 94% with
EmbeddingGemma and 53% / 68% / 79% with bge-small. Exact identifiers - flags, environment
variables, error strings - are 100% Top-1 with either preset, because FTS5 answers them.
Query latency is not in the table on purpose: it swings by 2-3x with what else the machine
is doing, so the baseline records it as informational and so should you.

The same run reports what the default `search_docs` call costs: the estimated tokens of the
text block the MCP server actually sends (one section in full, pointers to the rest and
`index_status`), beside
the section that answers each query, as a median, a p95 and a median ratio per set
(`--show-costs` lists every query). It is informational - it never gates and never enters
the baseline - and it depends on where the repository is checked out, since every hit
carries its absolute path, so compare two runs from the same checkout. On this corpus the
default call costs a median of about 600-700 tokens against answering sections of
roughly 50-80.

A no-answer stratum - 4 queries per split that the corpus cannot answer, written before any
abstention rule existed - is scored apart from the rest: how often the default call returns
nothing, and how often `keyword_match` says no section contains the terms. It never gates
either: at 4 queries a split, one query moves it 25pp.

Switching preset **discards the whole index**: the two produce vectors of different sizes,
which cannot be compared, so every documentation root has to be indexed again.
`index_directory` reports that when it happens.

**The first index of a large documentation set is slow.** Every paragraph, list item, table
row and code block costs one vector; a section costs none of its own, because its vector is
pooled from its passages. Measured on a 97-file set: 1,682 sections and 8,453 passages -
10,135 vectors - indexed in **24 minutes** at 7.1 vectors per second, peaking at 667 MB,
after which a query over those 1,682 sections takes about 200 ms. Budget for it, and run it
once: indexing is incremental by SHA-256, so a
re-index that finds nothing changed takes milliseconds (8 ms for 36 sections) and only
edited files are re-embedded. `bge-small` indexes several times faster at a real cost in
accuracy.

EmbeddingGemma is distributed under the [Gemma Terms of Use](https://ai.google.dev/gemma/terms);
the revision is pinned. See [License](#license) for what that means for you.

### When it goes wrong

- **The server starts even when the model cannot load.** Nothing loads it until the
  first search, so the failure surfaces there, as
  `Cannot load embedding model onnx-community/embeddinggemma-300m-ONNX: ...`. So "the server
  is running" is not evidence the model is there; `markdown-memory --download-model` is.
- **The first `search_docs` can block for the length of a 218 MB download.** Pre-download it
  (above) if that matters.
- **All logging goes to stderr.** stdout carries JSON-RPC frames only, so a client that
  shows you "the output" may be showing you nothing. Set `MARKDOWN_MEMORY_LOG_LEVEL=DEBUG`
  and read stderr.
- **Switching preset discards the index.** The two models produce vectors of different
  sizes, which cannot be compared, so every root must be indexed again. `index_directory`
  says so when it happens.
- **Searches come back empty or stale.** Run `index_directory` again; it is incremental, so
  it is cheap. If `index_status.coverage` stays `"unknown"`, its `message` names the files
  that could not be read.

## Tools

| Tool | Purpose |
| --- | --- |
| `index_directory(directory=None)` | Scan a tree, (re)index new/changed `.md` files (SHA-256), purge deleted ones |
| `list_documents(directory="")` | `{documents, index_status}`: indexed paths, titles and section counts, and whether a full index run vouches for them |
| `get_document_outline(file_path)` | Hierarchical TOC with line ranges and token estimates |
| `read_section(file_path, heading_path, include_subsections=False)` | Verbatim text of one section |
| `search_docs(query, limit=5)` | `{results, keyword_match, index_status}` (plus `keyword_message` unless matched): BM25 + passage-level vector search fused with Reciprocal Rank Fusion (k = 60); the first hit carries its section's `content`, the rest are pointers (`file_path`, `heading_path`, `lines`, `tokens`, and `matched_passage` when a passage won) for `read_section`; `keyword_match` says whether keyword search found the query's terms (`no_match`: no section contains them, so the hits are only semantic neighbours); and `index_status` says whether the tree searched is known to be whole |

Sections are addressed by breadcrumb: `Root > Child > Subchild`. Oversized sections
(> ~800 tokens) are stored as `Root > Child (Part 1)`, `(Part 2)`, ...; reading the base
path reassembles them byte-for-byte. `file_path` may be absolute, relative to the docs
root, or any unique path suffix. `heading_path` is matched exactly first, then ignoring
spacing around `>`, then ignoring case, then as a trailing fragment (`Child > Subchild`
or just the title); an ambiguous request lists the exact candidates.

## Configuration

| Environment variable | CLI flag | Default |
| --- | --- | --- |
| `MARKDOWN_MEMORY_DOCS_DIR` | `--docs-dir` | `$CLAUDE_PROJECT_DIR` if the client exports it, else the working directory |
| `MARKDOWN_MEMORY_DB` | `--db` | `$XDG_DATA_HOME/markdown-memory/projects/<root>-<digest>/index.db` — one index per docs root |
| `MARKDOWN_MEMORY_MODEL_CACHE` | - | `$XDG_CACHE_HOME/markdown-memory/models` (`~/.cache/...`) |
| `MARKDOWN_MEMORY_EXCLUDE` | `--exclude` (repeatable) | none of your own - the built-in rules below still apply |
| `MARKDOWN_MEMORY_LOG_LEVEL` | `--log-level` | `INFO` |
| `MARKDOWN_MEMORY_EMBEDDER` | `--embedder` | `embeddinggemma` (or `bge-small`) |
| `MARKDOWN_MEMORY_THREADS` | - | unset: onnxruntime picks. A positive integer caps the threads one embedding pass may use |
| `MARKDOWN_MEMORY_INDEX_WORKERS` | - | `2` - files read, parsed and embedded at the same time while indexing |
| `MARKDOWN_MEMORY_AUTO_INDEX` | `--no-auto-index` | on - `0`, `false`, `off` or `no` stops the server indexing its root by itself |
| `MARKDOWN_MEMORY_GITIGNORE` | `--no-gitignore` | on - `0`, `false`, `off` or `no` indexes what git ignores as well |

The default preset does not spin-wait between operators, which is what makes a query cost ~0.6 s of
CPU instead of ~5.5 s and leaves the process idle at 0 while nothing is being asked of it; the
thread *count* is left to onnxruntime, and `MARKDOWN_MEMORY_THREADS` is there for a machine that
disagrees with its choice. `bge-small` runs through `fastembed`, which exposes no such switch, so
for that preset the count is the only lever: `MARKDOWN_MEMORY_THREADS=4` took one query from 718 ms
of CPU to 95 ms.

Indexing embeds several files at once and writes them from one thread, in the order the
tree was walked. One ONNX session is shared, and its weights are mapped once however many
threads run against it, so each extra worker costs about the 150 MB of one forward pass.
Measured over 24 files of the eval corpus (879 passages): 196.3 s with one worker, 145.9 s
with two (1.35x) and 96.3 s with four (2.04x) - less than the embedding speed-up alone,
because parsing and the writes stay serial and a long file holds the head of the queue.
Two is the default because its peak measures around 0.8 GB (757-814 MB across runs),
well inside what a tool running beside an editor should take; raise `MARKDOWN_MEMORY_INDEX_WORKERS` on a machine with cores to spare.

`MARKDOWN_MEMORY_EXCLUDE` takes glob patterns separated by commas (only commas - a colon
would split a pattern that contains one). A pattern with no `/` matches that name at any
depth, the way `.gitignore` treats one: `eval_data` excludes `scripts/eval_data/corpus/`.
A pattern containing `/` is anchored at the documentation root (`tests/fixtures`,
`docs/generated/*`). Matching is case-sensitive everywhere, and a matching directory is
pruned, so its subtree costs nothing to skip. Files already indexed before an exclusion
was added are purged on the next index.

Without it, a repository that keeps fixtures, vendored documentation or a test corpus
in-tree indexes them as if they were its own docs.

Three rules apply before your exclusions, so a monorepo indexes its own documentation and
not copies of it:

- **Built-in skips**: `.git`, `node_modules`, virtualenvs and tool caches are pruned (see
  [Indexing rules](#indexing-rules)).
- **Linked worktrees**: a directory below the root that `git worktree add` made - its `.git`
  file names a gitdir holding a `commondir` - is not entered: it is a copy of some state of
  the repository. The root itself always is, so pointing a server at a worktree indexes that
  worktree. Submodules and nested clones are not copies and are indexed, unless git ignores
  them; their *own* `.gitignore` is not read, so exclude their generated output yourself.
- **What git ignores**: inside a git repository, paths `git ls-files --others --ignored
  --exclude-standard` names are left out, so generated output and caches are too. Tracked
  files are never ignored, and git's paths are matched literally (`[id]/` is a directory
  name, not a pattern). Without git, or outside a repository, the walk is unchanged.
  `MARKDOWN_MEMORY_GITIGNORE=0` turns this rule off for docs kept in ignored directories on
  purpose.

A Markdown symlink to a file the same run indexes is not indexed a second time; a link to
anything else (a target outside the root, excluded, or not named `.md`) is indexed under
its own path. Like an exclusion, each rule purges what it covers from an index built
before it applied.

Documents are stored under their absolute path, so one database *can* hold several
projects - but **search only ever answers from the root this server was started with**,
and `list_documents` shows only that root. Each project gets its own database by default,
so this matters only if you point two of them at one file with `MARKDOWN_MEMORY_DB`: the
second is then indexed, invisible, and paying for itself in disk.

### One index per project

The user-level registration above already gives every project its own index. To set
exclusions for one repository, drop a `.mcp.json` like this into it (it takes precedence
there):
Each project owns its index without being told to: the database is keyed on the
documentation root it serves, so a project gets its own exclusions and no chance of
another project's sections - or another project's documents, which stay resolvable by
path across any database they share - appearing in its results.

```json
{
  "mcpServers": {
    "markdown-memory": {
      "command": "markdown-memory",
      "env": {
        "MARKDOWN_MEMORY_EXCLUDE": "vendor,third_party,tests/fixtures"
      }
    }
  }
}
```

Any path you do add is written relative, on purpose. Claude Code expands only real
environment variables in `.mcp.json`: `${workspaceFolder}` is a VS Code idea, and even
`${CLAUDE_PROJECT_DIR}` is not set at expansion time - measured on Claude Code 2.1.278,
both produce a *"Missing environment variables"* warning and are passed through as literal
text, which would make the server index a directory named `${workspaceFolder}` and report
success over zero files. The server refuses such a value outright, and resolves a relative
path against `CLAUDE_PROJECT_DIR` (which Claude Code *does* export to the spawned server),
falling back to the working directory only when that is not set. The docs root defaults to
that same project root, so it needs no entry. Each worktree of a repository is its own
directory, so each gets its own index.

By default nothing is written into the repository. (A *relative* `MARKDOWN_MEMORY_DB`
is resolved against the project root and does land inside it - `.gitignore` covers
`.markdown-memory/` for that reason, and any other relative path you choose is yours to
ignore.)
The index lives under `$XDG_DATA_HOME/markdown-memory/projects/`, in a directory named for
the documentation root and a digest of its resolved path - out of reach of `git clean
-xdf`, writable when the checkout is not, and on local disk when the checkout is on a
network share, where SQLite's write-ahead log cannot take the locks it needs. Set
`MARKDOWN_MEMORY_DB` to override it; a relative value is resolved against the project
root. The index is a cache of the Markdown files and is rebuilt from them, so deleting it
costs only the time to index again.

## How search ranks

1. **Keywords (FTS5, BM25).** Stopwords are dropped, identifiers are kept verbatim. A hit
   only counts if it covers at least half of the query's IDF mass or matches an
   identifier-like term - a stray match on "data" or "deploy" no longer outvotes the
   vector index.
2. **Vectors.** Every paragraph, list item, table row (rendered as `Header: cell; ...`) and
   code block - also inside block quotes and list items - is embedded separately. The
   section's own vector is the mean of those passage vectors, not a separate embedding of
   the whole section: the model truncates at 512 tokens, which a long section exceeds. It
   is an aggregate of the passages rather than independent evidence about the section, and
   it finds no section that the passages do not. A passage longer than 600 characters is split into
   consecutive windows at line, sentence or word boundaries, so a long command list or
   configuration block keeps a vector for all of itself rather than for its first 600
   characters. A table split across `(Part n)` sections keeps its header for every part. A
   section is ranked by its closest vector, so one relevant table row is enough.
   Heading-only sections have no vectors and are never returned ahead of their children.
3. **Reciprocal Rank Fusion** of the two rankings.

Cross-encoder rerankers (MiniLM, bge-reranker-base, jina, ColBERT) were benchmarked and
rejected: every one lowered accuracy on technical documentation and cost 2-12 s a query.

All logging goes to **stderr**. stdout carries JSON-RPC frames only.

## How malformed Markdown is handled

- **Skipped heading levels** (`#` then `####`): a heading stack pops every level `>= L`
  before pushing, so breadcrumbs stay well formed.
- **Preamble**: badges/summary before the first heading become `[Overview / Preamble]`.
- **YAML front matter**: kept out of the AST (CommonMark would read it as a setext
  heading) and used as a title fallback. A leading `---` rule followed by prose is not
  mistaken for it. One case is inherently ambiguous - a single `key: value` line between
  two `---` lines - and is read as front matter, as static-site generators do, unless the
  key is an admonition word (`Note:`, `Warning:`, `TODO:` ...).
- **Unclosed code fences**: CommonMark runs them to EOF - or to the closing marker of a
  *later* fence - swallowing the sections in between. The fence is closed before the next
  blank-line-preceded ATX heading instead. A level-1 `# ...` line is treated as a comment
  unless the fence language cannot have `#` comments (JSON, Go, ...). This is a heuristic:
  a fence that merely *looks* closed is only cut on strong evidence (it contains another
  opening fence with an info string, or the document ends inside a bare fence and the cut
  makes the rest well formed), and never when the repair would lose a heading that was
  already found. Repair work is capped per document, so a pathological file costs a
  bounded number of extra parses rather than one per fence.
- **No headings / walls of text**: split on paragraph boundaries, then on lines, then on
  whitespace. A fenced block is only cut when it exceeds the limit by itself.
- **Colliding breadcrumbs** get a ` [2]`, ` [3]` suffix (also against generated
  `(Part n)` paths), so every stored path addresses exactly one section.
- **Headings like `Option<T>`** keep their type parameter; only lower-case formatting
  tags (`<b>`, `<sub>`, `<a>`, ...) are stripped from titles.

## Indexing rules

- `.md` / `.markdown`, regular files only, at most 10 MB; symlinked directories are not
  followed. `.git`, `node_modules`, virtualenvs and tool caches are pruned - index such a
  tree by passing a directory *inside* it, and it is then left alone when an ancestor is
  re-indexed.
- A document is purged when the walk could have found it and did not, or when it is out of
  scope (excluded, ignored by git, inside another checkout). Files under a directory that
  cannot be listed are kept and the directory is reported as an error.

## Development

```bash
uv run ruff check . && uv run ruff format --check .
uv run mypy --strict src/
uv run pytest -v                      # unit + integration (real ONNX model for semantic tests)
uv run python scripts/live_test.py    # spawns the server, drives it over stdio JSON-RPC
uv run python scripts/eval_retrieval.py --show-misses   # retrieval accuracy; fails on regression
uv run python scripts/eval_retrieval.py --rebuild       # ... after discarding the cached index
uv run python scripts/eval_retrieval.py --show-costs    # ... and what each default call costs
uv run python scripts/reindex_docs.py DIR --force       # forced re-index + integrity verification
scripts/check.sh                                        # the whole pre-commit gate, fail-fast
```

### For AI coding agents

`CLAUDE.md` is the developer guide (commands, layout, architecture rules, the retrieval
regression policy). `AGENTS.md` and `.cursorrules` carry the short form for Codex and
Cursor. All three share one block of rules for *using* the MCP tools - search first, then
outline, then read a single section; never dump whole Markdown files - and a test keeps the
three copies identical and in step with the code. Project skills for Claude Code live in
`.claude/skills/`: `run-eval`, `reindex-docs`, `test-regression`.

Built on the MCP Python SDK 2.x, where `FastMCP` was renamed `MCPServer`.

## License

MIT - see [`LICENSE`](LICENSE). Two things in this repository are *not* covered by it,
because they are not ours to license:

- **The embedding models.** EmbeddingGemma-300m, the default, is distributed under the
  [Gemma Terms of Use](https://ai.google.dev/gemma/terms), which are not an OSI-approved
  open-source licence; the revision is pinned. The `bge-small` preset is two licences at
  once: the `BAAI/bge-small-en-v1.5` weights are MIT, and `fastembed`, which loads them,
  is Apache-2.0. Nothing is bundled - both are downloaded on first use - but if the Gemma
  terms do not suit you, `MARKDOWN_MEMORY_EMBEDDER=bge-small` avoids them entirely.
- **The evaluation corpus.** `scripts/eval_data/corpus_v2/` is third-party documentation
  vendored verbatim from five projects, pinned by commit, and used only to measure
  retrieval accuracy. Each upstream keeps its own licence, and its licence and NOTICE
  files travel with it in `scripts/eval_data/corpus_v2_licenses/`. The table in
  [`scripts/eval_data/corpus_v2_LICENSES.md`](scripts/eval_data/corpus_v2_LICENSES.md)
  says what came from where; both it and the manifest beside it are generated by
  `scripts/fetch_eval_corpus.py`, so edit the script rather than the files.
