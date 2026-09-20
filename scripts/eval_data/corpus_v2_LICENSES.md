# Vendored documentation

The evaluation corpus is third-party documentation, copied verbatim at the commit
recorded in `sources.json` and used here only to measure retrieval accuracy.
Each set keeps its own licence; none of it is part of the markdown-memory package.

| Set | Upstream | Commit | Licence | Why it is here |
| --- | --- | --- | --- | --- |
| `ripgrep` | BurntSushi/ripgrep | `3fce3b5bb023` | Unlicense OR MIT | Dense flag prose and a changelog; the flags collide (-C, --context). |
| `cargo` | rust-lang/cargo | `cc7400190930` | MIT OR Apache-2.0 | Man-page style command reference: SYNOPSIS/OPTIONS sections that repeat across dozens of commands, which is exactly the near-duplicate case. The reference chapter is left out to stop one source dominating the corpus. |
| `gh` | cli/cli | `0cf1092493af` | MIT | Task-oriented guides and environment variables. |
| `compose-spec` | compose-spec/compose-spec | `914ec15d1fa4` | Apache-2.0 | Deeply nested declarative keys whose meaning is inherited from parents - the case that flatters passage vectors least. |
| `prometheus` | prometheus/prometheus | `64c05e80ccd1` | Apache-2.0 | Configuration blocks and query-language reference: long YAML samples with '#' comments inside fences, which must never be read as headings. |

Refresh with `uv run python scripts/fetch_eval_corpus.py`. Moving a commit
changes the benchmark: the frozen baseline has to be re-recorded with it.
