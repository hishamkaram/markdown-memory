# Vendored documentation

The evaluation corpus is third-party documentation, copied verbatim at the commit
recorded in `corpus_v2_sources.json` and used here only to measure retrieval
accuracy. Each set keeps its own licence; none of it is part of the
markdown-memory package, which is MIT (see `LICENSE` at the repository root).

Each upstream's own licence files - and its NOTICE, where it has one - are
vendored verbatim beside this table in `corpus_v2_licenses/<set>/`, taken from
the same pinned commit as the documentation. Apache-2.0 asks that the licence
travel with the copy and that a NOTICE be carried into anything redistributed
from a tree that has one; naming the licence in a table is not that.

| Set | Upstream | Commit | Licence | Carried verbatim | Why it is here |
| --- | --- | --- | --- | --- | --- |
| `ripgrep` | BurntSushi/ripgrep | `3fce3b5bb023` | Unlicense OR MIT | [COPYING](corpus_v2_licenses/ripgrep/COPYING), [LICENSE-MIT](corpus_v2_licenses/ripgrep/LICENSE-MIT), [UNLICENSE](corpus_v2_licenses/ripgrep/UNLICENSE) | Dense flag prose and a changelog; the flags collide (-C, --context). |
| `cargo` | rust-lang/cargo | `cc7400190930` | MIT OR Apache-2.0 | [LICENSE-APACHE](corpus_v2_licenses/cargo/LICENSE-APACHE), [LICENSE-MIT](corpus_v2_licenses/cargo/LICENSE-MIT) | Man-page style command reference: SYNOPSIS/OPTIONS sections that repeat across dozens of commands, which is exactly the near-duplicate case. The reference chapter is left out to stop one source dominating the corpus. |
| `gh` | cli/cli | `0cf1092493af` | MIT | [LICENSE](corpus_v2_licenses/gh/LICENSE) | Task-oriented guides and environment variables. |
| `compose-spec` | compose-spec/compose-spec | `914ec15d1fa4` | Apache-2.0 | [LICENSE](corpus_v2_licenses/compose-spec/LICENSE), [NOTICE](corpus_v2_licenses/compose-spec/NOTICE) | Deeply nested declarative keys whose meaning is inherited from parents - the case that flatters passage vectors least. |
| `prometheus` | prometheus/prometheus | `64c05e80ccd1` | Apache-2.0 | [LICENSE](corpus_v2_licenses/prometheus/LICENSE), [NOTICE](corpus_v2_licenses/prometheus/NOTICE) | Configuration blocks and query-language reference: long YAML samples with '#' comments inside fences, which must never be read as headings. |

Refresh with `uv run python scripts/fetch_eval_corpus.py`. Moving a commit
changes the benchmark: the frozen baseline has to be re-recorded with it.
