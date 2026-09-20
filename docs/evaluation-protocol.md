# Evaluation protocol

Fixed on 2026-09-20, **before** any query of the v2 benchmark was written or judged. The
point of writing it first is that the rules cannot be adjusted once the numbers arrive;
git history is the evidence that they were not. Changing anything here means a commit
that says what changed and why, dated before the run it governs.

## What is being measured

**Estimand.** The paired difference in Recall@5 between two systems answering the *same*
queries against the *same* index: `Recall@5(candidate) - Recall@5(baseline)`. Paired,
because the alternative - comparing two independent proportions - throws away the fact
that both systems see the same corpus and the same questions, and widens the interval for
nothing.

**A query is recalled** when a section graded `>= 2` appears in the top 5. Binary. The
0-3 grades are kept, but only nDCG@5 uses their magnitude.

**Metric hierarchy**, in the order disagreements are resolved:

| Metric | Role | Rule |
| --- | --- | --- |
| Recall@5 | primary | decides pass or fail |
| nDCG@5 | secondary | must be non-inferior; a Recall@5 gain with a material nDCG@5 loss is a trade-off to argue, not an automatic win |
| valid@1 | diagnostic | never decides anything on its own; valid@1 up with Recall@5 down is a regression |
| identifier Top-1 | hard guard | any drop below 100% fails the run regardless of every other number |

## Decision rule

- **Margin: -5pp.** Not -3pp. Simulated at this corpus's real shape - 97 documents, 400
  queries, bootstrap resampling documents - a true -3pp regression is detected 51% of the
  time at 8% discordance and 34% at 15%, while -5pp is detected 97%. A gate that misses
  half of what it exists to catch should not claim 3pp.
- **Alpha 0.05, two-sided**, reported as a 95% interval on the paired difference.
- **Bootstrap**: 10,000 resamples of *documents*, not queries, because queries from one
  document succeed and fail together. Resamples reduce Monte-Carlo noise; they add no
  information, and no number of them turns 400 queries into more evidence than 400
  queries.
- **Exit codes**: `0` the candidate is non-inferior (whole interval above -5pp); `1` a
  regression (interval entirely below -5pp, or the identifier guard fired); `2`
  inconclusive - the interval spans the margin.
- **`p >= 0.05` means a regression was not detected, never that the change is safe.** The
  gate prints the measured power beside its verdict so that an exit 2 cannot be read as
  an exit 0.

## Building the query set

**Pooling.** Candidates for judging come from the union of: BM25, a second embedder, the
production system's top-k, exact-identifier retrieval, **and the candidate system's
top-k**. A pool that excludes the system under test scores its novel results as failures
by construction.

**The origin section is forced into the pool.** If a query's originating section is not
retrieved by anything, that is a **failure to report**, not a query to discard. The
earlier rule - drop the query when its origin scores below 2 - removed exactly the
failures the benchmark exists to measure, and made Recall@5 conditional on already being
findable. Two numbers are reported separately: *origin missing from the initial pool* and
*origin still unjudged after force-insertion*.

**No-answer cases.** Between 5% and 10% of the set is questions the corpus cannot answer:
out of scope, about a document that was deleted, or answerable only by a section that
does not exist. Returning nothing is the correct response; a confident wrong section is a
failure. Real use contains these, and a benchmark without them measures a world where
every question has an answer.

Recall@5 cannot express that, and saying so afterwards would be too late: on a no-answer
query, a system that abstains and a system that returns five irrelevant sections both
score zero. So these queries are scored by their own metric, declared here:

- **abstention accuracy** = the fraction of no-answer queries answered with an empty
  result. Reported separately from Recall@5 and never averaged into it.
- **Margin: -10pp**, paired and bootstrapped the same way. Wider than the Recall@5 margin
  because the no-answer stratum is 20-40 queries, not 400, and a tighter number would be
  a claim the sample cannot support.
- **nDCG convention** for a query with no relevant section: 1.0 when nothing is returned,
  0.0 otherwise. Without this, nDCG is undefined on exactly the queries that need it.
- A regression here does **not** fail the gate on its own - the sample is too small - but
  it is reported beside the verdict, and a change that improves Recall@5 while making the
  system answer confidently where it should stay silent is a trade-off to argue, not a
  pass.

**Strata.** Heading-derived queries are **one** stratum, not the substrate: BM25 weights
headings 5.0 and every passage vector carries its breadcrumb, so a set generated from
heading paths flatters the exact heuristics under test. The remaining families come from
the preflight (`scripts/preflight.py`) - whatever real agents actually ask. Mechanical
strata (interaction, sibling-disambiguation) stay at 15-20%. Identifier lookups remain a
smoke-test canary, already at 100%.

**Judging.** Blind, randomised candidate order. The hard-negative strata are graded
twice, independently; weighted kappa below 0.60 after 100 double-judged items means the
labels are not reproducible and the run stops. Discard rate is reported by stratum,
heading depth and document.

**Splits.** `dev` and `held_out`, frozen as a committed fixture. Parameters are chosen on
`dev` only. A number from `held_out` that has influenced a decision is spent, and saying
so afterwards does not restore it.

## Abort thresholds

Stop and redesign rather than push through:

| Signal | Threshold |
| --- | --- |
| Origin missing from the initial pool | > 5% of queries |
| Mechanical strata | >= 90% Recall@5 for every system, spread < 2pp |
| Alternatives promoted | > 35% of queries promote >= 3 sections |
| Inter-judge agreement | weighted kappa < 0.60 after 100 double-judged items |
| Bootstrap half-width | > 5pp on the paired Recall@5 difference (the declared margin) |
| Run-to-run stability | > 0.5% of queries change their top 5 between identical runs |

## What this protocol does not cover

Recall@5 says nothing about how much of the returned page is noise, and a page whose
four other results are distractors still costs the agent its context. Precision and
context cost are measured and reported, but they do not gate, because there is no
evidence yet for where their threshold should sit. The same applies to latency.
