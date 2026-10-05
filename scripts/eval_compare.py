"""Hold one revision to another, query by query: no query may get worse (#94).

    uv run python scripts/eval_compare.py BASE.json CANDIDATE.json [--allow-label-changes]

Both files are written by ``scripts/eval_retrieval.py --record`` over the same corpus, preset
and splits. Totals hide a loss among gains - #81 gained three queries for embeddinggemma while
bge-small lost ``promtool check rules`` from #1 to #2 - so the rule is per query:

- the expected section's rank may not drop (absent from the first 20 counts as 21);
- the first result may not stop being a graded answer (any-valid@1);
- nDCG@5 may not fall by more than 1e-9;
- a query whose default page had results may not come back empty;
- a query the corpus cannot answer must get the same page: hits and ``keyword_match``.

Exit 0: no query got worse. Exit 1: at least one did, and each is named. Exit 2: the records
are malformed or not comparable - different corpus, preset, splits, queries or grading code, or
different labels without ``--allow-label-changes``. The McNemar p over Top-1 wins and losses is
descriptive: at these sizes it cannot certify anything, and it never decides the exit code.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import TypeGuard, get_args

from markdown_memory.models import KeywordMatch

SCHEMA = 1
SPLITS = ("dev", "held_out")
KINDS = ("paraphrase", "identifier", "no_answer")
ANSWERABLE = ("paraphrase", "identifier")
SHAPES = ("identifier", "question")  # eval_retrieval.NO_ANSWER_SHAPES
FIELDS = {
    "answerable": ("expected", "also_valid", "rank", "any_valid", "ndcg5", "top", "hits",
                   "keyword_match"),
    "no_answer": ("shape", "hits", "keyword_match"),
}  # fmt: skip
KEYWORD_STATES = get_args(KeywordMatch)
MAX_RANK = 20  # what eval_retrieval.evaluate searches
MAX_HITS = 5  # the default page
ABSENT = MAX_RANK + 1
NDCG_TOLERANCE = 1e-9
HEADER = ("corpus", "preset", "splits", "queries_sha256", "corpus_sha256", "code", "evaluator")


class RecordError(Exception):
    """The records cannot be compared: exit 2, nothing is judged."""


Record = dict[str, object]
Case = dict[str, object]


def _strict_object(pairs: list[tuple[str, object]]) -> dict[str, object]:
    keys = [key for key, _ in pairs]
    if len(keys) != len(set(keys)):
        raise RecordError(f"duplicate key in a JSON object: {sorted(keys)}")
    return dict(pairs)


def _is_int(value: object) -> TypeGuard[int]:
    return isinstance(value, int) and not isinstance(value, bool)


def _rank(case: Case) -> int:
    rank = case["rank"]
    return rank if _is_int(rank) else ABSENT


def _ndcg(case: Case) -> float:
    value = case["ndcg5"]
    return float(value) if isinstance(value, int | float) else math.nan


def _check_case(name: str, key: str, case: object) -> Case:
    if not isinstance(case, dict):
        raise RecordError(f"{name}: case {key!r} is not an object")
    kind = "no_answer" if key.split("|", 2)[1] == "no_answer" else "answerable"
    absent = [field for field in FIELDS[kind] if field not in case]
    if absent:
        raise RecordError(f"{name}: case {key!r} has no {', '.join(absent)}")
    hits, keyword = case.get("hits"), case.get("keyword_match")
    if not _is_int(hits) or not 0 <= hits <= MAX_HITS:
        raise RecordError(f"{name}: case {key!r} has hits {hits!r}")
    if keyword not in KEYWORD_STATES:
        raise RecordError(f"{name}: case {key!r} has keyword_match {keyword!r}")
    if kind == "no_answer":
        if case["shape"] not in SHAPES:
            raise RecordError(f"{name}: no-answer case {key!r} has shape {case['shape']!r}")
        return case
    rank, ndcg = case.get("rank"), case.get("ndcg5")
    if rank is not None and (not _is_int(rank) or not 1 <= rank <= MAX_RANK):
        raise RecordError(f"{name}: case {key!r} has rank {rank!r}")
    if not isinstance(case.get("any_valid"), bool):
        raise RecordError(f"{name}: case {key!r} has any_valid {case.get('any_valid')!r}")
    # `1e400` is valid JSON and decodes to infinity, which no subtraction can veto.
    if not (_is_int(ndcg) or isinstance(ndcg, float)) or not math.isfinite(ndcg):
        raise RecordError(f"{name}: case {key!r} has nDCG@5 {ndcg!r}")
    if not isinstance(case.get("expected"), str) or not isinstance(case.get("also_valid"), dict):
        raise RecordError(f"{name}: case {key!r} has no labels")
    top = case["top"]
    if not isinstance(top, list) or not all(isinstance(label, str) for label in top):
        raise RecordError(f"{name}: case {key!r} has top {top!r}")
    return case


def load(path: Path) -> Record:
    """Read one record, refusing anything that could make a comparison say less than it seems."""
    name = path.name
    try:
        record = json.loads(
            path.read_text(encoding="utf-8"),
            object_pairs_hook=_strict_object,
        )
    except (OSError, ValueError) as error:
        raise RecordError(f"{name}: {error}") from error
    if not isinstance(record, dict) or not _is_int(record.get("schema")):
        raise RecordError(f"{name}: not a record (schema {record.get('schema')!r})")
    if record["schema"] != SCHEMA:
        raise RecordError(f"{name}: not a schema-{SCHEMA} record")
    missing = [field for field in (*HEADER, "counts", "cases") if field not in record]
    if missing:
        raise RecordError(f"{name}: missing {', '.join(missing)}")
    splits, counts, cases = record["splits"], record["counts"], record["cases"]
    if (
        not isinstance(splits, list)
        or not splits
        or len(set(splits)) != len(splits)
        or any(split not in SPLITS for split in splits)
    ):
        raise RecordError(f"{name}: splits {splits!r}")
    if not isinstance(counts, dict) or not isinstance(cases, dict):
        raise RecordError(f"{name}: counts and cases must be objects")
    present: dict[str, int] = {}
    for key, case in cases.items():
        parts = key.split("|", 2)
        if len(parts) != 3 or parts[0] not in splits or parts[1] not in KINDS:
            raise RecordError(f"{name}: case {key!r} is outside the splits and kinds it declares")
        _check_case(name, key, case)
        stratum = f"{parts[0]}|{parts[1]}"
        present[stratum] = present.get(stratum, 0) + 1
    expected = {f"{split}|{kind}" for split in splits for kind in KINDS}
    if set(counts) != expected or not all(_is_int(n) for n in counts.values()):
        raise RecordError(f"{name}: counts {counts!r} do not cover {sorted(expected)}")
    for stratum in sorted(expected):
        if present.get(stratum, 0) != counts[stratum]:
            raise RecordError(
                f"{name}: {stratum} holds {present.get(stratum, 0)} of {counts[stratum]}"
            )
        # Two empty records would agree perfectly; an answerable stratum must hold queries.
        if stratum.endswith(ANSWERABLE) and not counts[stratum]:
            raise RecordError(f"{name}: {stratum} is empty")
    return record


def _labels(case: Case) -> tuple[object, object]:
    return case.get("expected"), case.get("also_valid")


def comparable(base: Record, candidate: Record, *, allow_label_changes: bool) -> list[str]:
    """Raise ``RecordError`` unless the two records measured the same thing; list label changes."""
    for field in ("corpus", "preset", "splits", "corpus_sha256", "evaluator"):
        if base[field] != candidate[field]:
            raise RecordError(
                f"the records differ in {field}: {base[field]!r} != {candidate[field]!r}"
            )
    cases_a, cases_b = base["cases"], candidate["cases"]
    assert isinstance(cases_a, dict) and isinstance(cases_b, dict)  # load() checked
    if cases_a.keys() != cases_b.keys():
        only = sorted(cases_a.keys() ^ cases_b.keys())
        raise RecordError(f"the records hold different queries: {only[:5]}")
    changed = [key for key in cases_a if _labels(cases_a[key]) != _labels(cases_b[key])]
    shapes = [k for k in cases_a if cases_a[k].get("shape") != cases_b[k].get("shape")]
    if shapes:
        raise RecordError(f"no-answer shapes differ: {shapes[:5]}")
    if changed and not allow_label_changes:
        raise RecordError(
            f"{len(changed)} case(s) are labelled differently; pass --allow-label-changes "
            f"after reviewing each migration: {changed[:5]}"
        )
    return changed


def violations(key: str, before: Case, after: Case) -> list[str]:
    """What the frozen #79/#80 rule finds wrong with one case."""
    if key.split("|", 2)[1] == "no_answer":
        page_before = (before["hits"], before["keyword_match"])
        if page_before != (after["hits"], after["keyword_match"]):
            return [
                f"no-answer page changed: hits {before['hits']}->{after['hits']}, "
                f"keyword_match {before['keyword_match']}->{after['keyword_match']}"
            ]
        return []
    found: list[str] = []
    if _rank(after) > _rank(before):
        found.append(f"rank {before['rank']}->{after['rank']}")
    if before["any_valid"] and not after["any_valid"]:
        found.append("any-valid@1 lost")
    if _ndcg(after) < _ndcg(before) - NDCG_TOLERANCE:
        found.append(f"nDCG@5 {_ndcg(before):.6f}->{_ndcg(after):.6f}")
    if before["hits"] and not after["hits"]:
        found.append(f"new empty page (hits {before['hits']}->0)")
    return found


def mcnemar(wins: int, losses: int) -> float:
    """Two-sided exact McNemar p over discordant pairs: descriptive, never a gate."""
    n = wins + losses
    if n == 0:
        return 1.0
    tail = sum(math.comb(n, i) for i in range(min(wins, losses) + 1))
    return min(1.0, 2 * tail / float(2**n))


def _changed(before: Case, after: Case) -> bool:
    fields = ("rank", "any_valid", "hits", "keyword_match")
    if any(before.get(field) != after.get(field) for field in fields):
        return True
    return "ndcg5" in before and abs(_ndcg(before) - _ndcg(after)) > NDCG_TOLERANCE


def compare(base: Record, candidate: Record, *, allow_label_changes: bool) -> int:
    relabelled = comparable(base, candidate, allow_label_changes=allow_label_changes)
    cases_a, cases_b = base["cases"], candidate["cases"]
    assert isinstance(cases_a, dict) and isinstance(cases_b, dict)
    splits = base["splits"]
    assert isinstance(splits, list)
    scope = f"{base['corpus']} {base['preset']} {'+'.join(map(str, splits))}"
    print(f"comparator {hashlib.sha256(Path(__file__).read_bytes()).hexdigest()[:12]}  "
          f"scope: {scope}")  # fmt: skip
    code_a, code_b = base["code"], candidate["code"]
    assert isinstance(code_a, dict) and isinstance(code_b, dict)
    print(f"base      {code_a.get('revision')}  package {str(code_a.get('sha256'))[:12]}")
    print(f"candidate {code_b.get('revision')}  package {str(code_b.get('sha256'))[:12]}")
    if code_a.get("sha256") == code_b.get("sha256"):
        same_labels = base["queries_sha256"] == candidate["queries_sha256"] and not relabelled
        print("same-code comparison" + (": a repeatability check" if same_labels else ""))
    for key in relabelled:
        print(f"  label changed (not an identical-definition comparison): {key}")
        print(f"    {_labels(cases_a[key])} -> {_labels(cases_b[key])}")

    failures: list[str] = []
    wins: dict[str, int] = {}
    losses: dict[str, int] = {}
    for key, before in cases_a.items():
        after = cases_b[key]
        split, kind, _ = key.split("|", 2)
        if _changed(before, after):
            ndcg = f"{_ndcg(before):.4f}->{_ndcg(after):.4f}" if "ndcg5" in before else "-"
            print(f"  changed {key}: rank {before.get('rank')}->{after.get('rank')}  "
                  f"any-valid {before.get('any_valid')}->{after.get('any_valid')}  "
                  f"nDCG@5 {ndcg}  "
                  f"hits {before['hits']}->{after['hits']}  "
                  f"keyword_match {before['keyword_match']}->{after['keyword_match']}")  # fmt: skip
        failures.extend(f"{key}: {found}" for found in violations(key, before, after))
        if kind in ANSWERABLE:
            stratum = f"{split} {kind}"
            first_a, first_b = before["rank"] == 1, after["rank"] == 1
            wins[stratum] = wins.get(stratum, 0) + (first_b and not first_a)
            losses[stratum] = losses.get(stratum, 0) + (first_a and not first_b)
    for stratum in wins:
        won, lost = wins[stratum], losses[stratum]
        print(f"{stratum}: Top-1 wins {won}, losses {lost} "
              f"(exact McNemar p={mcnemar(won, lost):.3f}, descriptive)")  # fmt: skip
    if failures:
        print(f"REJECT ({scope}): {len(failures)} violation(s)")
        for failure in failures:
            print(f"  {failure}")
        return 1
    print(f"PASS ({scope}): no query got worse - this cell only, not overall acceptance")
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    parser.add_argument("base", type=Path, help="record of the revision held as the reference")
    parser.add_argument("candidate", type=Path, help="record of the revision under test")
    parser.add_argument(
        "--allow-label-changes",
        action="store_true",
        help="compare cases whose labels differ (a reviewed label migration); each is listed",
    )
    arguments = parser.parse_args(argv)
    try:
        base, candidate = load(arguments.base), load(arguments.candidate)
        return compare(base, candidate, allow_label_changes=arguments.allow_label_changes)
    except RecordError as unusable:
        print(f"NOT COMPARABLE: {unusable}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
