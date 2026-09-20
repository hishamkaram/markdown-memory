"""Hybrid retrieval: BM25 (FTS5) and cosine vector search fused with Reciprocal Rank Fusion.

Two refinements, both measured on a labelled query set, sit in front of the fusion:

* **Passage max-sim.** A section is ranked by the closest of its vectors - the section
  as a whole or any single passage (table row, list item, paragraph, code block). One
  vector per section buries a relevant table row under everything around it.
* **IDF keyword gate.** RRF rewards a section for appearing in both rankings, so a stray
  match on a common word ("data", "deploy") used to lift a wrong section above the
  correct one that only the vector index had found. A keyword hit now counts only when
  it covers at least half of the query's information (IDF-weighted), or when it matches
  an identifier-like term (``--flag``, ``ENV_VAR``, ``/path``) that is rare in the corpus.
  Spelling alone does not make an identifier: ``HTTP``, ``API`` or ``2024`` look like one
  and are ordinary vocabulary wherever many sections mention them.
"""

from __future__ import annotations

import logging
import math
from collections.abc import Iterable, Sequence
from concurrent.futures import Future, ThreadPoolExecutor
from typing import TypeVar

from markdown_memory.db import Database
from markdown_memory.exceptions import MarkdownMemoryError, SearchError
from markdown_memory.indexer import Embedder
from markdown_memory.models import SearchResult

logger = logging.getLogger(__name__)

RRF_K = 60
CANDIDATES_PER_INDEX = 20
MAX_RESULT_LIMIT = 50
_MAX_QUERY_TERMS = 32
KEYWORD_GATE = 0.5  # minimum IDF-weighted share of the query a keyword hit must cover
# An identifier-like term passes the gate by itself only while it is rare: found in at most
# this many sections, or this share of all sections, whichever is larger.
IDENTIFIER_MAX_SECTIONS = 3
IDENTIFIER_MAX_SHARE = 0.05
_STALE_RETRIES = 1  # re-rank once when a concurrent re-index replaced ranked sections
_PASSAGES_PER_CANDIDATE = 10  # passage neighbours fetched per wanted section


# Function words carry no topical signal, yet OR-ing them into the MATCH expression makes
# FTS5 return a page of irrelevant sections - noise that RRF would then weigh equally
# against the vector ranking. They are dropped unless the query consists of nothing else.
_STOPWORDS = frozenset(
    [
        "a",
        "about",
        "after",
        "all",
        "also",
        "am",
        "an",
        "and",
        "any",
        "are",
        "as",
        "at",
        "be",
        "because",
        "been",
        "before",
        "being",
        "both",
        "but",
        "by",
        "can",
        "could",
        "did",
        "do",
        "does",
        "doing",
        "each",
        "for",
        "from",
        "had",
        "has",
        "have",
        "having",
        "he",
        "her",
        "here",
        "him",
        "his",
        "how",
        "i",
        "if",
        "in",
        "into",
        "is",
        "it",
        "its",
        "me",
        "more",
        "most",
        "my",
        "no",
        "nor",
        "not",
        "of",
        "on",
        "once",
        "only",
        "or",
        "other",
        "our",
        "out",
        "over",
        "own",
        "same",
        "she",
        "should",
        "so",
        "some",
        "such",
        "than",
        "that",
        "the",
        "their",
        "them",
        "then",
        "there",
        "these",
        "they",
        "this",
        "those",
        "through",
        "to",
        "too",
        "under",
        "until",
        "up",
        "us",
        "very",
        "was",
        "we",
        "were",
        "what",
        "when",
        "where",
        "which",
        "while",
        "who",
        "whom",
        "why",
        "will",
        "with",
        "would",
        "you",
        "your",
    ]
)


def build_fts_query(query: str) -> str | None:
    """Translate free text into a safe FTS5 MATCH expression (see ``fts_terms``)."""
    return " OR ".join(fts_terms(query)) or None


def fts_terms(query: str) -> list[str]:
    """The quoted FTS5 phrases a free-text query is searched by.

    Each whitespace-separated term becomes a quoted phrase, so FTS5 operators and
    punctuation in user input (``--max-retries``, ``MDMEM_DB_PATH``, ``NEAR(``, ``"``)
    are matched literally instead of being parsed as query syntax. The tokenizer
    splits a quoted term such as ``"MDMEM_DB_PATH"`` into the adjacent tokens
    ``mdmem db path``, giving exact-identifier matching. Terms are OR-ed: BM25
    ranks sections matching more (and rarer) terms first.
    """
    terms = [
        term for term in _sanitize(query).split() if any(character.isalnum() for character in term)
    ]
    # Stopwords go first, duplicates second: "where is the WHERE clause" must keep the
    # keyword even though its lower-case twin (a stopword) came earlier.
    meaningful = _without_duplicates(term for term in terms if not _is_stopword(term))
    chosen = (meaningful or _without_duplicates(terms))[:_MAX_QUERY_TERMS]
    return ['"' + term.replace('"', '""') + '"' for term in chosen]


def _sanitize(text: str) -> str:
    """Make ``text`` bindable: NUL ends SQLite's C strings, lone surrogates are not UTF-8."""
    return text.replace("\x00", " ").encode("utf-8", errors="replace").decode("utf-8")


def _without_duplicates(terms: Iterable[str]) -> list[str]:
    unique: dict[str, str] = {}
    for term in terms:
        unique.setdefault(term.lower(), term)
    return list(unique.values())


_IDENTIFIER_MARKS = frozenset("_./\\:@#$=")


def _is_identifier(quoted_term: str) -> bool:
    """Spelled like a flag, path, environment variable, constant or version.

    Spelling cannot tell ``ENOSPC`` from ``HTTP``: whether a match on such a term may
    bypass the keyword gate also depends on how rare it is (see ``HybridSearcher._gate``).
    """
    term = quoted_term.strip('"')
    return (
        term.startswith("-")
        or any(character in _IDENTIFIER_MARKS or character.isdigit() for character in term[:-1])
        or (len(term) > 1 and term.isupper())
    )


_SENTENCE_PUNCTUATION = "\"'()[]{}<>?!.,;:"


def _is_stopword(term: str) -> bool:
    """True for plain function words only.

    Anything that looks like an identifier is kept even when it spells a stopword:
    flags (``--all``, ``-i``), decorated names (``@Before``, ``IS_ON``) and upper-case
    keywords (``WHERE``, ``NOT NULL``) are exactly what keyword search exists for.
    """
    word = term.strip(_SENTENCE_PUNCTUATION)
    if not word.isalpha() or (len(word) > 1 and word.isupper()):
        return False
    return word.lower() in _STOPWORDS


def reciprocal_rank_fusion(rankings: Sequence[Sequence[int]], k: int = RRF_K) -> dict[int, float]:
    """``RRF(d) = sum over rankings of 1 / (k + rank(d))`` with 1-based ranks."""
    scores: dict[int, float] = {}
    for ranking in rankings:
        for rank, item in enumerate(ranking, start=1):
            scores[item] = scores.get(item, 0.0) + 1.0 / (k + rank)
    return scores


class HybridSearcher:
    """Runs keyword and vector search concurrently and fuses the two rankings."""

    def __init__(
        self,
        db: Database,
        embedder: Embedder,
        *,
        candidates_per_index: int = CANDIDATES_PER_INDEX,
        rrf_k: int = RRF_K,
    ) -> None:
        self._db = db
        self._embedder = embedder
        self._candidates = candidates_per_index
        self._rrf_k = rrf_k
        # Two long-lived workers so each keeps its own (per-thread) SQLite connection.
        self._pool = ThreadPoolExecutor(max_workers=2, thread_name_prefix="mdmem-search")

    def close(self) -> None:
        self._pool.shutdown(wait=True)

    def search(self, query: str, limit: int = 5) -> list[SearchResult]:
        """Top ``limit`` sections by descending RRF score."""
        query = _sanitize(query).strip()
        if not query:
            return []
        limit = max(1, min(limit, MAX_RESULT_LIMIT))
        # Re-indexing a document replaces its section rows, so ids ranked a moment ago can
        # be gone by the time they are fetched. The new rows are already committed: rank
        # again rather than hand back a short (or empty) page with no explanation.
        for _ in range(_STALE_RETRIES):
            results, stale = self._search_once(query, limit)
            if not stale:
                return results
            logger.info("Sections changed during the search; ranking again")
        return self._search_once(query, limit)[0]

    def _search_once(self, query: str, limit: int) -> tuple[list[SearchResult], bool]:
        """One ranking pass: the results, and whether a better-ranked section had vanished."""
        candidates = max(self._candidates, limit)
        try:
            fts_future = self._pool.submit(self._keyword_ranking, query, candidates)
            vec_future = self._pool.submit(self._vector_ranking, query, candidates)
        except RuntimeError as exc:  # the executor refuses work after close()
            raise SearchError("The search engine has been shut down") from exc
        fts_ranking, fts_error = _settle(fts_future, [])
        (vec_ranking, passages), vec_error = _settle(vec_future, ([], {}))
        if fts_error is not None and vec_error is not None:
            raise fts_error
        for name, error in (("keyword", fts_error), ("vector", vec_error)):
            if error is not None:
                logger.warning("%s search failed; using the other index only: %s", name, error)

        scores = reciprocal_rank_fusion([fts_ranking, vec_ranking], self._rrf_k)
        fts_ranks = {section_id: rank for rank, section_id in enumerate(fts_ranking, start=1)}
        vec_ranks = {section_id: rank for rank, section_id in enumerate(vec_ranking, start=1)}
        ordered = sorted(scores, key=lambda section_id: (-scores[section_id], section_id))

        hydrated = self._db.get_sections_with_documents(ordered[:limit])
        if len(hydrated) < len(ordered[:limit]):
            # Deleted by a concurrent re-index between ranking and fetch: the next-best
            # candidates fill the page instead of leaving it short.
            hydrated.update(self._db.get_sections_with_documents(ordered[limit:]))
        results: list[SearchResult] = []
        stale = False
        for section_id in ordered:
            if len(results) == limit:
                break
            pair = hydrated.get(section_id)
            if pair is None:
                stale = True
                continue
            section, document = pair
            results.append(
                SearchResult(
                    section_id=section.id,
                    file_path=document.file_path,
                    document_title=document.title,
                    heading_title=section.heading_title,
                    heading_path=section.heading_path,
                    content=section.content,
                    start_line=section.start_line,
                    end_line=section.end_line,
                    score=scores[section_id],
                    fts_rank=fts_ranks.get(section_id),
                    vec_rank=vec_ranks.get(section_id),
                    matched_passage=passages.get(section_id),
                )
            )
        return results, stale

    def _keyword_ranking(self, query: str, limit: int) -> list[int]:
        terms = fts_terms(query)
        if not terms:
            return []
        hits = self._db.fts_search(" OR ".join(terms), limit)
        hits = self._gate(terms, hits)
        # Heading-only sections are signposts: their children carry the same breadcrumb
        # words plus the actual text. They stay only when nothing else matched.
        with_body = self._db.sections_with_passages(hits)
        return [hit for hit in hits if hit in with_body] or hits

    def _gate(self, terms: Sequence[str], hits: list[int]) -> list[int]:
        """Keep the hits whose matched terms carry >= ``KEYWORD_GATE`` of the query's IDF."""
        if len(terms) < 2 or not hits:
            return hits
        total = self._db.count_rows("sections")
        weights: dict[str, float] = {}
        frequencies: dict[str, int] = {}
        matched: dict[str, set[int]] = {}
        for term in terms:
            frequency = self._db.fts_document_frequency(term)
            weights[term] = math.log(1 + (total - frequency + 0.5) / (frequency + 0.5))
            matched[term] = self._db.fts_matching(term, hits) if frequency else set()
            frequencies[term] = frequency
        budget = sum(weights.values()) or 1.0
        # A term that merely looks like an identifier ("HTTP", "RAM", "2024") and occurs
        # all over the corpus is vocabulary: admitting every section that mentions it is
        # exactly the noise this gate exists to remove. It still counts towards coverage.
        rare = max(IDENTIFIER_MAX_SECTIONS, int(total * IDENTIFIER_MAX_SHARE))
        exact: set[int] = set()
        for term in terms:
            if _is_identifier(term) and frequencies[term] <= rare:
                exact |= matched[term]

        def coverage(hit: int) -> float:
            return sum(weights[term] for term in terms if hit in matched[term]) / budget

        return [hit for hit in hits if hit in exact or coverage(hit) >= KEYWORD_GATE]

    def _vector_ranking(self, query: str, limit: int) -> tuple[list[int], dict[int, str]]:
        """Sections by their closest vector, plus each section's best-matching passage."""
        embedding = self._embedder.embed_query(query)
        best: dict[int, float] = dict(self._db.vec_search(embedding, limit))
        passages: dict[int, str] = {}
        for section_id, distance, passage in self._db.unit_search(
            embedding, limit * _PASSAGES_PER_CANDIDATE
        ):
            passages.setdefault(section_id, passage)  # closest first: keep the best one
            if distance < best.get(section_id, math.inf):
                best[section_id] = distance
        ranking = sorted(best, key=lambda section_id: (best[section_id], section_id))[:limit]
        return ranking, {sid: passages[sid] for sid in ranking if sid in passages}


_R = TypeVar("_R")


def _settle(future: Future[_R], empty: _R) -> tuple[_R, MarkdownMemoryError | None]:
    """Resolve one ranking; any failure degrades to ``empty`` plus its error.

    Unanticipated exceptions are wrapped rather than re-raised so that one broken index
    can neither abandon the other index's result nor surface as an opaque tool crash.
    """
    try:
        return future.result(), None
    except MarkdownMemoryError as exc:
        return empty, exc
    except Exception as exc:
        logger.exception("Unexpected failure in a search index")
        return empty, SearchError(f"{type(exc).__name__}: {exc}")
