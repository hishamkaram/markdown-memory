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
from itertools import pairwise
from typing import NoReturn, TypeVar

from markdown_memory.db import WEIGHTS_META_KEY, WEIGHTS_REVOKED, Database
from markdown_memory.embedders import Embedder, short_weights
from markdown_memory.exceptions import MarkdownMemoryError, SearchError
from markdown_memory.models import KeywordMatch, SearchPage, SearchResult, preview

logger = logging.getLogger(__name__)

RRF_K = 60
# Scoped search filters after each index has applied its own limit.
_SCOPED_OVERFETCH = 4
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
    """Spelled like a flag, path, environment variable, constant, camelCase key or version.

    Spelling cannot tell ``ENOSPC`` from ``HTTP``: whether a match on such a term may
    bypass the keyword gate also depends on how rare it is (see ``HybridSearcher._gate``).
    """
    # An agent quotes code the way Markdown does: `--flag` is the flag, for spelling too.
    term = quoted_term.strip("\"'`")
    return (
        term.startswith("-")
        or any(character in _IDENTIFIER_MARKS or character.isdigit() for character in term[:-1])
        or (len(term) > 1 and term.isupper())
        or any(lower.islower() and upper.isupper() for lower, upper in pairwise(term))
    )


def _is_identifier_lookup(terms: Sequence[str]) -> bool:
    """Every searched term is spelled like an identifier, and none went unsearched.

    At ``_MAX_QUERY_TERMS`` ``fts_terms`` may have cut the query short, and `no_match` would
    then speak for only part of it.
    """
    return 0 < len(terms) < _MAX_QUERY_TERMS and all(_is_identifier(term) for term in terms)


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
        scope: str | None = None,
    ) -> None:
        self._db = db
        self._embedder = embedder
        self._candidates = candidates_per_index
        self._rrf_k = rrf_k
        # One database can hold several documentation roots: the default is keyed per root,
        # but a configured MARKDOWN_MEMORY_DB can point two of them at one file. Without
        # this, an agent working in one project gets confident answers out of another
        # project's documentation.
        self._scope = scope
        # Two long-lived workers so each keeps its own (per-thread) SQLite connection.
        self._pool = ThreadPoolExecutor(max_workers=2, thread_name_prefix="mdmem-search")

    def close(self) -> None:
        self._pool.shutdown(wait=True)

    def search(self, query: str, limit: int = 5) -> list[SearchResult]:
        """Top ``limit`` sections by descending RRF score."""
        return list(self.search_page(query, limit).results)

    def search_page(self, query: str, limit: int = 5) -> SearchPage:
        """The top ``limit`` sections, and whether keyword search found the query's terms."""
        query = _sanitize(query).strip()
        if not query:
            return SearchPage((), "no_terms")
        limit = max(1, min(limit, MAX_RESULT_LIMIT))
        # Re-indexing a document replaces its section rows, so ids ranked a moment ago can
        # be gone by the time they are fetched. The new rows are already committed: rank
        # again rather than hand back a short (or empty) page with no explanation.
        for _ in range(_STALE_RETRIES):
            page, stale = self._search_once(query, limit)
            if not stale:
                return page
            logger.info("Sections changed during the search; ranking again")
        return self._search_once(query, limit)[0]

    def _search_once(self, query: str, limit: int) -> tuple[SearchPage, bool]:
        """One ranking pass: the page, and whether a better-ranked section had vanished."""
        candidates = max(self._candidates, limit)
        try:
            fts_future = self._pool.submit(self._keyword_ranking, query, candidates)
            vec_future = self._pool.submit(self._vector_ranking, query, candidates)
        except RuntimeError as exc:  # the executor refuses work after close()
            raise SearchError("The search engine has been shut down") from exc
        # A failed keyword index says so in the state it hands back instead of a ranking.
        (fts_ranking, keyword_match), fts_error = _settle(fts_future, ([], "unavailable"))
        (vec_ranking, passages), vec_error = _settle(vec_future, ([], {}))
        if fts_error is not None and vec_error is not None:
            raise fts_error
        if keyword_match == "no_match" and _is_identifier_lookup(fts_terms(query)):
            # No indexed section contains the identifier: its semantic neighbours name other
            # things, and an agent handed them answers from them (#33). An empty page says so.
            return SearchPage((), keyword_match), False
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
        page: list[int] = []
        stale = False
        for section_id in ordered:
            if len(page) == limit:
                break
            if section_id in hydrated:
                page.append(section_id)
            else:
                stale = True
        # A part of a split section shares its breadcrumb with every other part: say how it
        # begins. Its first stored passage is already plain text - markup stripped, a table
        # row's header restored - so nothing is parsed again here.
        firsts = self._db.first_passages([sid for sid in page if hydrated[sid][0].part_index > 0])
        results: list[SearchResult] = []
        for section_id in page:
            section, document = hydrated[section_id]
            first = firsts.get(section_id)
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
                    part_preview=None if first is None else preview(first),
                )
            )
        return SearchPage(tuple(results), keyword_match), stale

    def _keyword_ranking(self, query: str, limit: int) -> tuple[list[int], KeywordMatch]:
        """BM25 ranking after the gate, and which of the ways to find nothing this was."""
        terms = fts_terms(query)
        if not terms:
            return [], "no_terms"
        # Scoped inside the query, before its LIMIT: no rows means no section in this root
        # contains any searched term - the one state that may say so.
        raw = self._db.fts_search(" OR ".join(terms), limit, self._scope)
        if not raw:
            return [], "no_match"
        hits = self._gate(terms, raw)
        if not hits:
            return [], "filtered"
        # Heading-only sections are signposts: their children carry the same breadcrumb
        # words plus the actual text. They stay only when nothing else matched.
        with_body = self._db.sections_with_passages(hits)
        return [hit for hit in hits if hit in with_body] or hits, "matched"

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
        """Sections by their closest vector, plus each section's best-matching passage.

        Empty when the stored vectors came from other weights than the ones answering
        now: the distance between two models' vectors measures nothing, and returning it
        as a semantic result is worse than returning no semantic result at all. Keyword
        ranking reads no vector and is unaffected, so the search still answers - with the
        half of it that is still true, and `index_status` carries the reason.
        """
        embedding = self._embedder.embed_query(query)
        # After the embedding, never before: the embedder loads lazily and cannot say
        # which weights it is until it has loaded, so asking first would suppress
        # ranking on every first query of a process.
        recorded = self._refuse_foreign_vectors()
        best, passages = self._nearest(embedding, limit)
        # Again, against what was read rather than what was checked: a model *name* change
        # in another process discards every vector and rebuilds it, and a check that
        # happened before those rows were read cannot speak for them.
        if self._db.get_meta(WEIGHTS_META_KEY) != recorded or (
            # Let through only because nothing was stored to disagree with; whatever the
            # lookup found was written since, by weights other than these.
            best and recorded != self._embedder.weights_revision
        ):
            raise SearchError(
                "The index was rebuilt by another model while this search was ranking; "
                "only keyword ranking is used"
            )
        ranking = sorted(best, key=lambda section_id: (best[section_id], section_id))[:limit]
        return ranking, {sid: passages[sid] for sid in ranking if sid in passages}

    def _refuse_foreign_vectors(self) -> str | None:
        """Fail this ranking if the loaded model is not the one that built the vectors.

        Search asks for itself rather than trusting a flag an indexing run would have had
        to write: a cache whose weights changed while no document did leaves indexing a
        clean no-op, and nothing would ever have set that flag. Failing rather than
        returning nothing puts it on the path that already exists for one index being
        unusable - the other index answers alone, and only losing both is an error.

        Returns what was recorded, so the caller can tell whether it still is.
        """
        recorded = self._db.get_meta(WEIGHTS_META_KEY)
        weights = self._embedder.weights_revision
        if recorded is not None and weights != recorded and self._db.count_rows("units_vec") == 0:
            # A revision over no vectors - a run died between claiming it and writing the
            # first one - has nothing to rank against, so nothing to warn about either.
            return recorded
        if recorded is None:
            if weights is None or self._db.count_rows("units_vec") == 0:
                return None  # nothing named on either side, or nothing to rank
            # Vectors no revision vouches for, and weights that can say what they are:
            # nothing says the two are the same model, so they are not ranked together.
            # Recorded, so the next index run loads its model and re-embeds them.
            self._record(
                "No record says which weights built this index's vectors, so they are not "
                "compared with a query: only keyword ranking is used until index_directory "
                "re-embeds them."
            )
        if weights == recorded:
            # A mismatch recorded by another process is left standing, even though these
            # weights agree: it may be the only thing telling the next index run that a
            # repair is pending. That run withdraws it once the whole index agrees.
            return recorded
        if recorded == WEIGHTS_REVOKED:
            # An indexing run is replacing the vectors, and has said so where
            # `index_status` reads it; until it finishes, no weights - old or new - have
            # a whole index to rank against.
            raise SearchError(
                "This index is being re-embedded with other weights, so only keyword "
                "ranking is used until index_directory finishes."
            )
        message = (
            f"This index was built by weights {short_weights(recorded)} and the model "
            f"answering now reports {short_weights(weights)}: the distance "
            "between two models' vectors measures nothing, so only keyword ranking is used "
            "until index_directory re-embeds this documentation root."
        )
        self._record(message)

    def _record(self, message: str) -> NoReturn:
        """Persist why vectors are not ranked, then fail the vector half of this search.

        Persisted, because the answer this query is about to give is half of one, and the
        agent reading it is told the index is healthy by an `index_status` that no
        indexing run will correct - weights can change while no document does.
        """
        # Only where nothing is recorded yet, decided in the write itself: an indexing run's
        # account - which names the directories still to re-index - says more than this
        # query can, and may land between a check and a write.
        self._db.record_weights_mismatch(message, replace=False)
        raise SearchError(message)

    def _nearest(
        self, embedding: list[float], limit: int
    ) -> tuple[dict[int, float], dict[int, str]]:
        """Closest sections and their best passages, restricted to this server's root.

        A vec0 KNN query applies its own ``k`` before anything can filter it, so a scoped
        search widens ``k`` until it has a full page or has seen the whole index. A fixed
        multiplier is not enough: a neighbouring root in the same database can be
        arbitrarily larger than this one.
        """
        ceiling = max(self._db.count_rows("sections"), limit)
        fetch = limit if self._scope is None else min(limit * _SCOPED_OVERFETCH, ceiling)
        while True:
            best: dict[int, float] = dict(self._db.vec_search(embedding, fetch))
            # Nothing came back at all, so there is nothing a wider net can catch: the
            # root is mid-rebuild, or its vectors were discarded and not yet replaced.
            # Without this, each empty pass quadruples the fetch and asks again, all the
            # way up to the size of the corpus, on every scoped query.
            if not best:
                return {}, {}
            passages: dict[int, str] = {}
            for section_id, distance, passage in self._db.unit_search(
                embedding, fetch * _PASSAGES_PER_CANDIDATE
            ):
                passages.setdefault(section_id, passage)  # closest first: keep the best one
                if distance < best.get(section_id, math.inf):
                    best[section_id] = distance
            if self._scope is not None and best:
                allowed = self._db.sections_under(list(best), self._scope)
                best = {sid: distance for sid, distance in best.items() if sid in allowed}
                passages = {sid: text for sid, text in passages.items() if sid in allowed}
            if self._scope is None or len(best) >= limit or fetch >= ceiling:
                return best, passages
            fetch = min(fetch * _SCOPED_OVERFETCH, ceiling)


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
