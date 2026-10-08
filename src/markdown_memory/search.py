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

import dataclasses
import logging
import math
import re
import threading
from collections import Counter
from collections.abc import Iterable, Mapping, Sequence
from concurrent.futures import Future, ThreadPoolExecutor
from itertools import pairwise
from typing import NamedTuple, NoReturn, TypeVar

from markdown_memory.db import MODEL_META_KEY, WEIGHTS_META_KEY, WEIGHTS_REVOKED, Database
from markdown_memory.discovery import walk_order
from markdown_memory.embedders import Embedder, short_weights
from markdown_memory.exceptions import MarkdownMemoryError, SearchError
from markdown_memory.models import (
    Excerpt,
    KeywordMatch,
    SearchPage,
    SearchResult,
    Section,
    estimate_tokens,
    preview,
)
from markdown_memory.parser import (
    MAX_UNITS_PER_SECTION,
    MarkdownParser,
    Passage,
    ends_inside_fence,
    join_parts,
)

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
# Keyword candidates an identifier lookup checks for the identifier itself (#75). The tokenizer
# reads `--pre` as `pre`, so prose about `pre_start` can fill the first 20; the `--pre` section
# of the measured corpus sat at 25.
LITERAL_CANDIDATES = 200
# The top hit's excerpt (#76): the matched block, one before it and up to three after.
BLOCKS_BEFORE = 1
BLOCKS_AFTER = 3
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
        _called(term)
        for term in _sanitize(query).split()
        if any(character.isalnum() for character in term)
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


def _unwrapped(term: str) -> str:
    """``term`` without the quotes or backticks around it, or the punctuation ending a sentence."""
    return term.strip("\"'`").rstrip("?!,;:").removesuffix(".").strip("\"'`")


def _is_plain_call(term: str) -> bool:
    """`rate()`: a name and an empty call, as an agent writes a function it means."""
    return re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*\(\)", _unwrapped(term)) is not None


# A name, `(`, and something in it; the `)` may have gone to the next whitespace piece.
_CALL_WITH_ARGUMENTS = re.compile(r"([A-Za-z_][A-Za-z0-9_]*)\((.+)")


def _called(term: str) -> str:
    """`rate(x[5m])` is searched as the call `rate()` (#96).

    Written whole, it is one FTS5 phrase - `rate x 5m` - that no document holds unless it uses
    that very metric and range; and as the digit makes it an identifier, finding nothing empties
    the page. The function is what the documentation describes, so the name is what is looked
    for, with the call boundary #79 gave `rate()`. Only when the arguments read as code - a
    digit, an identifier mark or a bracket: `flag(s)` and `abs(v)` stay what they were. A piece
    closing more than it opens - `rate(x[5m]))` of `histogram_quantile(0.9, rate(x[5m]))` - is an
    argument of the call asked about, and is not made a call of its own to compete with it.
    """
    call = _CALL_WITH_ARGUMENTS.fullmatch(_unwrapped(term))
    if (
        call is None
        or call[2].count(")") > call[2].count("(") + 1
        or not any(
            character.isdigit() or character in _IDENTIFIER_MARKS or character in "[]{}"
            for character in call[2]
        )
    ):
        return term
    return call[1] + "()"


def _is_identifier(quoted_term: str) -> bool:
    """Spelled like a flag, path, environment variable, constant, camelCase key, version or call.

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
        or _is_plain_call(term)
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
    flags (``--all``, ``-i``), decorated names (``@Before``, ``IS_ON``), calls (``all()``)
    and upper-case keywords (``WHERE``, ``NOT NULL``) are what keyword search exists for.
    """
    word = term.strip(_SENTENCE_PUNCTUATION)
    if _is_plain_call(term) or not word.isalpha() or (len(word) > 1 and word.isupper()):
        return False
    return word.lower() in _STOPWORDS


# Characters that continue an identifier: `--pre` is not found inside `--pre-glob`.
_IDENTIFIER_EDGE = "A-Za-z0-9_-"


class _Literal:
    """Where an identifier term occurs as itself, rather than as the words the tokenizer sees.

    FTS5 splits `GH_REPO` into the phrase `gh repo` and stems `histogram_quantiles` like
    `histogram_quantile`, so a keyword hit for an identifier can be prose that never names it.
    Matched without regard to case; `exact` says whether the spelling agreed too.
    """

    def __init__(self, quoted_term: str) -> None:
        # `fts_terms` quotes each term FTS5-style; an agent may add quotes or backticks, and
        # end a sentence after them (`GH_REPO`.), so they are stripped on both sides of that.
        term = quoted_term[1:-1].replace('""', '"').strip("'`\"")
        term = term.rstrip("?!,;:").removesuffix(".").strip("'`\"")
        call = len(term) > 2 and term.endswith("()")
        self.name = term[:-2].strip("'`\"") if call else term  # `name`() quotes the name only
        # `name()` is a call: `histogram_quantile(φ, v)` names it, `histogram_quantiles(` does
        # not. Otherwise a `.` may follow (a sentence ends) but not start another component.
        tail = r"(?=\s*\()" if call else rf"(?![{_IDENTIFIER_EDGE}])(?!\.[A-Za-z0-9_])"
        pattern = rf"(?<![{_IDENTIFIER_EDGE}]){re.escape(self.name)}{tail}"
        self._loose = re.compile(pattern, re.IGNORECASE)
        self._exact = re.compile(pattern)

    def found(self, text: str) -> bool:
        return self._loose.search(text) is not None

    def exact(self, text: str) -> bool:
        return self._exact.search(text) is not None

    def heads(self, title: str) -> bool:
        return _bare(title) == _bare(self.name)


def _bare(text: str) -> str:
    return text.replace("`", "").strip().removesuffix("()").strip().lower()


def select_anchor(
    terms: Sequence[str], passages: Sequence[str], ordinal: int | None, heading: str = ""
) -> int | None:
    """Which passage of the top hit its excerpt is centred on, or None for the whole section.

    A section whose heading names everything the query asks for is the answer as a whole,
    and is not cut. An identifier lookup is anchored on the first passage that names the
    identifier, which is where a document introduces it: if none does - the heading may name
    it alone - an excerpt would hide what was asked for. Any other query is anchored on the
    passage that won the vector ranking, else the one holding the most terms. A query of
    nothing but stopwords says nothing about where in a section to look.
    """
    if not terms or all(_is_stopword(term.replace("`", "")) for term in terms):
        return None
    literals = [_Literal(term) for term in terms]
    if all(literal.found(heading) for literal in literals):
        return None
    found = [sum(literal.found(text) for literal in literals) for text in passages]
    if _is_identifier_lookup(terms):
        return next((index for index, count in enumerate(found) if count), None)
    if ordinal is not None and 0 <= ordinal < len(passages):
        return ordinal
    best = max(found, default=0)
    return found.index(best) if best else None


def excerpt_lines(content: str, passages: Sequence[Passage], anchor: int) -> tuple[int, int] | None:
    """The 0-based, end-exclusive lines of ``content`` an excerpt around ``anchor`` shows.

    The anchor's block, the block before it and up to three after, widened to whole lines: a
    passage that matches a question usually states the problem, and the answer - the script,
    the parameter, the rest of the syntax - follows it (#76). The pieces of a long block are one
    block, and so is a list, whose items rarely stand without the sentence that introduces it.
    A table row brings its table's header with it, contiguously.
    None whenever the excerpt would not be safe or would not be smaller: three passages or fewer
    (the window is lopsided, so it can leave out the first of three), a window covering every
    passage, a block without lines, a fence left open.
    """
    keys = [passage.listing or passage.lines for passage in passages]
    starts = [i for i in range(len(keys)) if i == 0 or keys[i] != keys[i - 1]]
    at = max(index for index, start in enumerate(starts) if start <= anchor)
    ends = [*starts[1:], len(passages)]
    low, high = max(0, at - BLOCKS_BEFORE), min(len(starts) - 1, at + BLOCKS_AFTER)
    window = passages[starts[low] : ends[high]]
    spans = [passage.lines for passage in window]
    if len(passages) <= 3 or any(span is None for span in spans):
        return None
    first = min(span[0] for span in spans if span is not None)
    end = max(span[1] for span in spans if span is not None)
    first = min([first, *(passage.table for passage in window if passage.table is not None)])
    lines = content.split("\n")
    while end > first and not lines[end - 1].strip():
        end -= 1  # markdown-it counts the blank line after a list item as part of it
    while first < end and not lines[first].strip():
        first += 1
    if all(p.lines is not None and first <= p.lines[0] and p.lines[1] <= end for p in passages):
        return None
    text = "\n".join(lines[first:end])
    if ends_inside_fence(text) or estimate_tokens(text) >= estimate_tokens(content):
        return None
    return first, end


class _Keyword(NamedTuple):
    """One keyword ranking, and what an identifier lookup learnt making it (#75)."""

    ranking: list[int]
    match: KeywordMatch
    literal: frozenset[int] = frozenset()  # sections that contain the identifier itself
    headings: tuple[int, ...] = ()  # those headed by it, in keyword order
    stale: bool = False  # a candidate vanished while it was being checked


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
        # A parser per thread: `MarkdownParser` holds a `MarkdownIt` with mutable state.
        self._parsers = threading.local()
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
            fts_future = self._pool.submit(self._keyword_pass, query, candidates)
            vec_future = self._pool.submit(self._vector_ranking, query, candidates)
        except RuntimeError as exc:  # the executor refuses work after close()
            raise SearchError("The search engine has been shut down") from exc
        # A failed keyword index says so in the state it hands back instead of a ranking.
        keyword, fts_error = _settle(fts_future, _Keyword([], "unavailable"))
        fts_ranking, keyword_match = keyword.ranking, keyword.match
        (vec_ranking, passages), vec_error = _settle(vec_future, ([], {}))
        if fts_error is not None and vec_error is not None:
            raise fts_error
        if keyword_match == "no_match" and _is_identifier_lookup(fts_terms(query)):
            # No indexed section contains the identifier: its semantic neighbours name other
            # things, and an agent handed them answers from them (#33). An empty page says so.
            return SearchPage((), keyword_match), keyword.stale
        for name, error in (("keyword", fts_error), ("vector", vec_error)):
            if error is not None:
                logger.warning("%s search failed; using the other index only: %s", name, error)

        scores = reciprocal_rank_fusion([fts_ranking, vec_ranking], self._rrf_k)
        fts_ranks = {section_id: rank for rank, section_id in enumerate(fts_ranking, start=1)}
        vec_ranks = {section_id: rank for rank, section_id in enumerate(vec_ranking, start=1)}
        ordered = self._in_order({section_id: -score for section_id, score in scores.items()})
        if keyword.literal:
            # An identifier lookup is answered by a section that names the identifier: a
            # vector-only neighbour must not tie with one at 1/61 and win on its place. One
            # headed by it comes first, so mentions with vector support cannot push it off.
            ordered = [
                *keyword.headings,
                *(sid for sid in ordered if sid in keyword.literal - set(keyword.headings)),
                *(sid for sid in ordered if sid not in keyword.literal),
            ]

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
                    matched_passage=passages[section_id][1] if section_id in passages else None,
                    part_preview=None if first is None else preview(first),
                )
            )
        if results:
            section = hydrated[page[0]][0]
            ordinal = passages[page[0]][0] if page[0] in passages else None
            excerpt = self._excerpt(section, fts_terms(query), ordinal)
            results[0] = dataclasses.replace(results[0], excerpt=excerpt)
        return SearchPage(tuple(results), keyword_match), stale or keyword.stale

    def _in_order(self, values: Mapping[int, float]) -> list[int]:
        """Section ids by ascending value; an exact tie goes to the one the walk reaches first.

        Not to the lower id (#103): ids follow the walk only on a fresh build. An edited
        document is stored again under new ids and would lose every tie it was in, so the
        same documentation would rank one way or another depending on its edit history.
        Positions are read only for ids that share a value. One deleted since the ranking
        was taken goes after its tie group; fetching the page then finds it gone and
        ranks again.
        """
        shared = Counter(values.values())
        tied = [sid for sid, value in values.items() if shared[value] > 1]
        positions = self._db.section_positions(tied) if tied else {}

        def key(sid: int) -> tuple[float, tuple[object, ...]]:
            position = positions.get(sid)
            if position is None:
                return values[sid], (1,)
            file_path, start_line, part_index = position
            return values[sid], (0, walk_order(file_path), start_line, part_index)

        return sorted(values, key=key)

    def _excerpt(
        self, section: Section, terms: Sequence[str], ordinal: int | None
    ) -> Excerpt | None:
        """The top hit's matched passage and its neighbours, when that is safe and smaller (#76).

        The passages are cut again from the stored text, by the parser that cut them for
        indexing, and used only while they are exactly what was stored: an index built by
        another version, or a part whose passages were cut with a table header it does not
        hold, gets the whole section, as does anything that raises.
        """
        try:
            stored = self._db.units_of(section.id)
            if not stored or len(stored) >= MAX_UNITS_PER_SECTION:
                return None
            parser = self._parser()
            passages = None
            for skip in (True, False):  # a preamble has no heading line to skip
                cut = parser.passages(section.content, skip_heading=skip)
                if [passage.text for passage in cut] == stored:
                    passages = cut
                    break
            if passages is None or (section.part_index and self._cut_through(section, parser)):
                return None
            anchor = select_anchor(terms, stored, ordinal, section.heading_title)
            window = None if anchor is None else excerpt_lines(section.content, passages, anchor)
        except Exception:  # an excerpt is an optimisation: whatever fails sends the section
            logger.debug("No excerpt for section %s", section.id, exc_info=True)
            return None
        if window is None:
            return None
        first, end = window
        text = "\n".join(section.content.split("\n")[first:end])
        return Excerpt(section.start_line + first, section.start_line + end - 1, text)

    def _cut_through(self, part: Section, parser: MarkdownParser) -> bool:
        """True when a boundary of this part falls inside a block of the section it was cut from.

        The splitter cuts a fence only when it alone exceeds a part, never tracks HTML, and
        cuts an over-long line at a space: a piece from inside any of them parses as
        something else, so its passages would point at the wrong lines.
        """
        parts = [s for s in self._db.get_sections(part.doc_id) if s.base_path == part.base_path]
        offset = 0
        for previous, current in zip([None, *parts], parts, strict=False):
            if previous is not None:
                offset += max(0, current.start_line - previous.end_line)
            if current.id == part.id:
                return parser.cuts_a_block(join_parts(parts), [offset, offset + len(part.content)])
            offset += len(current.content)
        return True  # the part is gone: nothing vouches for its lines

    def _parser(self) -> MarkdownParser:
        parser: MarkdownParser | None = getattr(self._parsers, "parser", None)
        if parser is None:
            parser = self._parsers.parser = MarkdownParser()
        return parser

    def _keyword_pass(self, query: str, limit: int) -> _Keyword:
        """Keyword ranking; for an identifier lookup, of the sections that name it (#75).

        Anything that finds no such section - or finds the identifier everywhere, which
        makes it vocabulary - is answered exactly as before, with the limit asked for.
        """
        terms = fts_terms(query)
        stale = False
        if _is_identifier_lookup(terms):
            wide, match = self._keyword_ranking(query, max(limit, LITERAL_CANDIDATES))
            if match == "no_match":  # nothing in scope contains the terms, at any limit
                return _Keyword(wide, match)
            if match == "matched":
                literal = self._literal_ranking(terms, wide, limit)
                if literal.ranking:
                    return literal
                stale = literal.stale  # a vanished candidate may have been the one naming it
        return _Keyword(*self._keyword_ranking(query, limit), stale=stale)

    def _literal_ranking(self, terms: Sequence[str], ranking: list[int], limit: int) -> _Keyword:
        """The candidates that contain an identifier term itself, best first; empty if none do.

        Its own heading and text only: a breadcrumb names every section under it. A term
        found in more sections than an identifier may be (the gate's rarity rule) is
        vocabulary - `HTTP`, `API` - and is not looked for.
        """
        hydrated = self._db.get_sections_with_documents(ranking)
        texts = {
            sid: (
                hydrated[sid][0],
                hydrated[sid][0].heading_title + "\n" + hydrated[sid][0].content,
            )
            for sid in ranking
            if sid in hydrated
        }
        stale = len(texts) < len(ranking)
        if not texts:
            return _Keyword([], "matched", stale=stale)
        total = self._db.count_rows("sections")
        rare = max(IDENTIFIER_MAX_SECTIONS, int(total * IDENTIFIER_MAX_SHARE))
        literals = [_Literal(term) for term in terms]
        found = {
            literal: {sid for sid, (_, text) in texts.items() if literal.found(text)}
            for literal in literals
        }

        def spread(term: str) -> float:
            """Sections in scope matching the term, per checked candidate matching it.

            The candidates are a sample when the keyword index matched more sections than
            were checked; what the sample found scales up by this. The index's count alone
            cannot judge rarity - `--files` matches "files" 495 times and names the flag 7.
            """
            checked = max(1, len(self._db.fts_matching(term, list(texts))))
            everywhere = self._db.fts_document_frequency(term)  # every root in the database
            if everywhere <= checked:
                return 1.0
            return max(1.0, len(self._db.fts_search(term, everywhere, self._scope)) / checked)

        wanted = [
            literal
            for literal, term in zip(literals, terms, strict=True)
            if found[literal] and len(found[literal]) * spread(term) <= rare
        ]
        if not wanted:
            return _Keyword([], "matched", stale=stale)
        order = {sid: rank for rank, sid in enumerate(ranking)}

        def key(sid: int) -> tuple[bool, int, int, bool, int]:
            section, text = texts[sid]
            named = [literal for literal in wanted if sid in found[literal]]
            heads = any(literal.heads(section.heading_title) for literal in named)
            exact = any(literal.exact(text) for literal in named)
            # Headed by the identifier first, the first part of a split one first among
            # those; then naming more of the terms, in the spelling asked for, then BM25.
            return (
                not heads,
                section.part_index if heads else 0,
                -len(named),
                not exact,
                order[sid],
            )

        best = sorted(set().union(*(found[literal] for literal in wanted)), key=key)[:limit]
        headings = tuple(sid for sid in best if not key(sid)[0])
        return _Keyword(best, "matched", frozenset(best), headings, stale)

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

    def _vector_ranking(
        self, query: str, limit: int
    ) -> tuple[list[int], dict[int, tuple[int, str]]]:
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
        model = self._db.get_meta(MODEL_META_KEY)
        recorded = self._refuse_foreign_vectors()
        best, passages = self._nearest(embedding, limit)
        # Again, against what was read rather than what was checked: another process may
        # re-embed the index, or fill an empty one, while this one ranks, and a check that
        # happened before those rows were read cannot speak for them. The model name is
        # compared too, because when neither model names its weights it is all that
        # changes (#91).
        if (
            self._db.get_meta(WEIGHTS_META_KEY) != recorded
            or self._db.get_meta(MODEL_META_KEY) != model
            or (
                # Let through only because nothing was stored to disagree with; whatever the
                # lookup found was written since, by weights other than these.
                best and recorded != self._embedder.weights_revision
            )
        ):
            raise SearchError(
                "The index was rebuilt by another model while this search was ranking; "
                "only keyword ranking is used"
            )
        ranking = self._in_order(best)[:limit]
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
            if weights is None:
                # Nothing named on either side, so only the model's name can tell (#91).
                # Not recorded: the index status derives the same sentence from the same
                # facts, so it goes the moment they do - a recorded one would outlive a
                # return to the old model, which no run could withdraw.
                renamed = unnamed_rename(self._db, self._embedder)
                if renamed is not None:
                    raise SearchError(renamed)
                return None
            if not self._db.has_vectors():
                return None  # nothing to rank
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
    ) -> tuple[dict[int, float], dict[int, tuple[int, str]]]:
        """Closest sections and their best passages - ordinal and text - within this root.

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
            passages: dict[int, tuple[int, str]] = {}
            for section_id, distance, passage, ordinal in self._db.unit_search(
                embedding, fetch * _PASSAGES_PER_CANDIDATE
            ):
                passages.setdefault(section_id, (ordinal, passage))  # closest first: the best
                if distance < best.get(section_id, math.inf):
                    best[section_id] = distance
            if self._scope is not None and best:
                allowed = self._db.sections_under(list(best), self._scope)
                best = {sid: distance for sid, distance in best.items() if sid in allowed}
                passages = {
                    sid: best_passage for sid, best_passage in passages.items() if sid in allowed
                }
            if self._scope is None or len(best) >= limit or fetch >= ceiling:
                return best, passages
            fetch = min(fetch * _SCOPED_OVERFETCH, ceiling)


_R = TypeVar("_R")


def unnamed_rename(db: Database, embedder: Embedder) -> str | None:
    """Why this model's vectors may not meet the stored ones, when only a name can tell.

    A model that cannot say which weights it runs, configured over vectors another model
    built: no revision can tell the two apart, so the stored model name is all there is
    (#91). Indexing refuses to write in that state, so it lasts until the person acts, and
    this is computed rather than stored so that it ends exactly when they do.
    """
    if embedder.weights_revision is not None:
        return None
    previous = db.get_meta(MODEL_META_KEY)
    if previous in {None, embedder.model_name} or not db.has_vectors():
        return None
    return (
        f"This index was built by {previous}, and the configured model {embedder.model_name} "
        "has not said which weights it runs, so their vectors are not compared: only keyword "
        f"ranking is used. Configure {previous} again, point --db / MARKDOWN_MEMORY_DB at "
        f"another file, or delete {db.path} with its -wal and -shm files while no "
        "markdown-memory process uses it; the next index_directory rebuilds it."
    )


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
