"""Vendor the Eval v2 corpus: real documentation, permissively licensed, pinned by commit.

    uv run python scripts/fetch_eval_corpus.py                 # write scripts/eval_data/corpus_v2
    uv run python scripts/fetch_eval_corpus.py --check         # verify what is vendored matches

The evaluation corpus decides what the accuracy numbers mean, so it is real documentation
rather than anything generated: a synthetic corpus and synthetic queries match each other
by semantic mirroring and hide the lexical mismatch that hybrid retrieval exists to solve.

Sources are chosen for the workload this server actually serves - a coding agent asking
about flags, configuration keys, API parameters and runbooks - and for being plain
CommonMark committed upstream, with no build step and no directive dialect (``:::``
admonitions, ``{{#include}}``, Jinja) that this project's users do not have.

Every source is pinned to a commit. Re-running with a new commit changes the benchmark,
so the frozen baseline must be re-recorded; that is a deliberate act, not a refresh.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import re
import sys
import tarfile
import urllib.request
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

EVAL_DATA = Path(__file__).resolve().parent / "eval_data"
CORPUS = EVAL_DATA / "corpus_v2"
MANIFEST = EVAL_DATA / "corpus_v2_sources.json"
LICENCES = EVAL_DATA / "corpus_v2_licenses"
# Anchored at the start of a line: a bare ":::" also appears inside IPv6 addresses, and
# "{{" inside shell or template samples that are themselves the documentation.
DIALECT_MARKERS = (
    re.compile(r"^\s*:::", re.MULTILINE),  # MkDocs / Docusaurus admonitions
    re.compile(r"^\s*\{\{#", re.MULTILINE),  # mdBook includes
    re.compile(r"^\s*\{!", re.MULTILINE),  # MkDocs snippets
    re.compile(r"^\s*\{%", re.MULTILINE),  # Jinja / Liquid
    re.compile(r"^\s*\{\{<", re.MULTILINE),  # Hugo shortcodes
)
MAX_FILE_BYTES = 400_000


@dataclass(frozen=True)
class Source:
    """One upstream documentation set, pinned to a commit."""

    name: str
    repo: str
    commit: str
    licence: str
    kind: str
    include: tuple[str, ...]
    note: str
    # The upstream's own licence files, vendored beside the documentation they cover.
    # Apache-2.0 requires that a NOTICE travel with anything redistributed from a tree
    # that has one, and that a copy of the licence itself go with the copy - naming the
    # licence in a table is not the same as carrying it.
    legal: tuple[str, ...]


SOURCES = (
    Source(
        name="ripgrep",
        repo="BurntSushi/ripgrep",
        commit="3fce3b5bb0236da2df6d99672afb8a719642eca7",
        licence="Unlicense OR MIT",
        kind="cli-guide",
        include=("GUIDE.md", "FAQ.md", "CHANGELOG.md"),
        note="Dense flag prose and a changelog; the flags collide (-C, --context).",
        legal=("COPYING", "LICENSE-MIT", "UNLICENSE"),
    ),
    Source(
        name="cargo",
        repo="rust-lang/cargo",
        commit="cc74001909309fff5322b4d91c29c31ef6139df7",
        licence="MIT OR Apache-2.0",
        kind="cli-reference",
        include=("doc/book/src/commands/",),
        note="Man-page style command reference: SYNOPSIS/OPTIONS sections that repeat "
        "across dozens of commands, which is exactly the near-duplicate case. The "
        "reference chapter is left out to stop one source dominating the corpus.",
        legal=("LICENSE-APACHE", "LICENSE-MIT"),
    ),
    Source(
        name="gh",
        repo="cli/cli",
        commit="0cf1092493af067646fc5f3db9421c6a6ec9c938",
        licence="MIT",
        kind="cli-guide",
        include=("docs/",),
        note="Task-oriented guides and environment variables.",
        legal=("LICENSE",),
    ),
    Source(
        name="compose-spec",
        repo="compose-spec/compose-spec",
        commit="914ec15d1fa498969c0df5c1d672306db3256089",
        licence="Apache-2.0",
        kind="config-reference",
        include=("spec.md", "build.md", "deploy.md", "service.md", "05-services.md"),
        note="Deeply nested declarative keys whose meaning is inherited from parents - "
        "the case that flatters passage vectors least.",
        legal=("LICENSE", "NOTICE"),
    ),
    Source(
        name="prometheus",
        repo="prometheus/prometheus",
        commit="64c05e80ccd15eae229980dcf15fa82bbb513113",
        licence="Apache-2.0",
        kind="config-reference",
        include=("docs/configuration/", "docs/querying/"),
        note="Configuration blocks and query-language reference: long YAML samples with "
        "'#' comments inside fences, which must never be read as headings.",
        legal=("LICENSE", "NOTICE"),
    ),
)


def archive_url(source: Source) -> str:
    return f"https://codeload.github.com/{source.repo}/tar.gz/{source.commit}"


def wanted(source: Source, relative: str) -> bool:
    return relative.endswith(".md") and any(
        relative == pattern or relative.startswith(pattern) or relative.endswith("/" + pattern)
        for pattern in source.include
    )


def fetch(source: Source) -> bytes:
    """The upstream tarball, read once and extracted twice: documentation and licences."""
    with urllib.request.urlopen(archive_url(source), timeout=180) as response:  # noqa: S310
        payload: bytes = response.read()
    return payload


def documents(source: Source, payload: bytes) -> Iterator[tuple[str, str]]:
    """Yield ``(relative path, text)`` for the Markdown this source contributes."""
    yield from _extract(source, payload, legal=False)


def legal_files(source: Source, payload: bytes) -> Iterator[tuple[str, str]]:
    """Yield ``(name, text)`` for the upstream's own licence and NOTICE files."""
    yield from _extract(source, payload, legal=True)


def _extract(source: Source, payload: bytes, *, legal: bool) -> Iterator[tuple[str, str]]:
    with tarfile.open(fileobj=io.BytesIO(payload), mode="r:gz") as archive:
        for member in archive.getmembers():
            if not member.isfile() or member.size > MAX_FILE_BYTES:
                continue
            relative = member.name.split("/", 1)[1] if "/" in member.name else member.name
            if legal:
                if relative not in source.legal:
                    continue
                handle = archive.extractfile(member)
                if handle is not None:
                    yield relative, handle.read().decode("utf-8", errors="replace")
                continue
            if not wanted(source, relative):
                continue
            handle = archive.extractfile(member)
            if handle is None:
                continue
            text = handle.read().decode("utf-8", errors="replace")
            dialect = next((m.pattern for m in DIALECT_MARKERS if m.search(text)), None)
            if dialect is not None:
                print(f"  skip {relative}: line matching {dialect} is not CommonMark")
                continue
            yield relative, text


def digest(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]


def vendor() -> int:
    CORPUS.mkdir(parents=True, exist_ok=True)
    manifest: dict[str, object] = {
        "_about": "Pinned upstream documentation; see corpus_v2_LICENSES.md."
    }
    for source in SOURCES:
        print(f"{source.name}: {source.repo}@{source.commit[:8]}")
        payload = fetch(source)
        files: dict[str, str] = {}
        for relative, text in documents(source, payload):
            destination = CORPUS / source.name / relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_text(text, encoding="utf-8")
            files[relative] = digest(text)
        if not files:
            print(f"  ERROR: {source.name} contributed no files")
            return 1
        legal: dict[str, str] = {}
        for name, text in legal_files(source, payload):
            destination = LICENCES / source.name / name
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_text(text, encoding="utf-8")
            legal[name] = digest(text)
        missing = [name for name in source.legal if name not in legal]
        if missing:
            # Silence here would be the whole risk: the obligation is carried by files
            # that are present, so a rename upstream must fail the refresh, not pass it.
            print(f"  ERROR: {source.name} is missing {', '.join(missing)}")
            return 1
        print(f"  {len(files)} file(s), {len(legal)} licence file(s)")
        manifest[source.name] = {
            "repo": source.repo,
            "commit": source.commit,
            "licence": source.licence,
            "kind": source.kind,
            "note": source.note,
            "files": files,
            "legal": legal,
        }
    MANIFEST.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (EVAL_DATA / "corpus_v2_LICENSES.md").write_text(licences(), encoding="utf-8")
    print(f"\nwrote {MANIFEST.relative_to(Path.cwd())}")
    return 0


def licences() -> str:
    lines = [
        "# Vendored documentation",
        "",
        "The evaluation corpus is third-party documentation, copied verbatim at the commit",
        "recorded in `corpus_v2_sources.json` and used here only to measure retrieval",
        "accuracy. Each set keeps its own licence; none of it is part of the",
        "markdown-memory package, which is MIT (see `LICENSE` at the repository root).",
        "",
        "Each upstream's own licence files - and its NOTICE, where it has one - are",
        "vendored verbatim beside this table in `corpus_v2_licenses/<set>/`, taken from",
        "the same pinned commit as the documentation. Apache-2.0 asks that the licence",
        "travel with the copy and that a NOTICE be carried into anything redistributed",
        "from a tree that has one; naming the licence in a table is not that.",
        "",
        "| Set | Upstream | Commit | Licence | Carried verbatim | Why it is here |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    for source in SOURCES:
        carried = ", ".join(
            f"[{name}](corpus_v2_licenses/{source.name}/{name})" for name in source.legal
        )
        lines.append(
            f"| `{source.name}` | {source.repo} | `{source.commit[:12]}` | {source.licence} "
            f"| {carried} | {source.note} |"
        )
    lines.append("")
    lines.append("Refresh with `uv run python scripts/fetch_eval_corpus.py`. Moving a commit")
    lines.append("changes the benchmark: the frozen baseline has to be re-recorded with it.")
    return "\n".join(lines) + "\n"


def check() -> int:
    if not MANIFEST.exists():
        print(f"{MANIFEST} is missing; run without --check to vendor the corpus")
        return 1
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    problems = 0
    for source in SOURCES:
        recorded = manifest.get(source.name)
        if not isinstance(recorded, dict) or recorded.get("commit") != source.commit:
            print(f"{source.name}: manifest does not match the pinned commit")
            problems += 1
            continue
        for relative, expected in recorded["files"].items():
            path = CORPUS / source.name / relative
            if not path.exists():
                print(f"{source.name}/{relative}: missing")
                problems += 1
            elif digest(path.read_text(encoding="utf-8")) != expected:
                print(f"{source.name}/{relative}: edited since it was vendored")
                problems += 1
        for name, expected in recorded.get("legal", {}).items():
            path = LICENCES / source.name / name
            if not path.exists():
                print(f"{source.name}/{name}: licence file missing")
                problems += 1
            elif digest(path.read_text(encoding="utf-8")) != expected:
                print(f"{source.name}/{name}: licence file edited since it was vendored")
                problems += 1
        for name in source.legal:
            if name not in recorded.get("legal", {}):
                print(f"{source.name}/{name}: not vendored; re-run without --check")
                problems += 1

        # A file the manifest does not mention is one nothing verifies: it survives an
        # upstream deletion, keeps being indexed, and would be redistributed with the rest.
        for tree, expected in (
            (CORPUS / source.name, set(recorded["files"])),
            (LICENCES / source.name, set(recorded.get("legal", {}))),
        ):
            if not tree.is_dir():
                continue
            found = {str(path.relative_to(tree)) for path in tree.rglob("*") if path.is_file()}
            for stray in sorted(found - expected):
                print(f"{tree.name}/{stray}: not in the manifest; delete it or re-vendor")
                problems += 1

    # Walking each known set catches a stray inside one, but not a whole set that was
    # retired from SOURCES: its directory is simply never visited, and everything under it
    # keeps being indexed and redistributed. So sweep the two roots themselves.
    known = {source.name for source in SOURCES}
    for root in (CORPUS, LICENCES):
        if not root.is_dir():
            continue
        for entry in sorted(root.iterdir()):
            if entry.name not in known:
                print(f"{root.name}/{entry.name}: not a set this script vendors; delete it")
                problems += 1

    # The attribution table is generated, so it can drift from the sources it describes -
    # a wrong licence name or a dead link would otherwise pass every other check here.
    # Compared as bytes: decoding would let a line-ending-only rewrite through.
    attribution = EVAL_DATA / "corpus_v2_LICENSES.md"
    if not attribution.exists():
        print(f"{attribution.name}: missing; re-run without --check")
        problems += 1
    elif attribution.read_bytes() != licences().encode("utf-8"):
        print(f"{attribution.name}: stale; re-run without --check to regenerate it")
        problems += 1

    print("corpus matches the manifest" if not problems else f"{problems} problem(s)")
    return 1 if problems else 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    parser.add_argument("--check", action="store_true", help="verify the vendored copy")
    return check() if parser.parse_args().check else vendor()


if __name__ == "__main__":
    sys.exit(main())
