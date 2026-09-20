"""Create the old-release database fixtures that tests/ migrates forward.

    uv run python scripts/make_db_fixture.py tests/fixtures/databases

A database written by an earlier release is the one input the test suite cannot
reconstruct from the current code by accident: simulating it (dropping tables from a
current database) tests the simulation, not the upgrade. These fixtures are built from
the frozen schema statements of each version plus rows in the shape that version wrote,
and are committed so every later release keeps having to migrate them.

They are stored gzipped: sqlite-vec pre-allocates a chunk per vector table, so a
four-section database is megabytes of mostly zeroes but kilobytes compressed. Tests have
to copy a fixture before opening it anyway - opening one migrates it in place.

Regenerate a fixture only to add a *new* version. Editing an existing one silently
weakens the upgrade tests.
"""

from __future__ import annotations

import argparse
import gzip
import shutil
import sqlite3
import sys
import tempfile
from pathlib import Path

import sqlite_vec

from markdown_memory.db import _schema_v1, _schema_v2, serialize_embedding

DIMENSION = 384
DOCUMENTS = (
    ("/docs/alpha.md", "Alpha", "hash-alpha", 1_700_000_000),
    ("/docs/beta.md", "Beta", "hash-beta", 1_700_000_100),
)
SECTIONS = (
    (1, "Overview", 1, "Alpha > Overview", "# Overview\n\nalpha overview body", 1, 3),
    (1, "Details", 2, "Alpha > Details", "## Details\n\nalpha details body", 5, 7),
    (2, "Setup", 1, "Beta > Setup", "# Setup\n\nbeta setup body", 1, 3),
    (2, "Usage", 2, "Beta > Usage", "## Usage\n\nbeta usage body", 5, 7),
)
UNITS = ("alpha overview body", "alpha details body", "beta setup body", "beta usage body")


def vector(seed: int) -> bytes:
    """A deterministic unit vector; the values only have to be finite and non-zero."""
    return serialize_embedding(
        [1.0 if index == seed % DIMENSION else 0.0 for index in range(DIMENSION)]
    )


def build(archive: Path, version: int) -> None:
    if archive.exists():
        raise SystemExit(f"{archive} already exists; delete it first if that is really intended")
    with tempfile.TemporaryDirectory() as workspace:
        path = Path(workspace) / archive.with_suffix("").name
        write(path, version)
        with path.open("rb") as raw, gzip.open(archive, "wb", compresslevel=9) as packed:
            shutil.copyfileobj(raw, packed)
    size = archive.stat().st_size
    print(f"wrote {archive} (schema v{version}, {len(SECTIONS)} sections, {size / 1024:.0f} KiB)")


def write(path: Path, version: int) -> None:
    conn = sqlite3.connect(path)
    conn.enable_load_extension(True)
    sqlite_vec.load(conn)
    conn.enable_load_extension(False)
    try:
        for statement in _schema_v1(DIMENSION):
            conn.execute(statement)
        conn.execute("INSERT INTO meta(key, value) VALUES ('embedding_dim', ?)", (str(DIMENSION),))
        if version >= 2:
            for statement in _schema_v2(DIMENSION):
                conn.execute(statement)
        for file_path, title, content_hash, modified in DOCUMENTS:
            conn.execute(
                "INSERT INTO documents(file_path, title, content_hash, last_modified) "
                "VALUES (?, ?, ?, ?)",
                (file_path, title, content_hash, modified),
            )
        for number, (doc_id, heading, level, heading_path, content, start, end) in enumerate(
            SECTIONS, start=1
        ):
            conn.execute(
                "INSERT INTO sections(id, doc_id, heading_title, heading_level, heading_path, "
                "content, start_line, end_line, part_index) VALUES (?, ?, ?, ?, ?, ?, ?, ?, 0)",
                (number, doc_id, heading, level, heading_path, content, start, end),
            )
            conn.execute(
                "INSERT INTO sections_vec(section_id, embedding) VALUES (?, ?)",
                (number, vector(number)),
            )
            if version >= 2:
                conn.execute(
                    "INSERT INTO units(id, section_id, ordinal, content) VALUES (?, ?, 0, ?)",
                    (number, number, UNITS[number - 1]),
                )
                conn.execute(
                    "INSERT INTO units_vec(unit_id, embedding) VALUES (?, ?)",
                    (number, vector(100 + number)),
                )
        # Deliberately no 'next_section_id': releases up to this point never wrote one.
        conn.execute(f"PRAGMA user_version = {version}")
        conn.commit()
        conn.execute("VACUUM")  # vec0 pre-allocates; a committed fixture should stay small
    finally:
        conn.close()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    parser.add_argument("directory", type=Path, help="where the fixtures are written")
    arguments = parser.parse_args()
    arguments.directory.mkdir(parents=True, exist_ok=True)
    for version in (1, 2):
        build(arguments.directory / f"v{version}_seed.db.gz", version)
    return 0


if __name__ == "__main__":
    sys.exit(main())
