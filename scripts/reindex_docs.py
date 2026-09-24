"""Re-index a documentation directory and verify the store afterwards.

    uv run python scripts/reindex_docs.py                  # incremental, configured docs root
    uv run python scripts/reindex_docs.py docs/ --force    # drop + re-embed all of docs/
    uv run python scripts/reindex_docs.py docs/ --db /tmp/index.db --embedder bge-small

``--force`` deletes every indexed document under the directory first, so unchanged files are
embedded again (the SHA-256 cache would otherwise skip them). Documents under other
directories are left alone. Exits non-zero when a file failed to index or the integrity
check finds a problem.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from markdown_memory.config import ServerConfig, resolve_config
from markdown_memory.server import MarkdownMemoryService, configure_logging


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    parser.add_argument("directory", nargs="?", type=Path, help="default: the configured docs root")
    parser.add_argument("--force", action="store_true", help="re-embed unchanged files too")
    parser.add_argument(
        "--db",
        type=Path,
        help="database path (default: MARKDOWN_MEMORY_DB, else the index keyed on the directory)",
    )
    parser.add_argument("--embedder", choices=("embeddinggemma", "bge-small"))
    arguments = parser.parse_args()
    configure_logging("WARNING")

    directory = (arguments.directory or ServerConfig.from_env().docs_dir).expanduser().resolve()
    # The same resolution the server itself does, for the same reason: the default database
    # is keyed on the documentation root, so naming a directory has to re-key it. Verifying
    # one project's index while writing to another's is the failure this script exists to
    # catch, and it would be the one doing it.
    config = resolve_config(db=arguments.db, docs_dir=directory, embedder=arguments.embedder)
    service = MarkdownMemoryService(config)
    try:
        database = service.db
        print(f"database : {database.path}")
        print(f"embedder : {service.embedder.model_name} ({service.embedder.dimension} dims)")
        print(f"directory: {directory}")
        if arguments.force:
            dropped = database.delete_documents(sorted(database.document_hashes(str(directory))))
            print(f"--force  : dropped {dropped} indexed document(s) under the directory")

        report = service.index_directory(str(directory))
        print("\n" + report.summary())

        tables = ("documents", "sections", "sections_fts", "sections_vec", "units", "units_vec")
        counts = {table: database.count_rows(table) for table in tables}
        print("\nrows     : " + ", ".join(f"{name}={count}" for name, count in counts.items()))
        print(
            f"vectors  : {database.embedding_dim} dimensions "
            f"(meta: {database.get_meta('embedding_dim')}, "
            f"model: {database.get_meta('embedding_model')})"
        )
        problems = database.integrity_problems()
    finally:
        service.close()

    for problem in problems:
        print(f"INTEGRITY PROBLEM: {problem}")
    if not problems:
        print("integrity: ok (pages, foreign keys, FTS5 index, passage vectors, dimensions)")
    if report.errors:
        print(f"{len(report.errors)} file(s) failed to index - see ERROR lines above")
    return 1 if problems or report.errors else 0


if __name__ == "__main__":
    logging.captureWarnings(True)
    sys.exit(main())
