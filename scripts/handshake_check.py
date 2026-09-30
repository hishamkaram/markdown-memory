"""Start an installed markdown-memory and time its answer to the MCP handshake.

    python scripts/handshake_check.py /path/to/bin/markdown-memory [--expect-version X.Y.Z]

A client gives a new server a few seconds to answer `initialize` - Codex waits 10 by
default - and gives up on it after that. The server once loaded its model before
answering, which took most of that budget on a fast machine; this is what notices if
anything heavy creeps back in front of the handshake. Standard library only, so it runs
against a bare wheel install without the project's environment.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile
import threading
import time
from pathlib import Path

#: Codex's default `startup_timeout_sec`; the tightest limit a supported client sets.
LIMIT_SECONDS = 10.0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("command", help="the markdown-memory executable to start")
    parser.add_argument("--expect-version", help="fail unless serverInfo.version is this")
    arguments = parser.parse_args()

    with tempfile.TemporaryDirectory() as scratch:
        env = {
            **os.environ,
            "MARKDOWN_MEMORY_DB": str(Path(scratch) / "index.db"),
            "MARKDOWN_MEMORY_DOCS_DIR": scratch,
        }
        request = {
            "jsonrpc": "2.0",
            "id": 1,
            "method": "initialize",
            "params": {
                "protocolVersion": "2025-06-18",
                "capabilities": {},
                "clientInfo": {"name": "handshake-check", "version": "0"},
            },
        }
        started = time.monotonic()
        server = subprocess.Popen(
            [arguments.command],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            env=env,
        )
        assert server.stdin is not None and server.stdout is not None
        answer: list[bytes] = []
        reader = threading.Thread(
            target=lambda: answer.append(server.stdout.readline()), daemon=True
        )
        try:
            server.stdin.write((json.dumps(request) + "\n").encode())
            server.stdin.flush()
            reader.start()
            # A server that never answers must fail here, not hang until the CI job's own
            # timeout: wait a little past the limit, then read whatever arrived.
            reader.join(LIMIT_SECONDS + 5)
            line = answer[0] if answer else b""
            elapsed = time.monotonic() - started
        finally:
            server.stdin.close()
            try:
                server.wait(timeout=30)
            except subprocess.TimeoutExpired:
                server.kill()
                server.wait()

    if not line:
        print(f"FAIL: no answer to initialize within {elapsed:.0f} s")
        return 1
    info = json.loads(line).get("result", {}).get("serverInfo")
    if not info:
        print(f"FAIL: the answer is not an initialize result: {line[:200]!r}")
        return 1
    print(f"initialize answered in {elapsed:.2f} s by {info.get('name')} {info.get('version')}")
    if elapsed > LIMIT_SECONDS:
        print(f"FAIL: slower than the {LIMIT_SECONDS:.0f} s a client waits")
        return 1
    if arguments.expect_version and info.get("version") != arguments.expect_version:
        print(f"FAIL: serverInfo.version is not {arguments.expect_version}")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
