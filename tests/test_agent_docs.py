"""Agent-facing docs and skills must describe the code that actually exists."""

from __future__ import annotations

import importlib.util
import json
import re
import subprocess
import sys
import tomllib
from pathlib import Path
from types import ModuleType

import pytest
from fakes import FakeEmbedder, vectors_for

from markdown_memory.config import ServerConfig
from markdown_memory.db import Database
from markdown_memory.embedders import Embedder
from markdown_memory.model_cache import GEMMA_MODEL_FILE, GEMMA_REVISION
from markdown_memory.models import SectionDraft
from markdown_memory.server import MarkdownMemoryService, create_server

ROOT = Path(__file__).parent.parent
#: Where the README's images live for every renderer, GitHub's and PyPI's alike.
RAW_MAIN = "https://raw.githubusercontent.com/hishamkaram/markdown-memory/main/"
AGENT_FILES = ("CLAUDE.md", "AGENTS.md", ".cursorrules")
SKILLS = ("run-eval", "reindex-docs", "test-regression")
# The gate, in order. `scripts/check.sh` runs it locally and `.github/workflows/gate.yml`
# runs the same list in CI; the tests below hold both to this one definition.
GATE_STEPS = (
    "uv run python scripts/fetch_eval_corpus.py --check",
    "uv run ruff check .",
    "uv run ruff format --check .",
    "uv run mypy --strict src/",
    "uv run pytest -q",
    "uv run python scripts/live_test.py",
)
# Every runner the gate uses, and what it actually is. A label says neither: `macos-latest`
# has been an Intel machine and is an Apple Silicon one now, and `platform.machine()` spells
# the same architecture `aarch64` on Linux and `arm64` on Darwin. Written down once, and the
# workflow, the classifiers, the README and the job's own assertion are all held to it.
GATE_RUNNERS = {
    "ubuntu-latest": ("POSIX :: Linux", "x86_64"),
    "ubuntu-24.04-arm": ("POSIX :: Linux", "aarch64"),
    "macos-latest": ("MacOS", "arm64"),
}
NAVIGATION_BLOCK = re.compile(
    r"<!-- markdown-memory:navigation-rules:start -->.*?"
    r"<!-- markdown-memory:navigation-rules:end -->",
    re.DOTALL,
)


def _gate_job() -> str:
    """Just the `gate` job's own text.

    The runner checks have to be scoped to it. Read over the whole file, the `package`
    job's plain `runs-on: ubuntu-latest` stands in for the gate's matrix dimension, and
    swapping the two jobs' runners leaves every label still present and the check green.
    """
    workflow = (ROOT / ".github/workflows/gate.yml").read_text(encoding="utf-8")
    jobs = workflow[workflow.index("\njobs:\n") :]
    blocks = re.split(r"^  (?=\w[\w-]*:$)", jobs, flags=re.M)
    gate = [block for block in blocks if block.startswith("gate:")]
    assert len(gate) == 1, f"expected exactly one `gate` job, found {len(gate)}"
    return gate[0]


def load_script(name: str) -> ModuleType:
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / f"{name}.py")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def all_agent_text() -> dict[str, str]:
    files = [ROOT / name for name in AGENT_FILES]
    files += [ROOT / ".claude" / "skills" / skill / "SKILL.md" for skill in SKILLS]
    return {str(path.relative_to(ROOT)): path.read_text(encoding="utf-8") for path in files}


class TestNavigationRules:
    def test_the_rule_block_is_identical_in_every_agent_file(self) -> None:
        blocks = {}
        for name in AGENT_FILES:
            match = NAVIGATION_BLOCK.search((ROOT / name).read_text(encoding="utf-8"))
            assert match is not None, f"{name} has no navigation-rules block"
            blocks[name] = match.group(0)
        assert len(set(blocks.values())) == 1, "navigation rules drifted between agent files"

    def test_rules_state_the_workflow_in_order(self) -> None:
        match = NAVIGATION_BLOCK.search((ROOT / "CLAUDE.md").read_text(encoding="utf-8"))
        assert match is not None
        block = match.group(0)
        assert "Never dump whole files" in block and "150 lines" in block
        positions = [
            block.index("1. **`search_docs("),
            block.index("2. **`get_document_outline("),
            block.index("3. **`read_section("),
        ]
        assert positions == sorted(positions)
        assert "matched_passage" in block and "include_subsections" in block

    async def test_every_tool_the_rules_mention_exists_with_those_parameters(
        self, tmp_path: Path
    ) -> None:
        service = MarkdownMemoryService(
            ServerConfig(db_path=tmp_path / "d.db", docs_dir=tmp_path), embedder=FakeEmbedder()
        )
        try:
            tools = {t.name: t for t in await create_server(service=service).list_tools()}
        finally:
            service.close()
        block = NAVIGATION_BLOCK.search((ROOT / "CLAUDE.md").read_text(encoding="utf-8"))
        assert block is not None
        mentioned = set(re.findall(r"`(\w+)\(", block.group(0)))
        assert mentioned == set(tools), (mentioned, set(tools))
        assert set(tools["read_section"].input_schema["properties"]) == {
            "file_path",
            "heading_path",
            "include_subsections",
        }
        assert set(tools["search_docs"].input_schema["properties"]) == {"query", "limit"}


class TestDocsMatchTheCode:
    def test_documented_gates_are_the_gates_the_eval_enforces(self) -> None:
        evaluation = load_script("eval_retrieval")
        floors = (
            f">= {evaluation.FLOOR_PARAPHRASE_TOP1:.0%}",
            f">= {evaluation.FLOOR_PARAPHRASE_TOP5:.0%}",
            f"{evaluation.FLOOR_IDENTIFIER_TOP1:.0%}",
        )
        assert floors == (">= 80%", ">= 90%", "100%")
        for name in ("CLAUDE.md", ".claude/skills/run-eval/SKILL.md"):
            text = all_agent_text()[name]
            for floor in floors:
                assert floor in text, f"{name} does not state the {floor} gate"

    def test_documented_query_counts_are_the_real_ones(self) -> None:
        queries = json.loads((ROOT / "scripts/eval_data/queries.json").read_text())
        counts = {f"{s}/{k}": len(v) for s in ("dev", "held_out") for k, v in queries[s].items()}
        assert counts == {
            "dev/paraphrase": 34, "dev/identifier": 10,
            "held_out/paraphrase": 34, "held_out/identifier": 8,
        }  # fmt: skip
        skill = all_agent_text()[".claude/skills/run-eval/SKILL.md"]
        assert f"{counts['dev/paraphrase']} paraphrase queries in each split" in skill
        assert f"{counts['dev/identifier']} dev and" in skill
        assert f"{counts['held_out/identifier']} held-out identifier queries" in skill
        for size, weight in (
            (counts["dev/identifier"], "10pp"),
            (counts["held_out/identifier"], "12.5pp"),
        ):
            assert f"{100 / size:g}".rstrip("0") in weight and weight in skill

    def test_documented_baseline_is_the_frozen_baseline(self) -> None:
        baseline = json.loads((ROOT / "scripts/eval_data/baseline.json").read_text())
        assert set(baseline) == {"embeddinggemma", "bge-small"}
        held_out = baseline["embeddinggemma"]["held_out/paraphrase"]
        quoted = (f"{held_out['top1']:.0%}", f"{held_out['top5']:.0%}")
        for name in ("CLAUDE.md", ".claude/skills/run-eval/SKILL.md"):
            text = all_agent_text()[name]
            assert f"| {quoted[0]} |" in text and f"| {quoted[1]} |" in text, (name, quoted)

        # Two gate tables are not all of it. The number is printed in five first-party
        # places, and checking only the two that happen to use a table cell is how it went
        # three merges out of date: the README preset row, the navigation block that is
        # byte-identical across the three agent files, and the floors comment beside the
        # thresholds themselves all quote it too.
        readme = (ROOT / "README.md").read_text(encoding="utf-8")
        assert f"| {quoted[0]} / {quoted[1]} /" in readme, ("README.md preset row", quoted)
        block = NAVIGATION_BLOCK.search(all_agent_text()["CLAUDE.md"])
        assert block is not None
        assert f"Top-1 ~{quoted[0]}" in block.group(0), ("navigation block", quoted)
        assert f"~{quoted[1]} reliable" in block.group(0), ("navigation block", quoted)
        floors = (ROOT / "scripts/eval_retrieval.py").read_text(encoding="utf-8")
        assert f"measured: {quoted[0]} / {quoted[1]} /" in floors, ("eval_retrieval.py", quoted)

        # The README prints a full row per preset, not just the default's two headline
        # figures, and the light preset drifts by the same mechanism as the default did.
        # Read each row by its preset name: looking the triple up anywhere in the file
        # passes just as happily when the two rows have been swapped, which is a table that
        # recommends the wrong model.
        for preset in ("embeddinggemma", "bge-small"):
            row = baseline[preset]["held_out/paraphrase"]
            printed = " / ".join(f"{row[key]:.0%}" for key in ("top1", "top3", "top5"))
            line = re.search(rf"^\| `{re.escape(preset)}`.*$", readme, re.M)
            assert line, f"README.md has no preset row for {preset}"
            assert f"| {printed} |" in line.group(0), (
                f"README.md's `{preset}` row reads {line.group(0)!r}, "
                f"but the baseline says {printed}"
            )
        for preset in baseline.values():
            assert set(preset) == {
                "dev/paraphrase",
                "dev/identifier",
                "held_out/paraphrase",
                "held_out/identifier",
            }
            assert preset["dev/identifier"]["top1"] == preset["held_out/identifier"]["top1"] == 1

    def test_every_referenced_repository_path_exists(self) -> None:
        pattern = re.compile(r"(?<![\w/.-])((?:src|scripts|tests|\.claude)/[\w./-]+\w)")
        for name, text in all_agent_text().items():
            for path in set(pattern.findall(text)):
                assert (ROOT / path).exists(), f"{name} references missing path {path}"

    def test_every_documented_flag_is_accepted_by_its_script(self) -> None:
        for script, flags in (
            ("scripts/eval_retrieval.py", ("--show-misses", "--update-baseline", "--embedder")),
            ("scripts/reindex_docs.py", ("--force", "--db", "--embedder")),
            ("src/markdown_memory/server.py", ("--db", "--docs-dir", "--log-level", "--embedder")),
        ):
            usage = subprocess.run(
                [sys.executable, str(ROOT / script), "--help"],
                capture_output=True, text=True, check=True,
            ).stdout  # fmt: skip
            for flag in flags:
                assert flag in usage, f"{script} does not accept {flag}"

    def test_no_placeholders(self) -> None:
        for name, text in all_agent_text().items():
            for marker in ("TODO", "TBD", "FIXME", "XXX", "<placeholder", "lorem ipsum"):
                assert marker.lower() not in text.lower(), f"{name} contains {marker!r}"

    def test_check_script_runs_the_documented_steps_in_order(self) -> None:
        script = (ROOT / "scripts/check.sh").read_text(encoding="utf-8")
        positions = [script.index(f"step {step}") for step in GATE_STEPS]
        assert positions == sorted(positions)
        assert "set -euo pipefail" in script
        assert (ROOT / "scripts/check.sh").stat().st_mode & 0o111, "check.sh is not executable"

    def test_ci_runs_the_same_steps_in_the_same_order(self) -> None:
        """Two gates that disagree are worse than one: the local hook is authoritative.

        The workflow lists the steps one by one so each is annotated in the Actions log,
        which is exactly the shape that drifts - a step added to `check.sh` and forgotten
        in CI passes on the laptop and nowhere else, or the reverse.
        """
        workflow = (ROOT / ".github/workflows/gate.yml").read_text(encoding="utf-8")
        positions = [workflow.index(f"- run: {step}") for step in GATE_STEPS]
        assert positions == sorted(positions), "CI runs the gate's steps out of order"

    def test_ci_loads_the_model_before_the_tests_that_would_skip_without_it(self) -> None:
        """A cache miss must fail the job, not quietly skip eighteen behaviours.

        The `embedding` fixture turns a model that will not load into `pytest.skip`, so
        without this step a broken cache leaves CI green over everything the real model
        covers - including the whole of `live_test.py`'s reason to exist.
        """
        workflow = (ROOT / ".github/workflows/gate.yml").read_text(encoding="utf-8")
        assert "warm_up()" in workflow
        assert workflow.index("warm_up()") < workflow.index("- run: uv run pytest -q")

    def test_ci_caches_the_model_revision_and_graph_the_code_pins(self) -> None:
        """The cache key is the pin, so moving the pin cannot serve the old weights.

        The graph belongs in it as well as the revision: this repository publishes several
        graphs at one revision, and an Actions cache entry is immutable once written. Keyed
        on the revision alone, the entry saved for one graph would be restored for ever,
        fail the stamp, and re-download the model on every job.
        """
        workflow = (ROOT / ".github/workflows/gate.yml").read_text(encoding="utf-8")
        key = next(
            line.split("key:", 1)[1].strip()
            for line in workflow.splitlines()
            if line.strip().startswith("key: mdmem-model-")
        )
        assert GEMMA_REVISION[:12] in key, key
        assert Path(GEMMA_MODEL_FILE).stem in key, key

    def test_ci_tests_every_python_version_the_metadata_claims(self) -> None:
        """`requires-python` and the classifiers are promises; this is what keeps them."""
        metadata = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
        claimed = {
            line.rsplit(" :: ", 1)[1]
            for line in metadata["project"]["classifiers"]
            if line.startswith("Programming Language :: Python :: 3.")
        }
        workflow = (ROOT / ".github/workflows/gate.yml").read_text(encoding="utf-8")
        matrix = re.search(r"python: \[(.+?)\]", workflow)
        assert matrix is not None, "the workflow has no python matrix"
        tested = set(re.findall(r'"([0-9.]+)"', matrix.group(1)))
        assert claimed == tested, f"classifiers claim {sorted(claimed)}, CI runs {sorted(tested)}"
        # The architecture legs arrive through `include`, which pins its own interpreter
        # outside that list. A version dropped from the classifiers while an include entry
        # still names it would leave one leg testing something nothing claims.
        pinned = set(re.findall(r'"python":"([0-9.]+)"', workflow))
        assert pinned <= claimed, f"the workflow pins {sorted(pinned - claimed)}, unclaimed"

    def test_ci_fails_when_the_golden_vector_skips(self) -> None:
        """The skip is the quiet failure mode, so something has to be loud about it.

        On a CPU that takes onnxruntime's other `MatMulNBits` kernel the golden vector
        skips rather than fails, because that machine is computing correctly. No runner in
        the gate is such a machine - so a skip there means the numbers moved, and deleting
        the step that notices would restore exactly the silence it was added to remove.
        """
        workflow = (ROOT / ".github/workflows/gate.yml").read_text(encoding="utf-8")
        nodeid = (
            "tests/test_embedders.py::test_the_real_model_returns_the_vector_it_was_baselined_on"
        )
        # Read out of `env`, not merely found somewhere in the file. Pointing that entry at
        # another passing test, while leaving this nodeid in a comment, was a false green
        # until the value itself was compared.
        declared = re.search(r"GOLDEN_VECTOR_TEST: >-\n\s+(\S+)", workflow)
        assert declared is not None, "the workflow no longer names the golden test in `env`"
        assert declared.group(1) == nodeid, (
            f"GOLDEN_VECTOR_TEST is {declared.group(1)!r}, expected {nodeid!r}"
        )
        # And the step has to run *that* name, rather than one written out beside it.
        assert 'pytest "${{ env.GOLDEN_VECTOR_TEST }}"' in workflow, (
            "the golden-vector step does not run the test `env` declares"
        )
        assert "--junitxml=golden.xml" in workflow, (
            "the golden vector's result is read from prose again, not from its JUnit record"
        )
        assert "'skipped': 0" in workflow, "the workflow no longer fails when it skips"
        # Collection matching nothing must not read as success either.
        assert "'tests': 1" in workflow, "the workflow no longer requires the test to have run"

    def test_ci_runs_on_exactly_the_runners_this_table_records(self) -> None:
        """A runner nobody wrote down is a runner nothing else in this file can check.

        Three places name one, and all three are read: the matrix dimension, the `include`
        entries that add an architecture, and a plain `runs-on` on a job with no matrix.
        Reading only some of them was this test's first bug - the base dimension was
        missed, and the `package` job's identical label stood in for it, so swapping the
        whole gate onto another operating system would have gone through unnoticed.
        """
        workflow = _gate_job()
        dimension = set(re.findall(r"^\s+runner: \[(.+?)\]$", workflow, re.M))
        assert dimension, "the gate has no `runner` matrix dimension"
        base = {label.strip().strip('"') for line in dimension for label in line.split(",")}
        included = set(re.findall(r'"runner":"([^"]+)"', workflow))
        assert included, "no architecture legs arrive through `include`"
        plain = {
            label
            for label in re.findall(r"runs-on: (\S+)", workflow)
            if not label.startswith("${{")
        }
        used = base | included | plain
        assert used == set(GATE_RUNNERS), (
            f"the workflow runs on {sorted(used)}, the table records {sorted(GATE_RUNNERS)}"
        )
        # Each source has to contribute its own share, or the union hides a swap. Moving
        # the base dimension to the arm runner leaves the union identical - the `package`
        # job still spells `ubuntu-latest` - while quietly moving all four interpreters
        # off x86. The Python matrix belongs on the architecture everything is measured on.
        assert base == {
            name for name, (_, machine) in GATE_RUNNERS.items() if machine == "x86_64"
        }, f"the Python matrix runs on {sorted(base)}, which is not the x86_64 runner"
        legs = set(GATE_RUNNERS) - base
        assert included == legs, (
            f"the architecture legs are {sorted(included)}, expected {sorted(legs)}"
        )
        # Naming the runners is half of it: the gate has to be *wired* to the dimension.
        # Pinning its `runs-on` back to a literal leaves every label above still present
        # and every architecture leg silently running on the same machine as the rest.
        assert "runs-on: ${{ matrix.runner }}" in workflow, (
            "the gate job no longer takes its runner from the matrix"
        )

    def test_ci_runs_every_operating_system_the_metadata_claims(self) -> None:
        """The twin of the Python test, for the promise no runner used to keep.

        `Operating System :: MacOS` sat in the classifiers while CI was Linux-only: a
        platform claimed and never once exercised, which is the shape of a broken install
        found by its first user.
        """
        metadata = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
        claimed = {
            line.split(" :: ", 1)[1]
            for line in metadata["project"]["classifiers"]
            if line.startswith("Operating System :: ")
        }
        tested = {system for system, _ in GATE_RUNNERS.values()}
        assert claimed == tested, f"classifiers claim {sorted(claimed)}, CI runs {sorted(tested)}"

    def test_the_readme_names_every_architecture_ci_covers(self) -> None:
        """Architecture has no trove classifier, so the prose is the only thing that claims it.

        It is worth claiming: the 4-bit graph picks its `MatMulNBits` kernel from what the
        CPU offers, so "runs on Linux" is a weaker statement than it looks.
        """
        readme = (ROOT / "README.md").read_text(encoding="utf-8")
        for architecture in {machine for _, machine in GATE_RUNNERS.values()}:
            spelled = {"x86_64": "x86-64", "aarch64": "arm64", "arm64": "arm64"}[architecture]
            assert spelled in readme, f"the README does not mention {spelled}"

    def test_the_job_asserts_the_same_architectures_this_table_records(self) -> None:
        """The workflow carries the table too, because only the job can see the machine.

        Both halves are checked. A mapping the job merely prints proves nothing, so the
        comparison that turns it into a failure is pinned as well - deleting that one line
        would otherwise leave a green job happily reporting the wrong architecture.
        """
        workflow = (ROOT / ".github/workflows/gate.yml").read_text(encoding="utf-8")
        asserted = dict(re.findall(r"'([a-z0-9.-]+)': '(x86_64|aarch64|arm64)',", workflow))
        expected = {runner: machine for runner, (_, machine) in GATE_RUNNERS.items()}
        assert asserted == expected, f"the job checks {asserted}, the table says {expected}"
        assert "assert actual == expected" in workflow, (
            "the job reads platform.machine() but no longer fails when it disagrees"
        )
        # And that what it compares is the machine, not a constant that always agrees.
        assert "actual = platform.machine()" in workflow, (
            "the job asserts on something other than platform.machine()"
        )


class TestSkills:
    @pytest.mark.parametrize("skill", SKILLS)
    def test_front_matter(self, skill: str) -> None:
        text = (ROOT / ".claude" / "skills" / skill / "SKILL.md").read_text(encoding="utf-8")
        match = re.match(r"---\n(.*?)\n---\n", text, re.DOTALL)
        assert match is not None, "SKILL.md must start with YAML front matter"
        fields = dict(line.split(": ", 1) for line in match.group(1).splitlines())
        assert fields["name"] == skill
        assert len(fields["description"]) > 80  # says what it does AND when to use it
        assert "Use " in fields["description"]

    def test_claude_md_lists_every_skill(self) -> None:
        text = (ROOT / "CLAUDE.md").read_text(encoding="utf-8")
        for skill in SKILLS:
            assert f"**`{skill}`**" in text


# ---------------------------------------------------------------------- what the skills run


def _draft(title: str, body: str) -> SectionDraft:
    return SectionDraft(
        heading_title=title, heading_level=2, heading_path=f"Doc > {title}",
        base_path=f"Doc > {title}", content=f"## {title}\n\n{body}", start_line=1, end_line=3,
        units=(body,),
    )  # fmt: skip


class TestIntegrityProblems:
    def test_sound_database_reports_nothing(
        self, db: Database, fake_embedder: FakeEmbedder
    ) -> None:
        assert db.integrity_problems() == []
        sections = [_draft("A", "alpha text"), _draft("B", "beta text")]
        db.replace_document(
            file_path="/d/a.md", title="Doc", content_hash="h", last_modified=1, mtime_ns=1,
            sections=sections, vectors=vectors_for(fake_embedder, sections),
        )  # fmt: skip
        assert db.integrity_problems() == []

    def test_each_kind_of_damage_is_named(self, db: Database, fake_embedder: FakeEmbedder) -> None:
        sections = [_draft("A", "alpha text"), _draft("B", "beta text")]
        db.replace_document(
            file_path="/d/a.md", title="Doc", content_hash="h", last_modified=1, mtime_ns=1,
            sections=sections, vectors=vectors_for(fake_embedder, sections),
        )  # fmt: skip
        conn = db.connection()
        conn.execute("DELETE FROM units_vec WHERE unit_id = (SELECT MIN(id) FROM units)")
        conn.execute("DELETE FROM sections_vec WHERE section_id = (SELECT MAX(id) FROM sections)")
        conn.execute("INSERT INTO sections_fts(sections_fts) VALUES ('delete-all')")
        conn.execute("UPDATE meta SET value = '999' WHERE key = 'embedding_dim'")
        problems = "\n".join(db.integrity_problems())
        assert "2 passages but 1 passage vectors" in problems
        assert "1 section vectors but 2 sections with passages" in problems
        assert "2 sections but 0 FTS rows" in problems
        assert "FTS5 index does not match the sections table" in problems
        assert "could not verify" not in problems
        assert "meta embedding_dim is 999, expected 384" in problems

    def test_outdated_schema_version_is_reported(self, db: Database) -> None:
        db.connection().execute("PRAGMA user_version = 1")
        assert db.integrity_problems() == ["schema version is 1, expected 7"]


class TestReindexScript:
    def run(self, *arguments: str, db: Path) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            [sys.executable, str(ROOT / "scripts/reindex_docs.py"), *arguments,
             "--db", str(db), "--embedder", "embeddinggemma"],
            capture_output=True, text=True, timeout=900,
        )  # fmt: skip

    @pytest.mark.embedding
    def test_forced_reindex_re_embeds_unchanged_files_and_verifies(
        self, tmp_path: Path, real_embedder: Embedder
    ) -> None:
        # real_embedder skips this test when the model cannot be loaded (offline, no cache);
        # the subprocess below loads the same, already cached, default model.
        assert real_embedder.dimension == 768
        docs = tmp_path / "docs"
        docs.mkdir()
        (docs / "a.md").write_text("# A\n\nalpha body\n\n## Child\n\n- one\n- two\n")
        database = tmp_path / "index.db"
        first = self.run(str(docs), db=database)
        assert first.returncode == 0, first.stdout + first.stderr
        assert "1 (re)indexed, 0 unchanged" in first.stdout
        incremental = self.run(str(docs), db=database)
        assert "0 (re)indexed, 1 unchanged" in incremental.stdout
        forced = self.run(str(docs), "--force", db=database)
        assert forced.returncode == 0, forced.stdout + forced.stderr
        assert "--force  : dropped 1 indexed document(s)" in forced.stdout
        assert "1 (re)indexed, 0 unchanged" in forced.stdout
        assert "768 dimensions (meta: 768" in forced.stdout
        assert "integrity: ok" in forced.stdout
        assert "INTEGRITY PROBLEM" not in forced.stdout


class TestTheDiagramStillMeasuresTheFilesItClaimsTo:
    """The README's picture prints token counts for four real files.

    They were measured, not invented - which means editing any of those files makes the
    picture false, silently, because nothing reads an SVG. The sweep that produced this
    branch did exactly that: it rewrote README.md after the diagram had been drawn, and the
    printed total was 319 tokens short of the truth until this test existed.
    """

    def _figures(self) -> list[tuple[str, int]]:
        """The (file, tokens) pairs the generator draws on the left-hand side.

        Imported rather than read off disk, so that mutating the generator mutates what
        this test measures - a test that re-read the checked-in file would score a
        falsified figure green.
        """
        import make_diagram

        figures = list(make_diagram.LEFT_FILES)
        assert len(figures) == 4, figures
        return figures

    def test_every_file_on_the_diagram_still_costs_what_it_says(self) -> None:
        from markdown_memory.models import estimate_tokens

        # The generator names them by basename; evaluation-protocol.md lives under docs/.
        roots = {"evaluation-protocol.md": ROOT / "docs"}
        for name, printed in self._figures():
            path = roots.get(name, ROOT) / name
            actual = estimate_tokens(path.read_text(encoding="utf-8"))
            assert actual == printed, (
                f"{name} is {actual} tokens, the diagram says {printed}. "
                f"Re-measure and re-run scripts/make_diagram.py."
            )

    def test_the_totals_the_readme_prints_are_the_sum_of_those_files(self) -> None:
        import make_diagram

        total = sum(tokens for _, tokens in self._figures())
        readme = (ROOT / "README.md").read_text(encoding="utf-8")
        assert f"{total:,}" in readme, f"README does not print the {total:,}-token total"
        for svg in ("how-it-works-light.svg", "how-it-works-dark.svg"):
            rendered = (ROOT / "docs/assets" / svg).read_text(encoding="utf-8")
            assert f"{total:,} tokens" in rendered, f"{svg} prints a stale total"

        # The <img> alt text repeats what comes back as well as what went in, and only the
        # left-hand total was pinned - so a re-measurement of RIGHT_HITS could redraw the
        # picture correctly and leave the sentence beside it describing the old one.
        assert total == make_diagram.TOTAL_TOKENS
        for figure in (f"{make_diagram.RETURNED_TOKENS:,}", str(make_diagram.BEST_HIT)):
            assert figure in readme, (
                f"README does not print {figure!r}, which the drawing beside it does"
            )

    def test_the_committed_drawing_is_the_one_the_generator_draws(self) -> None:
        """Byte for byte, so no edit to the picture can skip being redrawn.

        The figure checks below say what the drawing must contain; this says it contains
        nothing else either. It is what catches an element deleted from the generator whose
        text happens to be repeated somewhere - removing the 181 beside its bar still left
        a 181 in the caption underneath, and a search for the figure passed.
        """
        import make_diagram

        for theme, colours in make_diagram.THEMES.items():
            path = ROOT / "docs/assets" / f"how-it-works-{theme}.svg"
            assert path.read_text(encoding="utf-8") == make_diagram.draw(colours), (
                f"{path.name} is not what scripts/make_diagram.py draws today; re-run it"
            )

    def test_the_committed_drawing_prints_every_figure_the_generator_holds(self) -> None:
        """A correct total is not a correct picture.

        The committed SVGs once carried a per-file number 348 tokens below the bar beside
        it, with the right total printed underneath - the mutation sweep had written a
        falsified figure straight into `docs/assets/`, because the generator drew at import
        time. Checking the total alone let that through.
        """
        import make_diagram

        expected = [f"{tokens:,}" for _, tokens in make_diagram.LEFT_FILES]
        expected += [str(tokens) for tokens, _, _ in make_diagram.RIGHT_HITS]
        expected.append(f"answers is {make_diagram.BEST_HIT}")
        expected.append(f"{make_diagram.TOTAL_TOKENS:,} tokens")
        expected.append(f"{make_diagram.RETURNED_TOKENS:,} tokens")
        for svg in ("how-it-works-light.svg", "how-it-works-dark.svg"):
            rendered = (ROOT / "docs/assets" / svg).read_text(encoding="utf-8")
            # Only what the <text> elements draw. Searching the whole file would score the
            # aria-label, which carries the same sentence - so deleting the visible caption
            # and leaving the alt text behind would have passed.
            drawn = "\n".join(re.findall(r"<text[^>]*>(.*?)</text>", rendered, re.DOTALL))
            assert drawn, f"{svg} draws no text at all"
            for figure in expected:
                assert figure in drawn, (
                    f"{svg} does not draw {figure!r}; re-run scripts/make_diagram.py"
                )

    def test_the_drawing_says_the_same_thing_to_a_reader_who_cannot_see_it(self) -> None:
        """The aria-label is the picture, for anyone not looking at it.

        It used to spell its three figures out by hand while the caption beside them was
        derived, so a re-measurement moved the caption and left the alt text describing the
        previous one.
        """
        import make_diagram

        for svg in ("how-it-works-light.svg", "how-it-works-dark.svg"):
            rendered = (ROOT / "docs/assets" / svg).read_text(encoding="utf-8")
            label = re.search(r'aria-label="([^"]*)"', rendered)
            assert label, f"{svg} has no aria-label"
            for figure in (
                f"{make_diagram.TOTAL_TOKENS:,}",
                f"{make_diagram.RETURNED_TOKENS:,}",
                str(make_diagram.BEST_HIT),
            ):
                assert figure in label.group(1), (
                    f"{svg}'s aria-label does not carry {figure!r}: it describes a "
                    f"different picture than the one it labels"
                )

    def test_the_readme_falls_back_to_a_raster_every_client_can_draw(self) -> None:
        """Browsers get the vector; anything that ignores <picture> gets a raster.

        Every address is absolute, into this public repository's `main`. The same README is
        the PyPI project page, and PyPI resolves no relative path and drops <source>
        altogether - only the <img> survives its sanitiser, so that one above all must be a
        URL that works from anywhere. The raster stays for clients that do not implement
        <picture>, the PyPI page among them.
        """
        import make_diagram

        readme = (ROOT / "README.md").read_text(encoding="utf-8")
        picture = re.search(r"<picture>(.*?)</picture>", readme, re.DOTALL)
        assert picture, "the README no longer shows the diagram in a <picture>"
        block = picture.group(1)
        img = re.search(r'<img src="([^"]+)"', block)
        assert img and img.group(1).endswith(".png"), (
            "the <img> fallback must be a raster: it is what a client that ignores "
            "<picture> falls back to"
        )
        # Follow the path the README actually gives, rather than checking a name this test
        # chose: a fallback that ends in .png and points at nothing renders as the same
        # broken-image mark it exists to prevent.
        assert img.group(1).startswith(RAW_MAIN), (
            f"the fallback {img.group(1)} is not absolute: PyPI renders it as a broken image"
        )
        fallback = ROOT / img.group(1).removeprefix(RAW_MAIN)
        assert fallback.is_file(), f"the README's fallback {img.group(1)} does not exist"
        for theme in ("dark", "light"):
            source = f"docs/assets/how-it-works-{theme}.svg"
            assert f'srcset="{RAW_MAIN}{source}"' in block, theme
            assert (ROOT / source).is_file(), f"{source} is offered but not committed"
        assert "prefers-color-scheme: dark" in block, "nothing selects the dark drawing"
        assert fallback == ROOT / "docs/assets/how-it-works-light.png", (
            f"the fallback is {img.group(1)}; the checks below measure the light raster"
        )

        # A fallback nothing regenerates is a fallback that goes stale, so hold its size to
        # the drawing's own, at the scale the generator rasterises. Read straight out of the
        # PNG header rather than through an imaging library: Pillow is here only as
        # somebody else's transitive dependency, and a test should not rest on that.
        expected = (
            make_diagram.W * make_diagram.PNG_SCALE,
            make_diagram.H * make_diagram.PNG_SCALE,
        )
        blobs: dict[str, bytes] = {}
        for theme in ("light", "dark"):
            png = ROOT / "docs/assets" / f"how-it-works-{theme}.png"
            assert png.exists(), f"{png.name} is missing; run scripts/make_diagram.py"
            blobs[theme] = png.read_bytes()
            header = blobs[theme][:24]
            assert header[:8] == b"\x89PNG\r\n\x1a\n", f"{png.name} is not a PNG"
            assert header[12:16] == b"IHDR", f"{png.name} has no image header"
            size = (int.from_bytes(header[16:20], "big"), int.from_bytes(header[20:24], "big"))
            assert size == expected, (png.name, size, expected)
            # A blank canvas of the right dimensions would satisfy everything above. The
            # drawing is several hundred glyphs on a flat background, which does not
            # compress anywhere near this small; an empty one lands in the low tens of KB.
            assert len(blobs[theme]) > 60_000, (
                f"{png.name} is {len(blobs[theme])} bytes - too little to be the drawing"
            )
        assert blobs["light"] != blobs["dark"], (
            "the two rasters are byte-identical, so at least one was not drawn from its own theme"
        )

    def test_a_half_redrawn_diagram_is_a_failure_and_not_a_warning(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The generator rewrites the SVG before it rasterises the PNG.

        So a machine with no headless browser regenerates the vector, leaves the raster at
        the previous figures, and used to exit 0 over two images that now disagree. Run it
        against a scratch directory rather than the repository: the point is the exit code,
        and a test that redrew `docs/assets` would be a test that edits tracked files.
        """
        import make_diagram

        (tmp_path / "docs/assets").mkdir(parents=True)
        monkeypatch.setattr(make_diagram, "ROOT", tmp_path)
        monkeypatch.setattr(make_diagram, "rasterise", lambda svg, png: False)

        # `!= 0` would be satisfied by None, which SystemExit reads as success.
        assert make_diagram.main() == 1, (
            "main() reported success while the PNGs the README falls back to went stale"
        )
        drawn = sorted(p.name for p in (tmp_path / "docs/assets").iterdir())
        assert drawn == ["how-it-works-dark.svg", "how-it-works-light.svg"], drawn

    def test_the_sections_the_worked_example_returns_are_the_size_it_claims(self) -> None:
        """Ranking needs the model; a section's token estimate does not, so pin that."""
        from markdown_memory.models import estimate_tokens
        from markdown_memory.parser import MarkdownParser

        readme = (ROOT / "README.md").read_text(encoding="utf-8")
        example = re.search(r"```\nsearch_docs\(.*?\n\n(.*?)```", readme, re.DOTALL)
        assert example, "the README no longer shows a worked example this test can read"
        listed = re.findall(r"\s*(\d+) tok\s+(\S+)\s+(.+?)\s*$", example.group(1), re.M)
        assert len(listed) == 5, listed

        parser = MarkdownParser()
        cache: dict[str, dict[str, int]] = {}
        for printed, filename, heading_path in listed:
            if filename not in cache:
                parsed = parser.parse((ROOT / filename).read_text(encoding="utf-8"))
                cache[filename] = {
                    section.heading_path: estimate_tokens(section.content)
                    for section in parsed.sections
                }
            sizes = cache[filename]
            assert heading_path in sizes, (
                f"{filename} has no section '{heading_path}'; the example is stale"
            )
            assert sizes[heading_path] == int(printed), (
                f"{filename} '{heading_path}' is {sizes[heading_path]} tokens, "
                f"the README says {printed}"
            )

    def test_the_worked_example_adds_up_to_the_total_it_prints(self) -> None:
        readme = (ROOT / "README.md").read_text(encoding="utf-8")
        example = re.search(r"```\nsearch_docs\(.*?\n\n(.*?)```", readme, re.DOTALL)
        assert example
        returned = sum(int(n) for n in re.findall(r"(\d+) tok", example.group(1)))
        assert f"**{returned:,} tokens instead of" in readme, (
            f"the five hits total {returned:,}, which is not what the README claims"
        )
