"""What the package is allowed to look like, checked rather than described.

`indexer.py` reached 1575 lines holding four unrelated jobs, and `server.py` 901 holding
seven, because nothing failed while they grew. These tests are what failing looks like:
a module over its budget, a heavy dependency somewhere it does not belong, a module the
layout table never mentions, or a symbol nothing references.
"""

from __future__ import annotations

import ast
import io
import symtable
import tokenize
from pathlib import Path

ROOT = Path(__file__).parent.parent
PACKAGE = "markdown_memory"
PACKAGE_DIR = ROOT / "src" / PACKAGE

#: Logical lines of code per module - see `logical_lines`. A ceiling is a **growth alarm,
#: not proof of cohesion**: a module can mix concerns at any size. Each is 1.2x what the
#: module measured when it was written, so ordinary fixes fit and real accretion does not.
#: Raising one is a deliberate line in a diff, which is the whole point.
#:
#: `db.py` is the known outlier: 1004 logical lines in one `Database` class. It is
#: grandfathered rather than quietly given a number that makes this test vacuous. If it is
#: ever split, the seam is a search repository - the FTS and vector primitives that
#: `HybridSearcher` already calls - not the DDL helpers.
BUDGETS = {
    "__init__.py": 44,
    "autoindex.py": 130,
    "config.py": 138,
    "db.py": 1205,  # the outlier, grandfathered
    "discovery.py": 210,
    "embedders.py": 366,
    "exceptions.py": 16,
    "freshness.py": 69,
    "headings.py": 155,
    "indexer.py": 495,
    "model_cache.py": 173,
    "models.py": 254,
    "parser.py": 722,
    "search.py": 399,
    "server.py": 355,
}

#: Where a heavy dependency may be imported **at module scope**, which is what puts it in
#: another module's import graph. A lazy import inside a function is judged separately.
CONFINED = {
    "numpy": {"embedders.py"},
    "onnxruntime": {"embedders.py"},
    "fastembed": {"embedders.py"},
    "huggingface_hub": {"embedders.py"},
    "sqlite3": {"db.py"},
    "markdown_it": {"parser.py"},
    # The cache lock and the scan lock are different locks over different files.
    "fcntl": {"model_cache.py", "indexer.py"},
}

#: Lazy, function-local imports of a confined dependency, each with its reason. Not a
#: general escape hatch: one entry, and a wildcard is never allowed.
LAZY_IMPORTS = {
    ("model_cache.py", "fastembed"): (
        "reads the repository name out of fastembed's registry to locate the cache "
        "directory; never constructs an embedder"
    ),
}

#: Top-level names nothing references, each with the reason it survives anyway. Empty, and
#: meant to stay that way: an entry here is a claim that a dynamic use exists.
ALLOWED_UNREFERENCED: dict[str, str] = {}


def logical_lines(source: str) -> int:
    """Lines that carry code: blanks, comment lines and docstring lines do not count.

    Checked in beside the budgets because a ceiling nobody can reproduce is not a gate -
    two reasonable implementations of "logical lines" disagreed by about 1% on this
    package, which is the difference between passing and failing near a limit.
    """
    skip: set[int] = set()
    for node in ast.walk(ast.parse(source)):
        # The *first* statement of a module, function or class, and only that: a bare
        # string anywhere else is a no-op the author still wrote, and excusing it would
        # let a module grow by triple-quoting whatever it likes.
        if not isinstance(node, ast.Module | ast.FunctionDef | ast.AsyncFunctionDef | ast.ClassDef):
            continue
        first = node.body[0] if node.body else None
        if (
            isinstance(first, ast.Expr)
            and isinstance(first.value, ast.Constant)
            and isinstance(first.value.value, str)
        ):
            skip.update(range(first.lineno, (first.end_lineno or first.lineno) + 1))
    for token in tokenize.generate_tokens(io.StringIO(source).readline):
        # Only a line that is *nothing but* a comment: `value = 1  # why` is a code line,
        # and skipping it would let a module grow without the budget noticing.
        if (
            token.type == tokenize.COMMENT
            and not source.splitlines()[token.start[0] - 1][: token.start[1]].strip()
        ):
            skip.add(token.start[0])
    return sum(
        1
        for number, line in enumerate(source.splitlines(), 1)
        if line.strip() and number not in skip
    )


def package_modules() -> list[Path]:
    return sorted(PACKAGE_DIR.glob("*.py"))


def test_module_logical_lines_stay_within_budget() -> None:
    over: list[str] = []
    for path in package_modules():
        count = logical_lines(path.read_text(encoding="utf-8"))
        budget = BUDGETS.get(path.name)
        if budget is None:
            over.append(f"{path.name} declares no budget ({count} logical lines)")
        elif count > budget:
            over.append(f"{path.name} is {count} logical lines, over its budget of {budget}")
    assert over == [], "; ".join(over)


def test_no_budget_names_a_module_that_is_gone() -> None:
    present = {path.name for path in package_modules()}
    phantom = sorted(set(BUDGETS) - present)
    assert phantom == [], f"budgets name modules that do not exist: {phantom}"


def test_heavy_dependencies_stay_where_they_belong() -> None:
    """`numpy` in the parser, or `sqlite3` in the server, is how the junk drawers started."""
    trespass: list[str] = []
    for path in package_modules():
        tree = ast.parse(path.read_text(encoding="utf-8"))
        # An import is lazy only when it sits inside a function: a module-level `try`
        # around it still runs at import time, and would otherwise slip through as lazy.
        lazy = {
            child
            for scope in ast.walk(tree)
            if isinstance(scope, ast.FunctionDef | ast.AsyncFunctionDef)
            for child in ast.walk(scope)
            if isinstance(child, ast.Import | ast.ImportFrom)
        }
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                names = [alias.name for alias in node.names]
            elif isinstance(node, ast.ImportFrom):
                names = [node.module or ""]
            else:
                continue
            for name in names:
                root = name.split(".")[0]
                allowed = CONFINED.get(root)
                if allowed is None or path.name in allowed:
                    continue
                if node not in lazy:
                    trespass.append(f"{path.name}:{node.lineno} imports {root}")
                elif (path.name, root) not in LAZY_IMPORTS:
                    trespass.append(
                        f"{path.name}:{node.lineno} lazily imports {root} with no recorded reason"
                    )
    assert trespass == [], "; ".join(trespass)


def test_every_module_is_named_in_the_layout_table() -> None:
    """The table in CLAUDE.md is how an agent finds its way; a missing row is a dead end."""
    table = (ROOT / "CLAUDE.md").read_text(encoding="utf-8")
    listed = {
        line.split("`")[1].rsplit("/", 1)[1]
        for line in table.splitlines()
        if line.startswith("| `src/markdown_memory/") and "`" in line
    }
    present = {path.name for path in package_modules()} - {"__init__.py"}
    missing = sorted(present - listed)
    phantom = sorted(listed - present)
    assert missing == [], f"modules the layout table never mentions: {missing}"
    assert phantom == [], f"the layout table names modules that do not exist: {phantom}"


def _names_in_expression(text: str) -> set[str]:
    """The names in a snippet of source, or nothing if it does not parse as one."""
    try:
        tree = ast.parse(text, mode="eval")
    except SyntaxError:
        return set()
    return {node.id for node in ast.walk(tree) if isinstance(node, ast.Name)}


def _annotated_names(source: str) -> set[str]:
    """Names mentioned in an annotation.

    `from __future__ import annotations` is on across this package, so annotations are
    never evaluated and CPython's symbol table does not record them as reads at all. A
    Protocol or dataclass used only as a type would look dead. Collected separately, and
    deliberately without scoping: a name in an annotation can only keep a symbol alive,
    and being too generous here costs a missed dead symbol, never a failed build on
    working code.
    """
    named: set[str] = set()
    for node in ast.walk(ast.parse(source)):
        annotations = []
        if isinstance(node, ast.arg | ast.AnnAssign):
            annotations = [node.annotation]
        elif isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef):
            annotations = [node.returns]
        for annotation in annotations:
            if annotation is None:
                continue
            for child in ast.walk(annotation):
                if isinstance(child, ast.Name):
                    named.add(child.id)
                elif isinstance(child, ast.Constant) and isinstance(child.value, str):
                    # A forward reference - `x: "Thing"` - is the only mention some
                    # symbols get. It is source, so read it as source.
                    named.update(_names_in_expression(child.value))
    return named


def test_no_star_import_reaches_into_the_package() -> None:
    """`from markdown_memory.parser import *` is the one spelling the guard cannot read.

    It names no symbol, so nothing below could tell which of `parser`'s symbols it keeps
    alive - the check would report live code as dead. Modelling it would mean treating
    every symbol in a starred module as reached, which is a hole the size of the module.
    Banned instead, so the failure names the problem rather than misreporting it.
    """
    starred: list[str] = []
    for root in (PACKAGE_DIR, ROOT / "tests", ROOT / "scripts"):
        # A relative import means *this* package, so it only reaches ours from inside it:
        # `from . import *` in tests/ is the test directory's business, not this check's.
        inside = root == PACKAGE_DIR
        for path in root.glob("*.py"):
            for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
                if not isinstance(node, ast.ImportFrom):
                    continue
                written = node.module or ""
                # `markdown_memory` or a module in it - never `markdown_memory_tools`.
                ours = written == PACKAGE or written.startswith(f"{PACKAGE}.")
                if (ours or (node.level and inside)) and any(
                    alias.name == "*" for alias in node.names
                ):
                    starred.append(f"{path.name}:{node.lineno}")
    assert starred == [], (
        "star imports of package modules hide what is used from the dead-code check: "
        + "; ".join(starred)
    )


def _read_in(source: str) -> set[str]:
    """Names this one file reads *at module scope*. A `Store` target is a definition.

    Scope resolution is CPython's own, through `symtable`, rather than a walk of every
    `Load`: a plain walk makes any local variable vouch for a dead top-level name it
    happens to share a spelling with. Hand-rolling the rules instead was tried and
    rejected - review found four separate corners it got wrong (a parameter's default and
    a decorator are evaluated in the *enclosing* scope, a comprehension's first iterable
    likewise, a method does not close over class attributes, and a walrus inside a
    comprehension binds outside it). `symtable` is the implementation those rules come
    from, so it has no corners of its own.

    A name is a module-scope read when the module block references it, or a nested block
    references it as a global. Class blocks are counted whole: `class C: X = X` reads the
    module's `X` before binding its own, and `symtable` records only the binding.
    """
    reads: set[str] = set()

    def visit(table: symtable.SymbolTable, is_module: bool) -> None:
        for symbol in table.get_symbols():
            if symbol.is_referenced() and (
                is_module or table.get_type() == "class" or symbol.is_global()
            ):
                reads.add(symbol.get_name())
        for child in table.get_children():
            visit(child, False)

    visit(symtable.symtable(source, "<module>", "exec"), True)
    return reads | _annotated_names(source)


def _dotted(node: ast.expr) -> str:
    """`markdown_memory.parser` as written, so a two-deep attribute chain can be matched."""
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        prefix = _dotted(node.value)
        return f"{prefix}.{node.attr}" if prefix else ""
    return ""


def _names_the_module(node: ast.ImportFrom, module: str) -> bool:
    """`from markdown_memory.parser import X`, and its relative spelling `from .parser`."""
    written = node.module or ""
    return written == f"{PACKAGE}.{module}" or (bool(node.level) and written == module)


def _module_aliases(tree: ast.AST, module: str) -> set[str]:
    """The names this one file binds to `markdown_memory.<module>`, if any.

    Without this, `module.attr` is matched on spelling alone, and `db`, `config`, `parser`,
    `server` and `indexer` are all ordinary local variable names in this repository - a
    local `parser.parse(...)` from argparse would vouch for a top-level `parse` in
    `parser.py` that nothing calls. Measured: matching on spelling alone reaches 50 names
    that no import justifies.
    """
    bound: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and (
            node.module == PACKAGE or (node.level and node.module is None)
        ):
            bound.update(alias.asname or alias.name for alias in node.names if alias.name == module)
        elif isinstance(node, ast.Import):
            for alias in node.names:
                # `import markdown_memory.parser` binds the *package*, and the module is
                # then reached as `markdown_memory.parser.parse` - two attributes deep.
                # `import markdown_memory as mm` reaches it as `mm.parser.parse`.
                if alias.name == f"{PACKAGE}.{module}":
                    bound.add(alias.asname or f"{PACKAGE}.{module}")
                elif alias.name == PACKAGE:
                    bound.add(f"{alias.asname or PACKAGE}.{module}")
    return bound


def _reached_from_outside(sources: dict[str, str], module: str) -> set[str]:
    """Names other files reach in `module`: `from ....module import X`, or `alias.X` where
    that file imported the module as `alias`.

    Deliberately per-module. A package-wide pool of every name anybody reads is useless
    here: `logger` is read in eight modules, so pooling would keep a dead `logger` in a
    ninth alive forever - which is exactly the false negative this check was written
    to avoid, and exactly the one it had.
    """
    reached: set[str] = set()
    for source in sources.values():
        tree = ast.parse(source)
        aliases = _module_aliases(tree, module)
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom) and _names_the_module(node, module):
                reached.update(alias.name for alias in node.names)
            elif isinstance(node, ast.Attribute) and _dotted(node.value) in aliases:
                reached.add(node.attr)
    return reached


def test_nothing_in_the_package_is_dead() -> None:
    """A top-level name nothing reads is code that cannot be wrong, only misleading."""
    package = {path.name: path.read_text(encoding="utf-8") for path in package_modules()}
    everything = dict(package) | {
        str(path): path.read_text(encoding="utf-8")
        for directory in ("tests", "scripts")
        for path in (ROOT / directory).glob("*.py")
    }

    exported: set[str] = set()
    for node in ast.walk(ast.parse(package["__init__.py"])):
        if isinstance(node, ast.Assign) and any(
            isinstance(target, ast.Name) and target.id == "__all__" for target in node.targets
        ):
            exported.update(
                element.value
                for element in getattr(node.value, "elts", [])
                if isinstance(element, ast.Constant) and isinstance(element.value, str)
            )

    own_reads = {name: _read_in(source) for name, source in package.items()}
    outside_reach = {
        name: _reached_from_outside(
            {other: text for other, text in everything.items() if other != name},
            name.removesuffix(".py"),
        )
        for name in package
    }

    dead: list[str] = []
    candidates: set[str] = set()
    for name, source in package.items():
        for node in ast.parse(source).body:
            if isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef | ast.ClassDef):
                symbol = node.name
            elif (
                isinstance(node, ast.Assign)
                and len(node.targets) == 1
                and isinstance(node.targets[0], ast.Name)
            ):
                symbol = node.targets[0].id
            elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
                symbol = node.target.id
            else:
                continue
            if symbol.startswith("__") or symbol in exported:
                continue
            candidates.add(f"{name}:{symbol}")
            alive = symbol in own_reads[name] or symbol in outside_reach[name]
            if not alive and f"{name}:{symbol}" not in ALLOWED_UNREFERENCED:
                dead.append(f"{name}:{node.lineno} {symbol}")
    assert dead == [], "nothing references these: " + "; ".join(dead)

    stale = sorted(set(ALLOWED_UNREFERENCED) - candidates)
    assert stale == [], f"the allowlist names symbols that are gone: {stale}"


class TestTheGuardsOwnBlindSpots:
    """Every false negative a review found in the checks above, pinned.

    A guard that passes because it cannot see is worse than no guard: it is a green tick
    over the thing it was written to stop. These are the four ways this one was blind.
    """

    def test_a_line_carrying_a_trailing_comment_still_counts(self) -> None:
        assert logical_lines("value = 1  # why\n") == 1

    def test_a_string_that_is_not_a_docstring_counts(self) -> None:
        """Only the first statement of a scope is a docstring; the rest is a no-op."""
        source = 'def f() -> None:\n    """Real."""\n    """Not a docstring."""\n'
        assert logical_lines(source) == 2

    def test_a_local_variable_does_not_read_the_module_level_name(self) -> None:
        source = "VALUE = 1\n\n\ndef f() -> int:\n    VALUE = 2\n    return VALUE\n"
        assert "VALUE" not in _read_in(source)
        # The control: without the local binding, the same read does count.
        assert "VALUE" in _read_in("VALUE = 1\n\n\ndef f() -> int:\n    return VALUE\n")

    def test_a_global_declaration_reads_the_module_level_name(self) -> None:
        source = (
            "VALUE = 1\n\n\ndef f() -> int:\n    global VALUE\n    VALUE = 2\n    return VALUE\n"
        )
        assert "VALUE" in _read_in(source)

    def test_an_attribute_on_a_look_alike_local_reaches_nothing(self) -> None:
        """`parser` is an argparse local in nine files; `db` is a `Database` in twenty."""
        assert _reached_from_outside({"other.py": "parser.parse()\n"}, "parser") == set()
        imported = f"from {PACKAGE} import parser\n\nparser.parse()\n"
        assert _reached_from_outside({"other.py": imported}, "parser") == {"parse"}

    def test_a_default_or_decorator_reads_the_module_level_name(self) -> None:
        """Both are evaluated in the *enclosing* scope, whatever the function rebinds."""
        assert "VALUE" in _read_in(
            "VALUE = 1\n\n\ndef f(VALUE: int = VALUE) -> int:\n    return VALUE\n"
        )
        assert "DEC" in _read_in(
            "DEC = 1\n\n\ndef f() -> int:\n    DEC = 2\n    return DEC\n"
            "\n\n@DEC\ndef g() -> None: ...\n"
        )

    def test_a_class_body_reads_the_module_name_it_then_rebinds(self) -> None:
        assert "X" in _read_in("X = 1\n\n\nclass C:\n    X = X\n")

    def test_a_method_does_not_close_over_a_class_attribute(self) -> None:
        source = "X = 1\n\n\nclass C:\n    X = 2\n\n    def m(self) -> int:\n        return X\n"
        assert "X" in _read_in(source)

    def test_a_comprehension_reads_its_first_iterable_outside_itself(self) -> None:
        assert "ITEMS" in _read_in(
            "ITEMS = [1]\n\n\ndef f() -> list[int]:\n    return [ITEMS for ITEMS in ITEMS]\n"
        )

    def test_a_walrus_in_a_comprehension_binds_the_enclosing_scope(self) -> None:
        source = "N = 1\n\n\ndef f() -> int:\n    [(N := i) for i in range(3)]\n    return N\n"
        assert "N" not in _read_in(source)

    def test_a_name_used_only_as_a_type_is_not_dead(self) -> None:
        """Annotations are never evaluated here, so `symtable` cannot see them."""
        source = (
            "from __future__ import annotations\n\n\n"
            "class Thing: ...\n\n\ndef f(x: Thing) -> None: ...\n"
        )
        assert "Thing" in _read_in(source)

    def test_a_dotted_or_relative_import_still_reaches_the_module(self) -> None:
        dotted = f"import {PACKAGE}.parser\n\n{PACKAGE}.parser.parse()\n"
        assert _reached_from_outside({"other.py": dotted}, "parser") == {"parse"}
        assert _reached_from_outside({"other.py": "from .parser import parse\n"}, "parser") == {
            "parse"
        }

    def test_a_string_forward_reference_is_a_read(self) -> None:
        source = (
            'from __future__ import annotations\n\n\nclass T: ...\n\n\ndef f(x: "T") -> None: ...\n'
        )
        assert "T" in _read_in(source)

    def test_the_package_object_under_any_name_reaches_its_modules(self) -> None:
        aliased = f"import {PACKAGE} as mm\n\nmm.parser.parse()\n"
        assert _reached_from_outside({"other.py": aliased}, "parser") == {"parse"}
