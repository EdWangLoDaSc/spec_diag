"""Auto-tagger — extract capability tags from code via AST static analysis.

Replaces LLM-generated free-form tags with objective, deterministic tags
based on what the code actually does. Eliminates tag fragmentation
("recursion" / "tree_recursion" / "recursive_traversal" → just "recursion").

Fixed taxonomy (~20 tags):
  Control flow:  recursion, loop, nested_loop, conditional
  Data structures: list, dict, set, string, tuple, stack_queue, tree_graph
  Algorithms:    sorting, search, dp, math, bitwise
  Patterns:      comprehension, lambda, class, exception
  Complexity:    short (<10 lines), medium (10-30), long (>30)
"""

from __future__ import annotations

import ast
from typing import Any


def auto_tag(code: str) -> list[str]:
    """Extract capability tags from Python code via AST analysis."""
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return ["unparseable"]

    tags: set[str] = set()
    visitor = _TagVisitor()
    visitor.visit(tree)
    tags.update(visitor.tags)

    # Complexity by line count
    lines = [l for l in code.strip().split("\n") if l.strip() and not l.strip().startswith("#")]
    if len(lines) <= 10:
        tags.add("short")
    elif len(lines) <= 30:
        tags.add("medium")
    else:
        tags.add("long")

    return sorted(tags)


class _TagVisitor(ast.NodeVisitor):
    """Walk the AST and collect tags."""

    def __init__(self) -> None:
        self.tags: set[str] = set()
        self._func_names: set[str] = set()
        self._loop_depth = 0

    # ---- control flow ----

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        self._func_names.add(node.name)
        self.generic_visit(node)
        # Check recursion: function calls itself
        for child in ast.walk(node):
            if (isinstance(child, ast.Call)
                    and isinstance(child.func, ast.Name)
                    and child.func.id == node.name):
                self.tags.add("recursion")
                break

    def visit_For(self, node: ast.For) -> None:
        self._loop_depth += 1
        if self._loop_depth >= 2:
            self.tags.add("nested_loop")
        else:
            self.tags.add("loop")
        self.generic_visit(node)
        self._loop_depth -= 1

    def visit_While(self, node: ast.While) -> None:
        self._loop_depth += 1
        if self._loop_depth >= 2:
            self.tags.add("nested_loop")
        else:
            self.tags.add("loop")
        self.generic_visit(node)
        self._loop_depth -= 1

    def visit_If(self, node: ast.If) -> None:
        self.tags.add("conditional")
        self.generic_visit(node)

    # ---- data structures ----

    def visit_List(self, node: ast.List) -> None:
        self.tags.add("list")
        self.generic_visit(node)

    def visit_Dict(self, node: ast.Dict) -> None:
        self.tags.add("dict")
        self.generic_visit(node)

    def visit_Set(self, node: ast.Set) -> None:
        self.tags.add("set")
        self.generic_visit(node)

    def visit_Subscript(self, node: ast.Subscript) -> None:
        # list/dict/string indexing
        self.generic_visit(node)

    def visit_Call(self, node: ast.Call) -> None:
        name = ""
        if isinstance(node.func, ast.Name):
            name = node.func.id
        elif isinstance(node.func, ast.Attribute):
            name = node.func.attr

        # Data structure constructors & methods
        if name in ("dict", "defaultdict", "OrderedDict", "Counter"):
            self.tags.add("dict")
        elif name in ("set", "frozenset"):
            self.tags.add("set")
        elif name in ("list", "append", "extend", "insert", "pop", "remove"):
            self.tags.add("list")
        elif name in ("tuple",):
            self.tags.add("tuple")
        elif name in ("deque", "Queue", "LifoQueue"):
            self.tags.add("stack_queue")
        # Sorting
        elif name in ("sort", "sorted"):
            self.tags.add("sorting")
        # String methods
        elif name in ("split", "join", "strip", "replace", "find",
                       "startswith", "endswith", "upper", "lower",
                       "isdigit", "isalpha"):
            self.tags.add("string")
        # Math
        elif name in ("sqrt", "pow", "abs", "min", "max", "sum",
                       "factorial", "gcd", "lcm", "comb", "perm",
                       "ceil", "floor", "log", "log2"):
            self.tags.add("math")
        # Search
        elif name in ("bisect", "bisect_left", "bisect_right",
                       "index", "find"):
            self.tags.add("search")

        self.generic_visit(node)

    # ---- patterns ----

    def visit_ListComp(self, node: ast.ListComp) -> None:
        self.tags.add("comprehension")
        self.generic_visit(node)

    def visit_SetComp(self, node: ast.SetComp) -> None:
        self.tags.add("comprehension")
        self.tags.add("set")
        self.generic_visit(node)

    def visit_DictComp(self, node: ast.DictComp) -> None:
        self.tags.add("comprehension")
        self.tags.add("dict")
        self.generic_visit(node)

    def visit_GeneratorExp(self, node: ast.GeneratorExp) -> None:
        self.tags.add("comprehension")
        self.generic_visit(node)

    def visit_Lambda(self, node: ast.Lambda) -> None:
        self.tags.add("lambda")
        self.generic_visit(node)

    def visit_ClassDef(self, node: ast.ClassDef) -> None:
        self.tags.add("class")
        self.generic_visit(node)

    def visit_Raise(self, node: ast.Raise) -> None:
        self.tags.add("exception")
        self.generic_visit(node)

    def visit_Try(self, node: ast.Try) -> None:
        self.tags.add("exception")
        self.generic_visit(node)

    # ---- operators ----

    def visit_BinOp(self, node: ast.BinOp) -> None:
        if isinstance(node.op, (ast.LShift, ast.RShift, ast.BitAnd,
                                 ast.BitOr, ast.BitXor)):
            self.tags.add("bitwise")
        elif isinstance(node.op, (ast.Mod, ast.FloorDiv, ast.Pow)):
            self.tags.add("math")
        self.generic_visit(node)

    # ---- imports ----

    def visit_Import(self, node: ast.Import) -> None:
        for alias in node.names:
            self._check_import(alias.name)

    def visit_ImportFrom(self, node: ast.ImportFrom) -> None:
        if node.module:
            self._check_import(node.module)

    def _check_import(self, module: str) -> None:
        parts = module.split(".")
        if "math" in parts:
            self.tags.add("math")
        elif "collections" in parts:
            self.tags.add("dict")  # Counter, defaultdict, deque etc
        elif "heapq" in parts:
            self.tags.add("stack_queue")
        elif "bisect" in parts:
            self.tags.add("search")
        elif "itertools" in parts:
            self.tags.add("comprehension")
        elif "functools" in parts:
            self.tags.add("lambda")  # reduce, partial, etc

    # ---- string detection ----

    def visit_Constant(self, node: ast.Constant) -> None:
        if isinstance(node.value, str) and len(node.value) > 1:
            self.tags.add("string")
        self.generic_visit(node)

    def visit_JoinedStr(self, node: ast.JoinedStr) -> None:
        self.tags.add("string")
        self.generic_visit(node)
