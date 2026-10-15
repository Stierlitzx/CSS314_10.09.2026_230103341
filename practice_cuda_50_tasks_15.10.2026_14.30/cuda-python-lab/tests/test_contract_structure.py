"""Static API coverage, independent of CUDA availability."""
import ast
import re

from lab_support import ROOT


def test_all_fifty_signatures_match_the_task_sheet():
    sheet = (ROOT / "TASK_SHEET.md").read_text(encoding="utf-8")
    signatures = re.findall(r"\| (\d{2}) \| `([a-z0-9_]+)\(([^)]*)\)`", sheet)
    assert len(signatures) == 50
    modules = {}
    for path in (ROOT / "lab_kernels").glob("module*.py"):
        for node in ast.parse(path.read_text(encoding="utf-8")).body:
            if isinstance(node, ast.FunctionDef):
                modules[node.name] = [arg.arg for arg in node.args.args]
    for number, name, args in signatures:
        assert name in modules, f"Task {number}: missing {name}"
        assert modules[name] == [arg.strip() for arg in args.split(",")], name


def test_each_task_has_a_reference_test():
    found = set()
    for path in (ROOT / "tests").glob("test_module*.py"):
        for node in ast.parse(path.read_text(encoding="utf-8")).body:
            if isinstance(node, ast.FunctionDef):
                match = re.match(r"test_task_?(\d+)", node.name)
                if match:
                    found.add(int(match.group(1)))
    assert found == set(range(1, 51))