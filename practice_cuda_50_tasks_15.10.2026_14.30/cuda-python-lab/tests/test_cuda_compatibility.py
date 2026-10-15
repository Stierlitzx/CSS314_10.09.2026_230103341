"""Regression guards for compiler failures observed on the Colab Tesla T4.

These CPU-only checks do not replace compiling/executing on a real GPU.
"""
import ast

from lab_support import ROOT


def test_shared_allocations_use_literal_shapes():
    """Avoid arithmetic/global names inferred as nonliteral CUDA shape types."""
    allocations = 0
    for path in (ROOT / "lab_kernels").glob("module*.py"):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            if ast.unparse(node.func) != "cuda.shared.array":
                continue
            allocations += 1
            shape = node.args[0] if node.args else next(
                keyword.value for keyword in node.keywords if keyword.arg == "shape")
            assert isinstance(shape, (ast.Constant, ast.Tuple)), (
                f"{path.name}:{node.lineno}: shared shape must be a literal")
            dimensions = shape.elts if isinstance(shape, ast.Tuple) else [shape]
            assert all(isinstance(d, ast.Constant) and type(d.value) is int
                       and d.value > 0 for d in dimensions), (
                f"{path.name}:{node.lineno}: nonliteral shared dimension")
    assert allocations > 0


def test_cooperative_kernel_requests_cpp17_and_cooperative_launch():
    path = ROOT / "lab_kernels" / "module5_streams_cupy.py"
    tree = ast.parse(path.read_text(encoding="utf-8"))
    function = next(node for node in tree.body if isinstance(node, ast.FunctionDef)
                    and node.name == "grid_cooperative_barrier")
    calls = [node for node in ast.walk(function) if isinstance(node, ast.Call)
             and ast.unparse(node.func) == "cp.RawKernel"]
    assert len(calls) == 1
    keywords = {item.arg: item.value for item in calls[0].keywords}
    assert "--std=c++17" in ast.literal_eval(keywords["options"])
    assert ast.literal_eval(keywords["enable_cooperative_groups"]) is True