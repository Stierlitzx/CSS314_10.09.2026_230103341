"""CPU-only checks for report gating, not additional CUDA tasks."""
import json
from pathlib import Path

import pytest

from generate_submission_token import validate_report
from lab_support import ROOT, source_manifest


def test_notebook_is_valid_and_unexecuted():
    notebook = json.loads((ROOT / "colab_launcher.ipynb").read_text(encoding="utf-8"))
    assert notebook["nbformat"] == 4
    for cell in notebook["cells"]:
        if cell["cell_type"] == "code":
            assert cell["outputs"] == []
            assert cell["execution_count"] is None
            compile("".join(cell["source"]), "colab_cell", "exec")


def test_simulated_reports_cannot_be_signed():
    with pytest.raises(ValueError, match="complete real-GPU"):
        validate_report({"mode": "simulator", "certified": False})


@pytest.mark.parametrize("field,value", [
    ("counts", {"failed": 0, "skipped": 1}),
    ("passed_task_numbers", list(range(1, 50))),
    ("source_manifest", {}),
    ("module", 1),
    ("environment", {}),
])
def test_incomplete_or_stale_reports_are_rejected(field, value):
    report = {"mode": "real_gpu", "certified": True, "pytest_exit_code": 0,
              "module": None, "passed_task_numbers": list(range(1, 51)),
              "counts": {"failed": 0, "skipped": 0},
              "environment": {"nvidia_smi": "synthetic_metadata_for_gate_unit_test"},
              "source_manifest": source_manifest()}
    report[field] = value
    with pytest.raises(ValueError):
        validate_report(report)