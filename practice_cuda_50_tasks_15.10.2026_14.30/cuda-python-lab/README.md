# CUDA Python — 50 Practical Tasks

**Ayazgaliyev Aibek, 230103341 · 15.10.2026, 14:30**

The complete report, environment limitations, layout, and submission instructions
are in [the practice-lab README](../README.md). The task contract summary is in
[TASK_SHEET.md](TASK_SHEET.md). Run from this directory:

```shell
python -m pip install -r requirements.txt
python autograder.py
```

For functional CPU debugging only (not GPU certification):

```shell
python autograder.py --simulator --output result/local_simulator_report.json
```

The Colab launcher locates an uploaded/cloned lab and runs the real-GPU suite.
Module docstrings document each kernel's launch geometry and supported limits.
There are no silent CPU fallbacks, fabricated hardware results, or default HMAC keys.