"""Regression: _pipeline_translate_and_finalize must initialise the B-model locals.

The single-model MLX path (Mac default) never assigns ``qe_b``/``spanish_b``/
``lat_b``/``tps_b``; the summary print reads them, which raised
``UnboundLocalError: qe_b`` on every final until the prologue initialised them.
Checked structurally so it runs in CI without loading the pipeline.
"""

from __future__ import annotations

import ast
from pathlib import Path

REQUIRED = {"spanish_b", "lat_b", "tps_b", "qe_b", "qe_a"}


def _prologue_assigned_names(func: ast.AsyncFunctionDef) -> set[str]:
    names: set[str] = set()
    for node in func.body:
        if isinstance(node, ast.Try):
            break
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name):
                    names.add(target.id)
    return names


def test_finalize_initialises_b_model_locals_before_try():
    tree = ast.parse(Path("dry_run_ab.py").read_text())
    funcs = [
        n
        for n in ast.walk(tree)
        if isinstance(n, ast.AsyncFunctionDef) and n.name == "_pipeline_translate_and_finalize"
    ]
    assert len(funcs) == 1
    missing = REQUIRED - _prologue_assigned_names(funcs[0])
    assert not missing, f"locals not initialised before try: {sorted(missing)}"
