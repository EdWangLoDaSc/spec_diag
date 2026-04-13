"""Sanity test for the copied PythonExecutor.

Confirms the AZR-sourced executor imports cleanly in the new package tree
and can run trivial code.
"""

import pytest


def test_import():
    from spec_diag.executors.python_executor import PythonExecutor  # noqa: F401


@pytest.mark.xfail(
    reason="PythonExecutor runtime smoke test — enable once env deps are installed"
)
def test_run_trivial():
    from spec_diag.executors.python_executor import PythonExecutor

    executor = PythonExecutor()
    out, status = executor.run_code("print(1 + 1)", inputs=None, imports=None)
    assert "2" in out
    assert status == "Done"
