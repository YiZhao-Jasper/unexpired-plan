import json
import subprocess
import sys

import pytest

from unexpired_plan.cli import main
from unexpired_plan.demo import run_demo


def test_demo_accounts_for_reference_calls_and_absorbing_demotion():
    decisions, calls = run_demo()
    assert calls == [0, 5, 10]
    assert decisions[3].source == "unexpired_fallback"
    assert not decisions[3].reference_forward
    assert all(d.rung_after == 1 for d in decisions[3:])


def test_installed_module_jsonl_is_deterministic_and_machine_readable():
    command = [sys.executable, "-m", "unexpired_plan", "--format", "jsonl"]
    first = subprocess.run(command, check=True, capture_output=True, text=True).stdout
    second = subprocess.run(command, check=True, capture_output=True, text=True).stdout
    assert first == second
    rows = [json.loads(line) for line in first.splitlines()]
    assert rows[0]["kind"] == "synthetic_trace"
    assert rows[-1] == {"reference_calls": [0, 5, 10]}
    assert len(rows) == 14


def test_cli_rejects_invalid_call_count():
    result = subprocess.run(
        [sys.executable, "-m", "unexpired_plan", "--calls", "0"],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 2 and "must be positive" in result.stderr


def test_human_readable_entrypoint_and_version(capsys):
    main(["--calls", "4"])
    output = capsys.readouterr().out
    assert "unexpired_fallback" in output and "plan_deviation" in output
    assert "Reference calls: 0" in output
    with pytest.raises(SystemExit) as exc:
        main(["--version"])
    assert exc.value.code == 0
    assert capsys.readouterr().out.strip() == "0.2.0"
