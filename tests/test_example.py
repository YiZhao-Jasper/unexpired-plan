import json
import subprocess
import sys
from pathlib import Path


def test_complete_example_calibrates_selects_and_executes_feedback():
    root = Path(__file__).resolve().parents[1]
    completed = subprocess.run(
        [sys.executable, str(root / "examples/closed_loop.py")],
        check=True,
        capture_output=True,
        text=True,
        timeout=30,
    )
    result = json.loads(completed.stdout)
    assert result["kind"] == "synthetic_closed_loop_example"
    assert result["selected"]["name"] == "fast"
    assert result["selected"]["tail"] < 0.15
    assert result["repetition_calibration"]["threshold"] > 0
    assert result["reference_calls"] == [0, 5, 10, 15]
    assert result["fallback_calls"] == [3]
    assert result["final_rung"] == 1
    assert result["final_distance_to_target"] < 0.001
