"""Installed command and python -m entry point for the synthetic trace."""

import argparse
import json
from importlib.metadata import version

from .demo import run_demo


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Run the Unexpired Plan synthetic trace (not a benchmark)."
    )
    parser.add_argument("--calls", type=int, default=12, help="control calls (default: 12)")
    parser.add_argument(
        "--format",
        choices=("table", "jsonl"),
        default="table",
        help="human-readable table or one JSON object per line",
    )
    parser.add_argument("--version", action="version", version=version("unexpired-plan"))
    args = parser.parse_args(argv)
    if args.calls < 1:
        parser.error("--calls must be positive")

    decisions, reference_calls = run_demo(args.calls)
    if args.format == "jsonl":
        print(json.dumps({"kind": "synthetic_trace", "horizon": 10, "m": 5}))
        for decision in decisions:
            row = decision.as_dict()
            # Elapsed CPU time is neither deterministic nor a GPU cost estimate.
            row.pop("elapsed_seconds")
            print(json.dumps(row, allow_nan=False))
        print(json.dumps({"reference_calls": reference_calls}))
        return

    print("The Unexpired Plan | synthetic trace | H=10, H_exec=2, m=5")
    print("CALL  ACTION SOURCE         CANDIDATE       RUNG   REFERENCE  VETO")
    for d in decisions:
        rung = f"{d.rung_before}->{d.rung_after}"
        print(
            f"{d.call:>4}  {d.source:<20}  {d.candidate or '-':<14}  "
            f"{rung:<5}  {'yes' if d.reference_forward else 'no':<9}  "
            f"{', '.join(d.veto_reasons) or '-'}"
        )
    print("Reference calls: " + ", ".join(map(str, reference_calls)))
    print("A veto uses the saved slice; demotion persists until the episode resets.")
    print("Synthetic inputs and costs; this trace does not measure benchmark performance.")


if __name__ == "__main__":
    main()
