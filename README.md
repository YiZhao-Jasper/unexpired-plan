<div align="center">

<h1>The Unexpired Plan:<br>A Free Monitor for Accelerated Diffusion Policies</h1>

**[Yi Zhao](https://github.com/YiZhao-Jasper) · [Sebastian Scherer](https://www.ri.cmu.edu/ri-faculty/sebastian-scherer/)**

**[AirLab](https://theairlab.org/) · [The Robotics Institute](https://www.ri.cmu.edu/) · Carnegie Mellon University**

[Project website](https://yizhaojasper.com/unexpired-plan/) · [Quick start](#quick-start) · **[Core code](src/unexpired_plan/core.py)** · [Integration guide](examples/README.md) · [Research film](https://yizhaojasper.com/unexpired-plan/#film)

<img alt="Python 3.10 and newer" src="https://img.shields.io/badge/Python-3.10%2B-05639d?style=flat-square"> <img alt="NumPy core" src="https://img.shields.io/badge/Core-NumPy-05639d?style=flat-square"> <img alt="MIT code license" src="https://img.shields.io/badge/Code-MIT-05639d?style=flat-square">

</div>

<p align="center"><a href="https://yizhaojasper.com/unexpired-plan/#comparison"><img src="https://raw.githubusercontent.com/YiZhao-Jasper/unexpired-plan/main/docs/assets/comparison.gif" alt="Matched simulation: keep a reference plan, check accelerated actions, then reuse the saved slice and retain a more conservative accelerator after a veto. Monitoring off misses the target; Unexpired Plan reaches it." width="100%"></a></p>

<p align="center"><sub>Controlled, synthetic MuJoCo illustration · same disturbance, reference schedule, and playback speed.<br>Illustrates the mechanism; these trials are not benchmark measurements. <a href="https://yizhaojasper.com/unexpired-plan/#comparison">Watch the synchronized 1080p comparison</a> · <a href="https://yizhaojasper.com/unexpired-plan/#illustration-note">Details and credits</a>.</sub></p>

## A plan can outlive the call that made it

Action-chunking policies predict more actions than they execute immediately. **The Unexpired Plan** uses the remaining actions of a previously computed reference plan to monitor accelerated predictions—without an additional reference forward pass for the check.

1. **Align.** Compare the next executed actions with the same future indices of the saved plan.
2. **Check.** Detect excessive action deviation and near repetition of recent emitted chunks.
3. **Remember.** On a veto, execute the saved reference slice and permanently demote the accelerator for the episode.

## Quick start

Python 3.10 or newer is required. The core has one runtime dependency: NumPy. No GPU, model download, or simulator is needed for the examples.

```bash
git clone --depth 1 https://github.com/YiZhao-Jasper/unexpired-plan.git
cd unexpired-plan
python -m venv .venv
source .venv/bin/activate
python -m pip install .
python -m unexpired_plan
```

The command prints an inspectable decision table. At call **3**, a biased proposal triggers fallback and moves the ladder from `fast` to `conservative`. Reference calls remain **0, 5, 10**. No new reference call is made on the vetoing call, and demotion persists across refreshes.

Run the complete calibration → selection → deployment example:

```bash
python examples/closed_loop.py
```

This example includes a small feedback environment, physical-to-normalized action conversion, reference replay, both calibrated checks, candidate rollouts on their own states, and persistent fallback. It prints the selected candidate, calibration values, reference/fallback calls, and final target distance. **Its policies and costs are synthetic; it does not reproduce a paper benchmark.**

For a reproducible JSON trace or the installed console command:

```bash
python -m unexpired_plan --format jsonl > trace.jsonl
unexpired-plan --calls 12
```

On Windows, activate the environment with `.venv\Scripts\Activate.ps1` instead of `source`. Run `python -m unexpired_plan --help` for CLI options.

## Start with the code

| Task | Entry point |
| :--- | :--- |
| Read the online algorithm | **[`src/unexpired_plan/core.py`](src/unexpired_plan/core.py)** — `Candidate`, `MonitorConfig`, `UnexpiredPlanMonitor` |
| Run an end-to-end integration | **[`examples/closed_loop.py`](examples/closed_loop.py)** |
| Connect your own policy | [`examples/README.md`](examples/README.md#connect-your-policy) — action units, calibration, episode lifecycle |
| Calibrate from reference replay | [`calibration.py`](src/unexpired_plan/calibration.py) |
| Score and select candidates | [`selection.py`](src/unexpired_plan/selection.py) |
| Compare baselines and paired outcomes | [`baselines.py`](src/unexpired_plan/baselines.py), [`evaluation.py`](src/unexpired_plan/evaluation.py) |
| Check equations and scope | [Paper-to-code mapping](examples/README.md#paper-to-code-mapping), [`configs/paper_protocol.json`](configs/paper_protocol.json) |

Each predictor returns a finite `[H, action_dimension]` array in the same **normalized action space**. The monitor returns `decision.executed`, an `[H_exec, action_dimension]` prefix. The integration guide covers conversion back to the environment's action units and stateful predictor reset.

**Release scope.** This package implements the monitor and supporting calibration, selection, baselines, and evaluation. Original benchmark checkpoints, the full 114-configuration sweep, accelerator ladders, calibration replays, and paired episode logs are not included. The protocol lists missing inputs explicitly. The numerical results below are paper-reported values, not outputs of the examples.

## Development and validation

```bash
python -m pip install -e '.[dev]'
python -m pytest -q
python -m ruff check src tests examples
python -m ruff format --check src tests examples
python -m build
```

Tests cover time alignment, free fallback, persistent demotion, full-chunk selection, reference replay calibration, shared prediction buffers, malformed actions, and paired evaluation. The wheel contains the Python package and MIT license; the source distribution also includes examples, tests, and protocol files. Website videos and fonts are excluded from both distributions.

**Version 0.2.0 validation:** all **93 tests** pass in fresh installations outside the source checkout with Python 3.10 / NumPy 1.24, Python 3.11 / NumPy 2.4, and Python 3.14 / NumPy 2.5. Both wheel and source-distribution installation, the console command, the module entry point, and the complete closed-loop example were exercised. These checks validate the released software, not the paper's empirical results.

## Paper-reported results

Success-rate changes are relative to the unaccelerated reference, with **95% paired interval half-widths**. The equivalence margin is ±2 percentage points.

| Policy / benchmark | Reference success | Success-rate change | Effective compute speedup | Within ±2 pp? |
| :--- | ---: | ---: | ---: | :--- |
| Cosmos / LIBERO | 98.2% | +0.1 ± 1.1 pp | 1.55× | Yes |
| π₀.₅ / LIBERO-10 | 97.0% | −0.2 ± 0.9 pp | 1.67× | Yes |
| Diffusion Policy / PushT | 58.0% | +1.2 ± 2.6 pp | 1.75× | **Undecided** |
| 3D Diffusion Policy / MetaWorld | 94.2% | +0.1 ± 1.4 pp | 3.09× | Yes |
| Diffusion Policy / ManiSkill3 insertion, held out | 71.3% | −0.2 ± 1.2 pp | 1.96× | Yes |

The original four families use 1,500 paired episodes each. The held-out insertion evaluation uses 6,000 paired episodes after freezing the design. At 1.96× effective compute speedup, speed-matched step reduction changes success by −3.9 pp; the monitored arm changes it by −0.2 ± 1.2 pp. Keeping the reference schedule but removing monitoring yields −6.7 pp at 2.09×.

On three of the original four families, the reference schedule alone is already within the margin. PushT remains undecided. All reported closed-loop evaluations are in simulation. **Compute speedup is not task-duration speedup, peak latency, energy savings, or a safety guarantee.**

Machine-readable values: [`configs/reported_results.json`](configs/reported_results.json). Protocol and missing reproduction inputs: [`configs/paper_protocol.json`](configs/paper_protocol.json).

## Repository map

```text
src/unexpired_plan/    Python library and installed CLI — the main project code
examples/             One complete closed-loop example and one integration guide
tests/                Behavioral, numerical, and command-line regression tests
configs/              Paper protocol and clearly labeled reported results
docs/                 GitHub Pages website and media
```

The website is a static site with no build step or third-party runtime. GitHub Pages serves `main /docs`. For a local preview, serve `docs/` with any static HTTP server. Website and algorithm changes share this repository's version history.

## Citation

The archival paper link will be added when it is available. In the meantime, the project can be referenced as:

```bibtex
@misc{zhao2026unexpiredplan,
  title  = {The Unexpired Plan: A Free Monitor for Accelerated Diffusion Policies},
  author = {Yi Zhao and Sebastian Scherer},
  year   = {2026},
  url    = {https://yizhaojasper.com/unexpired-plan/}
}
```

## License and acknowledgments

Original code is available under the [MIT License](LICENSE). Research figures, fonts, third-party footage, and models retain their respective rights and licenses. See [third-party notices](THIRD_PARTY_NOTICES.md) and the [full media acknowledgments](https://yizhaojasper.com/unexpired-plan/credits.html).

<p><a href="https://github.com/YiZhao-Jasper"><img src="https://avatars.githubusercontent.com/u/255689395?s=80" width="36" height="36" alt="YiZhao-Jasper's GitHub avatar" align="left"></a> Repository maintained by <strong><a href="https://github.com/YiZhao-Jasper">YiZhao-Jasper</a></strong>.<br><sub>Research authors: Yi Zhao and Sebastian Scherer · AirLab, Carnegie Mellon University.</sub></p>
