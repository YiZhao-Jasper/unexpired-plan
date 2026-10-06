<div align="center">

<h1>The Unexpired Plan:<br>A Free Monitor for Accelerated Diffusion Policies</h1>

**[Yi Zhao](https://github.com/YiZhao-Jasper) · [Sebastian Scherer](https://www.ri.cmu.edu/ri-faculty/sebastian-scherer/)**

**[AirLab](https://theairlab.org/) · [The Robotics Institute](https://www.ri.cmu.edu/) · Carnegie Mellon University**

[Paper](https://arxiv.org/abs/2610.05747) · [Website](https://yizhaojasper.com/unexpired-plan/) · [Film](https://yizhaojasper.com/unexpired-plan/#film)

</div>

<p align="center"><a href="https://yizhaojasper.com/unexpired-plan/#comparison"><img src="https://raw.githubusercontent.com/YiZhao-Jasper/unexpired-plan/main/docs/assets/comparison.gif" alt="Matched simulation: keep a reference plan, check accelerated actions, then reuse the saved slice and retain a more conservative accelerator after a veto. Monitoring off misses the target; Unexpired Plan reaches it." width="100%"></a></p>

<p align="center"><sub>Illustrative MuJoCo comparison with synthetic inputs, matched disturbance, reference schedule, and playback speed. <a href="https://yizhaojasper.com/unexpired-plan/#illustration-note">Setup and credits</a>.</sub></p>

## A plan can outlive the call that made it

Action-chunking policies predict more actions than they execute immediately. **The Unexpired Plan** uses the remaining actions of a previously computed reference plan to monitor accelerated predictions—without an additional reference forward pass for the check.

1. **Align.** Compare the next executed actions with the same future indices of the saved plan.
2. **Check.** Detect excessive action deviation and near repetition of recent emitted chunks.
3. **Remember.** On a veto, execute the saved reference slice and permanently demote the accelerator for the episode.

## Quick start

Requires Python 3.10+ and NumPy. Install the package and run the synthetic integration example:

```bash
git clone --depth 1 https://github.com/YiZhao-Jasper/unexpired-plan.git
cd unexpired-plan
python -m venv .venv
source .venv/bin/activate
python -m pip install .
python examples/closed_loop.py
```

The example covers reference replay, calibration, candidate selection, and monitored deployment. Run `python -m unexpired_plan` for a compact decision trace. See the [integration guide](examples/README.md) to connect a policy and configure its benchmark assets.

## Start with the code

| Task | Entry point |
| :--- | :--- |
| Online monitor | **[`src/unexpired_plan/core.py`](src/unexpired_plan/core.py)** |
| Calibration and candidate selection | [`calibration.py`](src/unexpired_plan/calibration.py), [`selection.py`](src/unexpired_plan/selection.py) |
| Policy integration | [`examples/closed_loop.py`](examples/closed_loop.py), [adapter guide](examples/README.md#connect-your-policy) |
| Baselines and paired evaluation | [`baselines.py`](src/unexpired_plan/baselines.py), [`evaluation.py`](src/unexpired_plan/evaluation.py) |
| Equations and experiment settings | [Paper-to-code mapping](examples/README.md#paper-to-code-mapping), [`paper_protocol.json`](configs/paper_protocol.json) |

## Experimental results

Results from **Tables II and IV of the paper**. Success-rate changes are relative to the unaccelerated reference; ± denotes the **95% paired interval half-width**. Equivalence requires the full interval to lie within ±2 percentage points.

| Policy / benchmark | Reference success | Success-rate change | Effective compute speedup | Within ±2 pp? |
| :--- | ---: | ---: | ---: | :--- |
| Cosmos / LIBERO | 98.2% | +0.1 ± 1.1 pp | 1.55× | Yes |
| π₀.₅ / LIBERO-10 | 97.0% | −0.2 ± 0.9 pp | 1.67× | Yes |
| Diffusion Policy / PushT | 58.0% | +1.2 ± 2.6 pp | 1.75× | **Undecided** |
| 3D Diffusion Policy / MetaWorld | 94.2% | +0.1 ± 1.4 pp | 3.09× | Yes |
| Diffusion Policy / ManiSkill3 insertion, held out | 71.3% | −0.2 ± 1.2 pp | 1.96× | Yes |

The original four families use 1,500 paired episodes each; held-out insertion uses 6,000 after freezing the design. On insertion, speed-matched step reduction gives −3.9 pp at 1.96×; the same reference schedule without monitoring gives −6.7 pp at 2.09×. The reference schedule alone is already within the margin on three of the original four families.

All evaluations are in simulation. Speedups measure effective inference compute, including scheduled reference calls. [Machine-readable results](configs/reported_results.json).

## Citation

```bibtex
@misc{zhao2026unexpiredplan,
  title         = {The Unexpired Plan: A Free Monitor for Accelerated Diffusion Policies},
  author        = {Yi Zhao and Sebastian Scherer},
  year          = {2026},
  eprint        = {2610.05747},
  archivePrefix = {arXiv},
  primaryClass  = {cs.RO},
  doi           = {10.48550/arXiv.2610.05747},
  url           = {https://arxiv.org/abs/2610.05747}
}
```

## License and acknowledgments

Code: [MIT License](LICENSE). Figures, fonts, and third-party media retain their respective rights. See [third-party notices](THIRD_PARTY_NOTICES.md) and [media acknowledgments](https://yizhaojasper.com/unexpired-plan/credits.html).

<p><a href="https://github.com/YiZhao-Jasper"><img src="https://avatars.githubusercontent.com/u/255689395?s=80" width="36" height="36" alt="YiZhao-Jasper's GitHub avatar" align="left"></a> Maintained by <strong><a href="https://github.com/YiZhao-Jasper">YiZhao-Jasper</a></strong>.</p>
