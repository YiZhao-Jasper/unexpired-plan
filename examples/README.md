# Integrating The Unexpired Plan

Start with [`closed_loop.py`](closed_loop.py), a complete, deterministic example of reference replay, calibration, on-policy selection, and monitored feedback control. Run it from the repository root after `python -m pip install .`:

```bash
python examples/closed_loop.py
```

The small 2-D integrator and analytic policies make the interfaces runnable without downloading a model. Its costs and disturbance are illustrative inputs, not benchmark measurements or deployment settings.

## Connect your policy

Your reference and candidate adapters each implement `predict(observation) -> ndarray[H, d]`. Every chunk must be finite and use the same horizon, action order, normalization, and time convention. Predictors must treat observations as read-only; the environment's reset adapter must preserve the supplied initial-state record. Convert model tensors explicitly at this boundary; checkpoint loading, observation preprocessing, device placement, and accelerator caches belong to the adapter.

The monitor reads the policy's **normalized** action space. If a predictor returns physical actions, wrap it with `NormalizedPolicy(predict, ActionNormalizer(center, scale))`. Normalization is `(action - center) / scale`; convert the returned prefix back with `normalizer.denormalize(decision.executed)` if the environment accepts physical units. Do not normalize a model's already-normalized output twice. Rotations and gripper coordinates must share the reference convention.

The following integration function is callable with your predictors and environment. `environment.step` must execute the supplied prefix in normalized coordinates, or perform the denormalization in its adapter as the runnable example does.

```python
from unexpired_plan import MonitorConfig, UnexpiredPlanMonitor


def run_episode(
    environment,
    initial_state,
    reference,
    ladder,
    *,
    horizon,
    repetition_threshold,
    drift_by_age,
    max_calls=1000,
):
    # ladder: cost-ordered Candidate instances, starting at the selected rung.
    monitor = UnexpiredPlanMonitor(
        reference,
        ladder,
        MonitorConfig(
            horizon=horizon,
            execution_horizon=2,
            theta=0.15,
            repetition_threshold=repetition_threshold,
            drift_by_age=drift_by_age,
        ),
    )
    for predictor in [reference, *(candidate.predict for candidate in ladder)]:
        if hasattr(predictor, "reset"):
            predictor.reset()
    observation = environment.reset(initial_state)
    trace = []
    for _ in range(max_calls):
        decision = monitor.step(observation)
        observation, done = environment.step(decision.executed)
        trace.append(decision.as_dict())
        if done:
            return trace
    raise RuntimeError("Episode did not terminate within max_calls")
```

`monitor.reset()` resets the monitor only. Reset stateful predictors separately at episode boundaries; do not reset at a reference refresh. Predictors that share weights should still have independent mutable caches and random-number state. The monitor calls only the active predictor; any candidate warm-up or extra reference call inside an adapter must be explicitly accounted for in cost.

`Decision` exposes the executed prefix, source, candidate, reference age, raw/corrected deviation, repetition score, veto reasons, rung before/after, and reference-forward flag. Invalid predictions raise before actions are returned and do not advance the monitor. Predictor side effects cannot be rolled back; an application must handle its adapter's failure state before continuing. `elapsed_seconds` includes Python overhead and is not synchronized GPU timing.

## Calibrate, select, then deploy

1. **Reference replay.** Run the unaccelerated reference on its own states. Store a full normalized chunk at every call, grouped by episode. Every calibration episode needs at least two chunks; calibrating all reference ages requires sufficiently long replay.
2. **Repetition.** `calibrate_repetition(replay)` returns an empirical episode-level lower-tail threshold. Validate false vetoes on separate reference replay. Its default target is `0.014`; the numerical threshold is data dependent. A threshold of zero disables the repetition veto.
3. **Replanning drift.** `calibrate_drift_from_replay(replay, H, H_exec)` compares each newly replanned prefix with the matching slice from anchors at calls `0, m, 2m, ...`. It returns corrections indexed by age `1 ... m-1`, in **control calls**, with `m = floor(H / H_exec)`. Only `mechanism="recompute"` receives this correction; `reuse` and `blend` do not.
4. **On-policy scoring.** Each accelerator controls its own environment. `score_closed_loop` evaluates the reference on the states that candidate visits. It scores **full chunks** with Eq. (1) while executing only their first `H_exec` rows. This is paid offline work, separate from the free online prefix check. The environment contract is `reset(initial_state) -> observation` and `step(prefix) -> (observation, done)`; an optional `close()` runs even on failure.
5. **Stability and selection.** `score_until_stable` repeats scoring across supplied initial-state batches until the pooled 95th percentile stabilizes. It returns the deviation samples, tail history, and `converged` flag. `select_candidate` selects the cheapest eligible record with tail `<= theta`; no success labels are read. It rejects an explicitly unconverged record. If no candidate qualifies, use an empty ladder to run the reference every call.
6. **Ladder.** Start at the selected candidate and retain slower, validated configurations. Costs must be positive and measured on matched hardware; they order the ladder but cannot establish that a candidate is more conservative. Establish that property separately during validation.

Use the **same `epsilon`** in calibration, selection, and `MonitorConfig`. The implementation default is `1e-8`; the manuscript does not supply its exact numerical value. `theta=0.15`, `K=8`, and `H_exec=2` are the paper settings, not universal safe settings for new policies.

## Paper-to-code mapping

| Paper behavior | Implementation | Regression coverage |
| :--- | :--- | :--- |
| Eq. (1): reference-normalized L1 deviation | [`metrics.deviation`](../src/unexpired_plan/metrics.py) | Denominator, shapes, finite values, large inputs |
| Eq. (2): compare the next executed indices | [`UnexpiredPlanMonitor.step`](../src/unexpired_plan/core.py) | Aligned slices, nondivisible horizons, `m=1` |
| Repetition over recent full chunks; drift only for recompute | [`core.py`](../src/unexpired_plan/core.py), [`calibration.py`](../src/unexpired_plan/calibration.py) | Lag blind spot, age calibration, branch-specific correction |
| Free fallback and permanent demotion | [`core.py`](../src/unexpired_plan/core.py) | No extra forward on veto; demotion persists across refresh |
| §IV-D: own-state full-chunk 95th-percentile selection | [`selection.py`](../src/unexpired_plan/selection.py) | Full-tail errors, candidate-controlled states, convergence limits |
| Eq. (3): compute speedup including reference calls | [`metrics.effective_speedup`](../src/unexpired_plan/metrics.py) | Fixed-rung ceiling at `m` |
| Eqs. (4–6): exposure and mechanism miss rates | [`metrics.py`](../src/unexpired_plan/metrics.py) | Coverage identities and dependence bounds |
| §V: paired bootstrap clustered on initial state | [`evaluation.paired_bootstrap`](../src/unexpired_plan/evaluation.py) | Binary pairs, cluster resampling, strict equivalence margin |
| Schedule-only, replay, paid comparison, admission coin | [`baselines.py`](../src/unexpired_plan/baselines.py) | Distinct compute accounting and response behavior |

The controller refreshes the reference every `m` calls. At age `j`, it checks the candidate's executed prefix against reference rows `[j*H_exec:(j+1)*H_exec]`. A veto executes that saved slice and advances the rung once, even if both checks fire. After ladder exhaustion, every call runs the reference. Only an episode reset re-arms acceleration.

## Explicit implementation choices

The paper does not specify every implementation detail. These choices are exposed rather than presented as recovered experimental settings:

- **Fallback history:** a fallback emits only `H_exec` rows. It is not padded into a fabricated full plan, and the rejected candidate is not recorded as emitted. Previous full emitted chunks remain eligible within the last `K` control calls.
- **Threshold boundaries:** deviation vetoes use strict `>`; repetition vetoes use strict `<`.
- **Repetition estimator:** take the lower order statistic of per-episode minimum replay distances. At most `floor(alpha*N)` of the calibration episodes lie strictly below it. This is an empirical calibration property, not a distribution-free guarantee.
- **Drift estimator:** the replay helper defaults to the median per age. `quantile` exposes that choice. The exact paper estimator and corrections are unavailable.
- **Score stability:** by default require two consecutive changes in the pooled 95th percentile of at most `1e-3`, bounded by ten rounds. `atol`, `stable_rounds`, and `max_rounds` are configurable implementation choices. Failure to converge is reported explicitly.
- **Baseline scope:** the admission coin is an explicit matched-admission helper. It does not recreate the separate retirement-matched experiment or unpublished accelerator configurations.

Version 0.2 corrects offline scoring to full chunks, rejects inconsistent replay/shapes, and snapshots the paid baseline's reference before a candidate can overwrite a shared output buffer. The CLI now defaults to a readable table; use `--format jsonl` for structured output.

## Evaluation and release scope

Pair trials on identical initial states. `paired_bootstrap` accepts binary outcomes and initial-state cluster IDs, preserves candidate/reference pairing, and reports a percentile 95% interval in percentage points. It requires at least two clusters. Equivalence requires the entire interval to lie **strictly** within the margin, default ±2 pp; a point estimate inside the margin is insufficient. The default bootstrap seed is a helper setting, not an original experiment seed.

Measure compute on the same hardware, precision, and batch size, synchronizing the device as needed. Include reference calls and all candidate work. Eq. (3) is a fixed-rung mean-compute model; trajectories with demotion require actual per-rung counts and costs. Compare with the same reference schedule without monitoring as well as the reference-every-call policy. Compute speedup is not task-duration speedup, peak latency, energy, or a safety guarantee.

Benchmark integration requires the frozen checkpoints, accelerator ladders, reference calibration replays, and initial-state files for the selected policy and task. The original experiment artifacts—including the 114-configuration sweep, raw timing records, and per-episode outcomes—are not bundled with this release. [`paper_protocol.json`](../configs/paper_protocol.json) records the protocol and configuration inputs. [`reported_results.json`](../configs/reported_results.json) contains the paper's results from Tables II and IV and Figure 1, with their uncertainty and interpretation. All reported closed-loop evaluations are in simulation.

The website in [`docs/`](../docs/) is served from `main /docs` by GitHub Pages. Its paired MuJoCo illustration uses synthetic inputs, choreographed object attachment, and disabled object contact. Real-robot footage supplies context, not evidence of deployment of this method. Media provenance and rights remain in [`THIRD_PARTY_NOTICES.md`](../THIRD_PARTY_NOTICES.md) and the [credits page](https://yizhaojasper.com/unexpired-plan/credits.html).
