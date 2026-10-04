# Method and paper alignment

The implementation uses the same time-aligned reference comparison, veto-only response, and persistent ladder described in the paper. This page distinguishes specified behavior from implementation choices where the paper does not determine an exact implementation.

## Algorithm

At call `t`, a policy produces an `H × d` action chunk, and only its first `H_exec` rows execute. Let `m = floor(H / H_exec)`.

1. On a reference call, compute and store a full reference chunk. Execute its first `H_exec` rows.
2. At age `j`, compare the accelerated prefix with reference rows `[j H_exec : (j+1) H_exec]`.
3. Compute `delta(A, R) = ||A − R||₁ / (||R||₁ + epsilon)` in the policy's normalized action coordinates.
4. On the recompute branch only, subtract the calibrated reference drift at age `j`, clamped at zero.
5. Compute near repetition as the minimum full-chunk deviation from full emitted chunks in the previous `K=8` control calls.
6. If corrected deviation exceeds `theta`, or repetition is below the calibrated repetition threshold, execute the aligned saved reference slice and retire the current rung.
7. Refresh the reference every `m` calls. Once the ladder is exhausted, run the reference on every call. Only an episode reset re-arms the ladder.

There is no extra reference forward pass on the vetoing call. A veto changes subsequent candidate choice, not the reference refresh schedule before ladder exhaustion.

## Implementation map

| Paper concept | Implementation | Behavioral checks |
| :--- | :--- | :--- |
| Eq. (1), normalized L1 deviation | `metrics.deviation` | Reference denominator, finite inputs, matching shapes |
| Eq. (2), aligned unexpired slice | `core.UnexpiredPlanMonitor.step` | Exact indices, nondivisible horizons, `m=1` |
| Repetition and recompute-only drift correction | `core`, `calibration` | Repetition blind spot; no correction on reuse/blend |
| Veto fallback and absorbing ladder | `core` | No extra reference call; demotion survives refresh |
| On-policy tail selection | `selection` | Candidate-visited states; success labels ignored |
| Eq. (3), compute ceiling | `metrics.effective_speedup` | Reference cost included; ceiling at `m` |
| Exposure and mechanism miss rates | `metrics` | Counting identities and dependence bounds |
| Paired, clustered evaluation | `evaluation.paired_bootstrap` | Resample initial-state clusters, preserve paired outcomes |
| Schedule and paid-comparison baselines | `baselines` | Distinct reference accounting and response behavior |

## Explicit implementation choices

- **Repetition history after fallback.** A fallback executes only a reference slice, not a new full chunk. We do not pad it into a fabricated full plan. Existing full chunks remain eligible within the last eight control calls. The paper does not completely specify history handling after a partial fallback.
- **Calibration.** The repetition estimator uses a lower order statistic of per-episode minimum reference-replay distances, with a strict `<` veto. Its target false-veto rate is 0.014 by default, matching the paper's stated empirical scale. This is an explicit estimator choice, not an assertion that it reconstructs the original calibration data or threshold.
- **Drift estimator.** The supplied calibration helper defaults to a median by age. The paper specifies calibration from reference replay and correction only on recompute, but does not provide its exact estimator or numerical corrections.
- **Threshold boundaries.** Drift vetoes use `>`; repetition vetoes use `<`. A repetition threshold of zero disables that veto.
- **Invalid predictions.** Nonfinite, wrong-horizon, or changing-dimension predictions raise an error before execution. A rejected predictor result does not advance the monitor's call or ladder state.
- **Reset.** `monitor.reset()` resets monitor state. Stateful policy adapters must also be reset at episode boundaries. The offline scoring helper calls a predictor's `reset()` hook when one is supplied.
- **Cost.** Candidate costs order the ladder. They are provided by the caller and must come from matched measurements. The monitor's elapsed time includes Python overhead and is not a substitute for synchronized accelerator timing.

## Release scope

This is a usable core reference implementation, not a claim that all empirical results have been regenerated. The repository contains neither trained policy weights nor fabricated replacements for the paper's original assets.

Full benchmark reproduction additionally needs the frozen checkpoint hashes, exact accelerator configurations and ladders, initial-state/seed files, reference calibration replays, native task setups, paired episode logs, and raw timing records listed in `configs/paper_protocol.json`.

The website and README use the reported values stored in `reported_results.json`. The moving demonstration has synthetic inputs, choreographed object attachment, and disabled object contact. Its outcomes illustrate the monitor mechanism; they do not estimate benchmark success rates.

## Interpretation

Effective compute speedup includes scheduled reference calls. A fresh reference is not free; reading its still-valid actions is. Full coverage gives a counting property, not distribution-free safety. The reference can be stale, sustained small errors can remain below threshold, and the calibration depends on the evaluated families. All paper-reported closed-loop evaluations use simulation.
