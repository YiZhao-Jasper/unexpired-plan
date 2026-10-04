# Integrating an action-chunking policy

## Predictor contract

Each predictor accepts the current observation and returns a finite NumPy-compatible array with shape `[H, action_dimension]`. The reference and every candidate must use the same horizon, action ordering, normalization, and execution convention. The monitor executes exactly the first `H_exec` actions per control call.

The package does not load policy checkpoints or infer observation preprocessing. Implement those in the policy adapter. If a model emits tensors, explicitly detach, transfer, and convert them at the adapter boundary. Keep device synchronization explicit when measuring compute cost.

`ActionNormalizer(center, scale)` applies `(action − center) / scale` per dimension; its inverse is `normalized * scale + center`. `NormalizedPolicy` wraps a physical-coordinate predictor. When the model already emits its own normalized action space, do not normalize it a second time. Keep rotation, gripper, and joint conventions consistent.

## Calibration and selection

1. Collect unaccelerated reference replay, grouped by episode, as full normalized action chunks.
2. Use `calibrate_repetition` to estimate a threshold for near repetition. Validate it on separate reference replay and inspect per-episode false vetoes.
3. Measure reference-versus-reference drift at each reference age. Supply the resulting calibration through `drift_by_age`. Only candidates with `mechanism="recompute"` use this correction.
4. Run each accelerator in closed loop under its own actions. Score against reference predictions at the states that candidate visits. `score_closed_loop` expects an environment exposing `reset(initial_state)` and `step(action_prefix) -> (next_observation, done)`. It closes the environment when a `close()` hook exists and stops a malformed episode after `max_calls`.
5. Repeat scoring until the candidate's tail score stabilizes, as described in the paper. `select_candidate` computes the 95th-percentile score and chooses the cheapest candidate at or below `theta` from supplied `state_distribution="on_policy"` records. Success labels are not read by this selector.
6. Construct a ladder ordered by measured cost, from cheapest to more conservative. Candidate mechanism labels are `recompute`, `reuse`, and `blend`.

`examples/calibrate_and_select.py` exercises these APIs with clearly labeled synthetic data. Its numbers are illustrative configuration inputs, not recommended thresholds for a robot.

## Episode lifecycle

Instantiate or reset the monitor at each episode boundary. Reset any stateful reference/candidate predictor separately. Call `monitor.step(observation)` once per executed action prefix, then execute `decision.executed` in the original action units after any required denormalization.

Each `Decision` records the source, candidate, reference age, raw/corrected deviation, repetition score, veto reasons, ladder state, and whether a reference forward pass occurred. `as_dict()` makes actions JSON serializable. A veto fallback contains the aligned reference prefix already in memory. The next call uses the demoted rung, unless the ladder is exhausted.

Do not reset the monitor at a reference refresh; that would erase the absorbing response. Do not call the reference inside an accelerated predictor without accounting for that work. An invalid prediction raises an exception and must be handled by the surrounding controller before hardware execution.

## Fair evaluation

Pair candidate and reference trials on identical initial states. Retain outcomes per episode and cluster IDs for the initial states. `paired_bootstrap` resamples those clusters while preserving the pairing, returning the success-rate difference in percentage points and a percentile 95% interval. The interval must lie strictly within the specified margin to report equivalence. The default bootstrap seed is a reproducibility choice for the helper, not the paper's experiment seed.

Measure compute on the same hardware, precision, and batch configuration, with appropriate device synchronization. Count reference refreshes and all candidate computation. `effective_speedup(m, raw_speedup)` describes the ideal fixed-rung mean compute model; a real run with demotion must use its actual measured costs and call counts.

Compare with the same reference schedule without the monitor, not only with an unaccelerated policy. Also distinguish scheduled open-loop replay from a fresh paid reference comparison. Those baselines answer different questions.

## Website maintenance

The complete website is in `docs/` and uses relative asset URLs. GitHub Pages should serve the `main` branch and `/docs` folder. It needs no bundler, Node dependency, analytics service, or external runtime.

When the archival paper becomes available, add its verified URL to the website, README, and citation metadata together. Keep `research/reported_results.json`, the README result table, and the website synchronized. Do not relabel the controlled animation as a trained-policy benchmark rollout.
