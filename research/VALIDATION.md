# Release validation

The initial public release was validated on Linux with Python 3.11 and Chrome 151. Computation and browser rendering were performed on a separate compute server.

## Core and package

- `python -m pytest -q`: **49 passed**.
- `python examples/monitor_trace.py`: completed; reference calls were `[0, 5, 10]`, with a saved-reference fallback at call 3 and persistent demotion afterward.
- `python examples/calibrate_and_select.py`: completed using labeled synthetic reference replay and candidate records.
- A wheel was built, installed with its declared dependencies in a separate environment, and its installed CLI was run from outside the source checkout.

The tests cover reference accounting, time alignment, nondivisible horizons, one-call horizons, repetition, drift correction, monotone ladder response, rejected predictions, history expiration, on-policy scoring, calibration boundaries, clustered paired evaluation, and numerical behavior.

## Website

**38 browser checks passed** for the presentation update, including:

- Layout at widths 360, 390, 820, 1440, and 1920 pixels, with no page-wide horizontal overflow.
- Synchronized playback and seeking of both 13-second comparison clips.
- All five temporal-alignment states and reference refresh on expiration.
- Agreement, drift, and repetition interaction states.
- Original-paper image switching and vector-figure dialogs.
- The 240-second V3 film and chapter navigation.
- Native 1920 × 1080 video dimensions for both comparison clips and the full film; the V3 source streams are retained without another lossy video encode.
- AirLab's logo appears before the paper title in the hero.
- Reduced-motion behavior, English content, the two research authors, font/image loading, and preservation of reported uncertainty.
- No JavaScript exceptions, failed asset requests, or external runtime assets.

The README animation is rendered at 1920 × 1008 from the same matched V3 simulation clips, with brief overlays explaining saved actions, aligned checks, and persistent demotion. It remains a synthetic illustration, not a benchmark measurement. Its source clips play at the same speed; the final state is held before the animation repeats.

The public file set is checked separately for private production paths and submission metadata. Media attribution is preserved. These checks validate this implementation and presentation; they do not reproduce the paper's benchmark results or establish hardware safety.
