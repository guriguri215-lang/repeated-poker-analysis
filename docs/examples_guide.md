# Examples Guide

## Purpose

- Explain what each example under `examples/` demonstrates.
- Help readers choose the right example to run first.
- Clarify that the examples are small abstract demonstrations, not real poker
  recommendations.

## Recommended order

1. `examples/nuts_chop_river.py` - start here: the smallest benchmark, where
   every showdown is a chop, so the accounting is easy to check by hand.
2. `examples/candidate_library.py` - generate candidates and compare them
   against a baseline Villain.
3. `examples/candidate_filters.py` - prune candidates before comparison and read
   the exclusion reasons.
4. `examples/analysis_pipeline.py` - the most important example for an MVP check:
   it wires every stage together end to end.
5. `examples/markdown_summary.py` - render an existing report as Markdown.
6. `examples/analysis_report.py` - build the consolidated report directly, at a
   lower level than the pipeline.
7. `examples/value_bluff_river.py` - a contrasting non-chop spot.

The order moves from the simplest hand-checkable tree, through candidate
generation / filtering, to the full pipeline, and finally to presentation and
contrast examples. For a quick MVP sanity check, `analysis_pipeline.py` is the
single most important example to run.

## Example reference

### `examples/nuts_chop_river.py`

- **What it demonstrates**: the true nuts-chop river benchmark, where any legal
  holding chops at showdown, plus the exact Villain best response to a fixed
  Hero strategy.
- **Why it matters**: it is small enough to verify by hand, so it anchors trust
  in the engine.
- **What to look for**: the best-response EVs and the rake accounting; with zero
  rake the payoffs are zero-sum.
- **Limitations**: a single tiny abstract tree, not a real board or range.

### `examples/value_bluff_river.py`

- This is the former nuts-chop example, now separated as a value/bluff example.
- Shows a non-chop, value/bluff-like abstract spot where showdowns are not
  always tied.
- Useful as a contrast to the true nuts-chop benchmark.
- It is not the main repeated-game MVP example.

### `examples/candidate_library.py`

- Shows candidate generation and comparison against a baseline Villain strategy.
- Useful to understand candidates before running the full pipeline.

### `examples/candidate_filters.py`

- Shows pre-comparison pruning of generated candidates.
- Explains the kept/excluded split and the English exclusion reasons.

### `examples/analysis_report.py`

- Shows consolidated report building directly.
- More detailed but lower-level than the pipeline.

### `examples/markdown_summary.py`

- Shows presentation-only Markdown rendering of an analysis report.
- Does not change analysis results.

### `examples/analysis_pipeline.py`

- The recommended MVP entry point.
- Wires generation, filtering, comparison, report, detection, and the Markdown
  summary into one call.
- The best example for an end-to-end sanity check.

## How to run examples

The examples need both `src` and `examples` on the Python path (some reuse the
nuts-chop tree and baseline strategies defined in other example modules).

```powershell
python examples/nuts_chop_river.py
python examples/candidate_library.py
python examples/candidate_filters.py
python examples/analysis_pipeline.py
python examples/markdown_summary.py
python examples/analysis_report.py
python examples/value_bluff_river.py
```

## How to interpret common output fields

- **generated**: number of candidates produced before filtering.
- **kept**: candidates that survived the pre-filter and are compared.
- **excluded**: candidates removed by the pre-filter.
- **exclusion reasons**: English codes explaining why a candidate was removed
  (for example `l1_distance_exceeds_limit`, `required_observations_below_limit`).
- **fixed_hero_ev**: Hero EV when the candidate is locked and Villain keeps the
  baseline strategy (no best response yet).
- **post_response_hero_ev_worst**: Hero EV after Villain plays its worst-for-Hero
  exact best response to the locked candidate.
- **t_deadline**: the latest opportunity at which Villain may adapt while the
  locked policy stays at least as valuable as baseline (or `-` when none).
- **t_detect_estimated_opportunities**: estimated opportunities before the
  candidate is statistically distinguishable from baseline at its information
  set (only when an occurrence probability is supplied).
- **detected_adaptation_is_at_least_baseline**: whether Hero is at least at
  baseline EV at the estimated detection timing (the economic read).

## What examples are not

- They are not real hand recommendations.
- They are not proof of profitable play.
- They are not full solver outputs.
- They do not import real ranges.
- They do not model full opponent adaptation.

## Related docs

- [MVP Walkthrough](mvp_walkthrough.md)
- [Assumptions and Limitations](assumptions_and_limitations.md)
