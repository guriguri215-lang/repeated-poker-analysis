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
5. `examples/ranking.py` - sort the pipeline's report rows by diagnostic criteria.
6. `examples/markdown_summary.py` - render an existing report as Markdown.
7. `examples/analysis_report.py` - build the consolidated report directly, at a
   lower level than the pipeline.
8. `examples/value_bluff_river.py` - a contrasting non-chop spot.

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

### `examples/ranking.py`

- Sorts the pipeline's analysis report rows by diagnostic criteria (for example,
  post-response worst-case Hero EV difference or local `T_detect`).
- Diagnostic / presentation only: it does not auto-select a candidate and makes
  no optimality claim.

## How to run examples

The examples need both `src` and `examples` on the Python path (some reuse the
nuts-chop tree and baseline strategies defined in other example modules).

```powershell
python examples/nuts_chop_river.py
python examples/candidate_library.py
python examples/candidate_filters.py
python examples/analysis_pipeline.py
python examples/ranking.py
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

## JSON scenario input

Instead of a hand-written Python example, an abstract river spot can be loaded
from a JSON file. See `examples/scenarios/nuts_chop_steal_bet98.json` for the
BET=98 nuts-chop steal case.

The input has two modes:

- **single-hand mode**: a top-level `showdown` result and a single
  `baseline_hero_strategy` at the `IP_vs_bet` information set (the nuts-chop
  sample above).
- **abstract weighted range mode**: a `hero_range` list of weighted hands, each
  with its own `hand_id`, `weight`, `showdown`, and `baseline_strategy`. See
  `examples/scenarios/abstract_range_steal_bet98.json` for a two-hand range (one
  hand chops and folds at baseline, one hand wins at showdown and calls). A
  chance node draws the hand bucket; Villain shares one `OOP_river` information
  set across all buckets (it does not see Hero's hand), while Hero gets a
  per-hand `IP_vs_bet::<hand_id>` information set. This is an *abstract* range:
  the weights and per-hand showdown results are given directly. It is **not** a
  real card or hand-range parser. The two modes are mutually exclusive.

What the abstract range mode does **not** do in v1:

- It is Hero-range-only. It does not model Villain private hand buckets yet, so
  Villain cannot have its own range or matchup-dependent decisions.
- Each hand's `showdown` is a fixed abstract outcome for that Hero bucket. There
  is no showdown matrix or equity matrix that resolves Hero vs Villain holdings.
- The action tree is fixed to OOP `check`/`bet` and IP `call`/`fold`. Raises and
  arbitrary betting trees are not generated from JSON, even though the core
  `GameTree` supports arbitrary action labels.

When Villain hand buckets are added later, Villain will know its own bucket, so
each Villain bucket needs its own information set; but because Villain still does
not see Hero's bucket, that information set must be shared across Hero buckets
within the same Villain bucket.

Both modes work with the same helper scripts and runner below.

There are two helper scripts for a scenario file:

- `scripts/run_river_scenario.py` is a quick sanity check: it prints the terminal
  EVs, the baseline and locked-call responses, the candidate count, and the
  `T_deadline` table, without running the candidate-analysis pipeline.
- `scripts/run_river_scenario_analysis.py` runs the full candidate-analysis
  pipeline (generation, comparison, reporting) and prints the Markdown summary,
  plus an optional ranking section.

```powershell
python scripts/run_river_scenario.py examples/scenarios/nuts_chop_steal_bet98.json
python scripts/run_river_scenario_analysis.py examples/scenarios/nuts_chop_steal_bet98.json
```

The analysis script accepts `--horizon`, `--discount`, `--rank-by`, `--top-k`,
and `--no-markdown`. From Python the same run is available as
`run_river_scenario_analysis`:

```python
from repeated_poker import run_river_scenario_analysis

result = run_river_scenario_analysis(
    "examples/scenarios/nuts_chop_steal_bet98.json"
)
print(result.markdown_summary)
```

To stop at the building blocks instead, build the game and feed it into the
pipeline yourself:

```python
from repeated_poker import (
    build_river_steal_game_from_scenario,
    generate_shift_candidates,
    load_river_scenario_json,
)

build = build_river_steal_game_from_scenario(
    load_river_scenario_json("examples/scenarios/nuts_chop_steal_bet98.json")
)
candidates = generate_shift_candidates(
    build.tree, build.baseline_hero_strategy, build.shift_amounts
)
```

`build.tree`, `build.baseline_hero_strategy`, and `build.baseline_villain_strategy`
are also ready to pass into `run_candidate_analysis_pipeline`. This v1 input is
an abstract spot only; it does not parse real cards, hand ranges, or solver
exports.

## Related docs

- [MVP Walkthrough](mvp_walkthrough.md)
- [Assumptions and Limitations](assumptions_and_limitations.md)
