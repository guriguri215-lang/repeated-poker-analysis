# MVP Walkthrough

This is a technical walkthrough of the current minimum viable workflow: what the
repository can do today, how to run it, and how to read its output. It is not a
tutorial on poker strategy and not promotional material.

## What this project currently does

- Works on small abstract two-player non-zero-sum poker subgames.
- Generates candidate Hero commitment strategies.
- Optionally pre-filters candidates.
- Compares candidates against a baseline Villain strategy.
- Computes exact Villain best responses for small trees.
- Builds an analysis report with:
  - fixed-baseline EVs
  - post-response Hero EV worst/best
  - selection labels
  - `T_deadline`
  - local `T_detect`
- Can render a Markdown summary.
- Provides a high-level pipeline API.

## What this project does not yet do

- It is **not a full poker solver**.
- It does not import real solver ranges yet.
- It does not model full tree reach for detection.
- It does not model real opponent psychology or learning speed.
- It does not guarantee profitable poker play.
- It does not provide gambling, bankroll, or financial advice.
- It does not yet support large-scale range solving.
- It does not yet implement STT / ICM / preflop push-fold analysis.

## Quick start

From the project root, run the test suite and the end-to-end example:

```powershell
python -m pytest -q
python examples/analysis_pipeline.py
```

For a one-command MVP sanity check, run:

```powershell
python scripts/check_mvp.py
```

If you want an isolated environment, see the `## Development` section of the
[README](../README.md) for the virtual-environment and editable-install steps.
The examples only need the package on the Python path; they do not require any
external solver or network access.

For a guide to every example script and the recommended order for running them,
see [Examples Guide](examples_guide.md).

## Minimal Python example

The snippet below mirrors `examples/analysis_pipeline.py`. The tree and the
baseline strategies come from the worked examples (`examples/nuts_chop_river.py`
and `examples/candidate_library.py`), so run it with both `src` and `examples`
on the Python path.

```python
from nuts_chop_river import build_nuts_chop_river, default_hero_strategy
from candidate_library import baseline_villain_strategy

from repeated_poker import (
    CandidateFilterConfig,
    CandidateGenerationConfig,
    run_candidate_analysis_pipeline,
)

tree = build_nuts_chop_river()
baseline_hero = default_hero_strategy()
baseline_villain = baseline_villain_strategy()

result = run_candidate_analysis_pipeline(
    tree,
    baseline_hero,
    baseline_villain,
    generation=CandidateGenerationConfig(shift_amounts=[0.1, 0.2]),
    horizon=5,
    profit_tolerance=-2.0,
    max_selection_l1_distance=0.3,
    detection_log_likelihood_threshold=3.0,
    detection_occurrence_probability_per_opportunity=0.5,
    filtering=CandidateFilterConfig(max_l1_distance=0.3, min_required_observations=5),
)

counts = result.filter_result.summary_counts
print(f"generated={len(result.generated_candidates)} "
      f"kept={counts.kept} excluded={counts.excluded}")
print(result.markdown_summary)
```

`run_candidate_analysis_pipeline` runs candidate generation, the optional
pre-filter, the fixed-profile comparison, the analysis report, and (by default)
the Markdown rendering. It returns a `CandidateAnalysisPipelineResult` whose
`generated_candidates`, `filter_result`, `comparison_report`, `analysis_report`,
and `markdown_summary` you can inspect directly.

## How to read the output

The Markdown summary has a configuration block, a summary-counts block, and one
table row per *kept* candidate. The key fields:

- **generated / kept / excluded**: how many candidates were produced, how many
  survived the pre-filter (and are therefore compared), and how many were
  pruned. Only kept candidates appear in the table.
- **fixed_hero_ev**: Hero EV when the candidate is locked and Villain keeps the
  fixed baseline strategy (no best response yet).
- **post_response_hero_ev_worst**: Hero EV after Villain plays its exact
  worst-for-Hero best response to the locked candidate.
- **post_response_hero_ev_worst_diff**: that worst-case Hero EV minus the
  baseline profile's Hero EV. Negative means worse than baseline.
- **robustly_profitable**: whether the worst-case post-response Hero EV strictly
  exceeds the baseline profile's Hero EV.
- **is_eligible**: whether the candidate passed selection screening (profit
  tolerance and the optional selection L1 cap).
- **t_deadline**: the latest opportunity `m` (in `1..N`) at which Villain may
  adapt while the locked policy stays at least as valuable as baseline, or `-`
  when no such opportunity exists.
- **t_detect_estimated_opportunities**: the estimated number of opportunities
  before the candidate is statistically distinguishable from baseline at its own
  information set (only present when an occurrence probability is supplied).
- **detected_adaptation_delta_from_baseline**: the Hero total-EV gap from
  baseline if Villain adapts at the estimated detection opportunity.
- **detected_adaptation_is_at_least_baseline**: whether Hero is at least at
  baseline EV at that estimated detection timing. This is the economic read.
- **exclusion_reasons**: for excluded selection labels, the English reason codes
  (for example `not_robustly_profitable`, `l1_distance_exceeds_limit`).

## T_deadline vs T_detect

These two measures answer different questions and must not be conflated:

- `T_deadline` is an **economic adaptation deadline**: how late Villain can
  switch before the locked policy stops beating baseline.
- `T_detect` is a **local observable-distribution sensitivity estimate**: how
  many observations Villain plausibly needs to tell the candidate apart from
  baseline at the candidate's own information set.
- `t_detect_is_no_later_than_t_deadline` is **only a timing comparison**
  (`estimated_opportunities <= t_deadline`). Because `t_deadline` is just the
  latest passing opportunity and Hero EV need not be monotone in the switching
  opportunity, this flag is **not an economic-safety statement**.
- For the economic read at the estimated detection timing, use:
  - `detected_adaptation_delta_from_baseline`
  - `detected_adaptation_is_at_least_baseline`

`T_detect` here is a local model conditional on reaching the candidate's
information set; it ignores tree reach probability and does not model real
opponent learning.

## Motivating nuts-chop steal regression

The repository now includes a regression test
(`tests/test_nuts_chop_steal_commitment.py`) for the original motivating
nuts-chop steal spot. The same spot can now also be loaded from JSON
(`examples/scenarios/nuts_chop_steal_bet98.json`) via the river scenario input.

Parameters:

- initial commitment = 1
- initial pot = 2
- bet = 98
- rake = 5%
- rake cap = 4

Terminal EVs:

| Line | Hero/IP EV | Villain/OOP EV |
|---|---:|---:|
| check-check | -0.05 | -0.05 |
| bet-fold | -1.00 | +1.00 |
| bet-call | -2.00 | -2.00 |

Interpretation:

- Single-hand play prefers fold for IP and bet for OOP.
- Locking IP to call makes OOP's best response check.
- This is a toy regression example, not a real-hand recommendation.

`T_deadline` inputs and results (discount = 1.0):

- baseline Hero EV = -1.00
- pre-adaptation Hero EV = -2.00
- post-adaptation Hero EV = -0.05

| N | T_deadline |
|---:|---:|
| 10 | 5 |
| 20 | 10 |
| 50 | 25 |
| 100 | 49 |

## Recommended MVP workflow

1. Start with the nuts-chop example.
2. Generate small candidate libraries.
3. Apply conservative pre-filters.
4. Compare kept candidates exactly.
5. Inspect Markdown summaries.
6. Only then consider adding richer game trees or external solver adapters.

## Known limitations

- The exact Villain response enumerates the whole pure-strategy space and is
  intended for small abstract trees only; it is guarded by a configurable
  `max_pure_strategies` limit.
- Utilities are net chips over the hand. With zero rake the game is zero-sum;
  rake makes it non-zero-sum.
- Strategy-space L1 distance is not an observable behavioural distance.
- The adaptation-deadline model uses opportunity-independent values, so
  `V_lock(m)` is monotone in `m`; the API still scans every `m` and does not
  assume monotonicity.
- Detection is a local, reach-conditional sensitivity estimate, not a
  prediction of real adaptation.
- No real card ranges, card removal, external solver I/O, CLI, file output, or
  STT / ICM are implemented yet.

For a more explicit list of modelling assumptions and non-claims, see
[Assumptions and Limitations](assumptions_and_limitations.md).

## Roadmap after MVP

- better candidate generation
- real input adapters
- more examples
- STT / ICM push-fold prototype
- documentation and DEV.to development notes

Longer-form development notes may later be published separately.
