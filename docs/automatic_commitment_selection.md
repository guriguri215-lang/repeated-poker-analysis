# Automatic conditional Hero commitment selection v1

This document specifies the bounded two-player selector implemented by
`repeated_poker.automatic_commitment_selection`. It answers one conditional
question:

> Given this finite candidate library and an opponent adaptation opportunity
> `m`, which kept Hero candidate has the largest repeated-game total Hero-EV
> delta from the baseline?

It is a conditional diagnostic, not a strategy recommendation. In particular,
it is not a continuous or global optimum, an equilibrium result, a prediction
of when an opponent adapts, or a profitability guarantee.

## Search universe and response semantics

The selector consumes a completed `CandidateComparisonReport` and searches
every entry in `report.comparisons`. It deliberately does not reduce that set
to candidates carrying an `eligible`, minimum-Villain-EV, Pareto, or ranking
label. If an upstream candidate filter was requested, the report's kept set is
the bounded universe and the coverage metadata records the generated input and
kept candidate IDs.

For every adaptation opportunity `m = 1, ..., N + 1`, the selector reuses
`calculate_candidate_adaptation_deadlines` with `response_mode="worst"`. Thus a
candidate has its fixed-profile Hero EV before `m` and its exact post-response
Hero-worst EV from `m` onward. With horizon `N`, discount `d`, and candidate
`c`, the repeated total is

```text
sum(t=1..m-1) d^(t-1) * fixed_hero_ev(c)
+ sum(t=m..N) d^(t-1) * post_response_hero_worst_ev(c)
```

and the objective is that total minus the correspondingly discounted baseline
total. `m = N + 1` means no response occurs inside the modeled horizon.

## Selection threshold and ties

For a row, let `best_delta` be the maximum repeated total Hero-EV delta. A
commitment is selected only when

```text
best_delta > minimum_total_uplift + tolerance
```

The inequality is strict. Otherwise the row status is
`NO_BENEFICIAL_COMMITMENT` and `selected_candidate_id` is `null`.

All candidates within `tolerance` of `best_delta` are preserved in
`primary_tie_candidate_ids`. The representative display candidate is then
chosen deterministically using these secondary comparisons, each applying the
same tolerance where it compares floats:

1. larger post-response Hero-worst EV difference from baseline;
2. smaller strategy-space L1 distance;
3. lexicographically smaller stable candidate ID.

The display candidate remains available as tie-break evidence even in a
no-benefit row; it must not be mistaken for a selected commitment.

## Validation, bounds, and failure behavior

`AutomaticCommitmentSelectionConfig` has three controls:

- `minimum_total_uplift`: non-negative selection threshold, default `0.0`;
- `max_candidates`: maximum kept comparisons, defaulting to the existing
  candidate-generation cap;
- `max_timing_rows`: maximum `kept candidates * (N + 1)`, default `1_000_000`.

Request parameters, relevant report scalars, generated repeated values, IDs,
and coverage metadata are validated. Non-finite numbers, duplicate or empty
candidate IDs, mismatched coverage IDs, non-positive caps, or bounds violations
fail with `ValueError`. Candidate and timing-row caps are checked before the
bulk deadline results are materialized. The selector returns no partial report
on error.

An empty kept set is valid after request and coverage validation. It produces
exactly `N + 1` deterministic no-benefit rows. Input order does not affect the
serialized result.

## Result identity and serialization

`AutomaticCommitmentSelectionReport.to_dict()` emits deterministic,
JSON-compatible data and includes:

- contract version, complete status, bounded claim scope, and Hero-worst
  response semantics;
- horizon, discount, tolerance, and effective selector configuration;
- search-coverage provenance and the full candidate universe;
- one row for every `m = 1, ..., N + 1`;
- a SHA-256 baseline identity.

The v1 baseline identity algorithm hashes the canonical serialized
`FixedProfileValue` stored in the comparison report. It binds the selector
output to that report's baseline Hero, Villain, and total EV triple; it does not
claim to identify the full game tree or strategy objects.

## Direct API

```python
from repeated_poker.automatic_commitment_selection import (
    AutomaticCommitmentSelectionConfig,
    AutomaticCommitmentSearchCoverage,
    select_automatic_commitments,
)

selection = select_automatic_commitments(
    comparison_report,
    horizon=100,
    discount=0.99,
    configuration=AutomaticCommitmentSelectionConfig(
        minimum_total_uplift=0.25,
    ),
    search_coverage=AutomaticCommitmentSearchCoverage(
        input_candidate_ids=generated_candidate_ids,
        kept_candidate_ids=kept_candidate_ids,
        source="generated_candidates_after_optional_filter",
    ),
)
payload = selection.to_dict()
```

The coverage argument is optional for direct use. When omitted, the input and
kept ID sets both default to all comparisons in the report.

## Pipeline opt-in

`run_candidate_analysis_pipeline` preserves its previous behavior by default:
`result.automatic_selection_report` is `None`. Pass a configuration to opt in:

```python
from repeated_poker.automatic_commitment_selection import (
    AutomaticCommitmentSelectionConfig,
)
from repeated_poker.pipeline import run_candidate_analysis_pipeline

result = run_candidate_analysis_pipeline(
    tree,
    baseline_hero_strategy,
    baseline_villain_strategy,
    horizon=100,
    discount=0.99,
    automatic_selection=AutomaticCommitmentSelectionConfig(
        minimum_total_uplift=0.25,
    ),
)
selection = result.automatic_selection_report
```

The pipeline records generated and kept IDs plus its generation/filter
settings in the selection report. Because v1 is defined only for exact
Hero-worst response semantics, opting in while the pipeline uses any other
`response_mode` fails before candidate generation.

This opt-in report is intentionally separate from the existing analysis-report
schema and Markdown renderer. It adds no scenario or STT schema field, CLI or
saved-file behavior, real-card adapter, raw-solver import, or three-player
selection mode.
