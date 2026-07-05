# STT Push/Fold Scenario Format Reference

This document describes the experimental `stt_pushfold-1` JSON format. It is a
small abstract input format for a single-table-tournament spot where everyone
folds to the small blind, SB chooses `shove` or `fold`, and BB chooses `call` or
`fold` against a shove.

The value backend is Malmuth-Harville ICM. Outputs are modelled tournament prize
EV deltas for the supplied stack vector and payout vector. They are not real
tournament predictions, not real-money advice, and not a push/fold chart.

## Scope

Implemented in `stt_pushfold-1`:

- Independent abstract SB and BB bucket ranges.
- Direct showdown probability inputs through `outcome_matrix`.
- A scalar `sb_win_probability_matrix` shorthand, interpreted as no chops.
- ICM prize EV deltas at terminal stack vectors.
- Hero seat selection with `hero_seat` set to `sb` or `bb`.
- Optional explicit baseline strategy for the non-Hero side.
- Existing baseline strategy fields as the v1 normalized baseline-solution
  profile import boundary. See
  [baseline_solution_import_format.md](baseline_solution_import_format.md).
- Shared candidate-analysis pipeline, including `T_deadline` and optional
  `T_detect`.

Public observability for `T_detect` follows the shared contract in
[public_observables_and_adaptation.md](public_observables_and_adaptation.md):
one terminal observation is the public action path plus optional
builder-supplied reveal labels. In STT v1, SB fold and BB fold-after-shove
terminals reveal no bucket, while call terminals reveal
`(sb_bucket_id, bb_bucket_id)` in the abstract all-in showdown model. These
reveal labels are internal builder annotations, not JSON input fields.

The optional physical-hand conversion available through the shared report /
pipeline API is likewise not a JSON input field. In STT outputs it can only be
read as a diagnostic conversion from repeated comparable push/fold opportunities
to a supplied physical-hand scale. It is not a tournament simulation, not a
blind-increase model, not Future-ICM / FGS, and not a prediction of real
tournament volume or opponent learning.

Out of scope for this format:

- Future-ICM, FGS, or tournament simulation.
- Real-card evaluation, 169-hand chart generation, or card removal.
- Limping, min-raising, non-all-in sizes, side pots, and partial blind posting.
- GUI editing or batch runner integration.
- Raw solver-export parsing or external solver control.
- Solver-grade, optimal, guaranteed, or profitable strategy claims.

## Top-Level Fields

Required fields:

- `format_version`: must be the string `stt_pushfold-1`.
- `scenario_id`: non-empty string.
- `stacks`: positive finite chip stacks, one per remaining player.
- `sb_index`: integer index of the SB player in `stacks`.
- `bb_index`: integer index of the BB player in `stacks`; it must differ from
  `sb_index`.
- `prizes`: non-empty, finite, non-negative, non-increasing payout vector, with
  length at most the number of players. Missing trailing prizes are zero.
- `small_blind`: positive finite number.
- `big_blind`: positive finite number, at least `small_blind`.
- `ante`: finite non-negative number.
- `hero_seat`: either `sb` or `bb`.
- `sb_range`: list of SB buckets.
- `bb_range`: list of BB buckets.
- Exactly one of `outcome_matrix` or `sb_win_probability_matrix`.
- The baseline strategy for the Hero side:
  - `baseline_sb_strategy` when `hero_seat` is `sb`.
  - `baseline_bb_strategy` when `hero_seat` is `bb`.

Optional fields:

- `description`: string, defaulting to empty.
- The non-Hero side baseline strategy. When present, it is used as an explicit
  fixed comparison profile and the build records `baseline_villain_source` as
  `explicit`. When omitted, the build derives a pure exact best response to the
  Hero baseline and records `baseline_villain_source` as
  `auto_best_response`.
- `repeated`: object with optional `horizon` and optional `discount`.
- `candidates`: object with optional `shift_amounts` and
  `max_simultaneous_info_sets`.
- `max_icm_orderings`: positive integer safety limit for ICM enumeration.
- `max_matchups`: positive integer safety limit for `len(sb_range) *
  len(bb_range)`.

## Ranges

`sb_range` and `bb_range` are independent bucket lists:

```json
[
  {"id": "strong", "weight": 0.4},
  {"id": "weak", "weight": 0.6}
]
```

Each `id` must be a non-empty string and unique within that side. Each `weight`
must be finite and positive, and weights must sum to 1. These are abstract
buckets, not real cards or parsed hand ranges. Card-removal and joint range
weights are not implemented in v1.

## Showdown Outcomes

`outcome_matrix` is keyed by `[sb_id][bb_id]` and each cell has:

- `sb_win`
- `bb_win`
- `chop`

Each probability must be finite and non-negative, and the three values must sum
to 1.

```json
"outcome_matrix": {
  "strong": {
    "call_good": {"sb_win": 0.55, "bb_win": 0.40, "chop": 0.05}
  }
}
```

`sb_win_probability_matrix` is a shorthand for cells with no chops:

```json
"sb_win_probability_matrix": {
  "strong": {
    "call_good": 0.55
  }
}
```

The scalar value is `P(SB wins)`, `P(BB wins)` is `1 - value`, and `P(chop)` is
0. This is not the same concept as the river `equity_matrix`; `equity_matrix` is
rejected in STT push/fold scenarios because ICM is non-linear in terminal stack
vectors.

## Baseline Strategies

The existing baseline strategy fields are the STT side of the v1
baseline-solution profile import boundary. They are scenario-native bucket maps,
not raw solver-export fields, and this format adds no new JSON field for
external solver metadata.

SB baseline strategy is keyed by SB bucket id and uses actions `shove` and
`fold`:

```json
"baseline_sb_strategy": {
  "strong": {"shove": 1.0, "fold": 0.0},
  "weak": {"shove": 0.3, "fold": 0.7}
}
```

BB baseline strategy is keyed by BB bucket id and uses actions `call` and
`fold`:

```json
"baseline_bb_strategy": {
  "call_good": {"call": 1.0, "fold": 0.0},
  "trash": {"call": 0.1, "fold": 0.9}
}
```

The Hero-side baseline is required. The non-Hero baseline is optional. An
explicit non-Hero baseline is a chosen comparison profile, not an equilibrium
claim and not a best-response claim unless it was produced elsewhere and the
scenario author chooses to say so outside this format.

Each bucket distribution must use finite, non-negative, non-boolean
probabilities that sum to `1` within the existing parser tolerance. A legal
action omitted inside a distribution is interpreted as probability `0`; unknown
actions or unknown bucket ids are rejected.

When the non-Hero side is omitted, the existing automatic exact best response is
used. When it is present, the build records `baseline_villain_source` as
`explicit`; otherwise it records `auto_best_response`. The run manifest's
scenario SHA-256 captures the exact input file, so this contract adds no
separate manifest field.

## Payoff Accounting

Terminals carry the existing core triple:

```text
(hero_ev, villain_ev, house_rake)
```

For STT push/fold, the third slot is not rake. It is the bystander prize EV
delta, meaning the sum of ICM prize-EV changes for players other than SB and BB.
It may be negative. The builder validates the tree with
`validate_tree(..., allow_negative_residual=True)`, while the default river
validation continues to reject negative `house_rake`.

For every terminal:

```text
hero_ev + villain_ev + bystander prize EV delta == 0
```

The JSON and CSV exporters still use the historic `house_rake` field name in
candidate rows for API compatibility. STT documentation and stdout describe the
quantity as bystander prize EV delta.

## Repeated And Candidate Config

`repeated`:

```json
"repeated": {
  "horizon": 50,
  "discount": 1.0
}
```

`horizon` is the default fixed repeated-spot horizon for the runner. It is a
sensitivity assumption that repeats the same abstract spot, not a tournament
simulation and not a model of blind increases or future stack evolution. If
`T_detect` is compared with `T_deadline`, that comparison is only a diagnostic
under the idealized threshold-observer convention described in
[public_observables_and_adaptation.md](public_observables_and_adaptation.md), not
a prediction of real opponent adaptation.

`candidates`:

```json
"candidates": {
  "shift_amounts": [0.1, 0.2],
  "max_simultaneous_info_sets": 1
}
```

Candidate generation reuses the shared Hero probability-shift library. For SB
Hero scenarios, candidates shift `shove` and `fold` probabilities at `SB:<id>`
information sets. For BB Hero scenarios, candidates shift `call` and `fold`
probabilities at `BB_vs_shove:<id>` information sets.

## Runner

Run the example:

```powershell
python scripts/run_stt_pushfold_analysis.py examples/stt_pushfold_2x2.json
```

The runner supports the same output flags as the river scenario runner:

- `--output-json`
- `--output-markdown`
- `--output-csv`
- `--strict-json`

The run manifest records the scenario SHA-256, `scenario_format_version` set to
`stt_pushfold-1`, the package version, a best-effort git commit, UTC timestamp,
and effective parameters.
