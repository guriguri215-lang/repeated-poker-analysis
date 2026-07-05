# Baseline Solution Import Format

This document defines the v1 contract for using an externally prepared baseline
profile with this project. It is a documentation contract over existing
scenario JSON fields. It does not add a parser, a solver adapter, a command-line
flag, a manifest field, or a new scenario schema field.

## 1. Purpose

A baseline solution is a reference mixed-strategy profile for the current
abstract scenario. The analysis can compare candidate Hero commitments against
that reference profile, but the profile is only meaningful inside the supplied
game abstraction.

The purpose of this document is to name the import boundary clearly:
external-source results may be converted outside this project into the existing
scenario-native baseline strategy fields. The project then reads those existing
fields as ordinary scenario input.

## 2. V1 import boundary

In v1, the normalized baseline-solution import format is the existing
scenario-native mixed strategy map format:

- an information-set id or bucket id;
- an action label that is legal at that information set or bucket;
- a finite, non-negative probability;
- probabilities that sum to `1` within the existing validation tolerance;
- omitted legal actions interpreted as probability `0`, matching the current
  parsers.

Unknown actions, unknown information sets, unknown buckets, non-finite
probabilities, negative probabilities, boolean probabilities, and distributions
that do not sum to `1` are rejected by the existing scenario loaders.

Raw external solver output is outside the project boundary. A user or future
adapter may transform an external result into these scenario-native fields, but
this project does not parse, control, validate, or certify raw solver exports in
v1.

The import is profile-only. It does not import or certify game trees, ranges,
board textures, real-card equities, card removal, rake modelling, ICM modelling,
solver metadata, exploitability numbers, or any external abstraction.

## 3. What a baseline solution means here

Here, a baseline solution means a chosen reference mixed-strategy profile over
the exact abstract tree built from the scenario. It is not proof of equilibrium,
not proof that the external abstraction matches the scenario, and not a claim
about play outside the model.

For Hero, the baseline profile is the locked reference strategy from which
candidate shifts are generated. For the non-Hero side, an explicit baseline
profile is a fixed comparison profile. If the non-Hero profile is omitted, the
project keeps its existing automatic exact best-response behavior.

## 4. Scenario-native profile format

The common shape is a JSON object whose leaves are action distributions:

```json
{
  "some_info_set_or_bucket": {
    "action_a": 0.75,
    "action_b": 0.25
  }
}
```

The key names are not free-form solver labels. They must match the names used by
the project scenario format:

- river single-hand and explicit river Villain profiles use built information
  set ids;
- river range Hero profiles use the relevant Hero bucket id or per-bucket
  decision point;
- STT push/fold profiles use SB or BB bucket ids.

The scenario author is responsible for matching the external abstraction to the
project scenario: actions, bucket ids, ranges, payoff model, rake or ICM
assumptions, and information-set meanings.

## 5. River profile fields

River scenarios use the existing `scenario_format_reference.md` fields:

- single-hand mode uses top-level `baseline_hero_strategy`;
- Hero-range-only mode and simple matrix modes use each Hero bucket's
  `baseline_strategy`;
- river betting-tree mode uses each Hero bucket's `baseline_strategies`;
- optional top-level `baseline_villain_strategy` pins a chosen comparison
  Villain profile over the built Villain information sets.

When `baseline_villain_strategy` is omitted, the baseline Villain falls back to
the existing automatic exact best response to the baseline Hero profile. When it
is present, every built Villain information set must be assigned, and the
profile is used as an explicit fixed comparison profile.

The build metadata reports `baseline_villain_source` as `explicit` or
`auto_best_response`. The run manifest already records the scenario SHA-256,
which captures the exact imported profile when the scenario was loaded from a
file. No new manifest field is added for this contract.

See [scenario_format_reference.md](scenario_format_reference.md) for the field
details, information-set names, and validation errors.

## 6. STT push/fold profile fields

STT push/fold scenarios use the existing `stt_pushfold-1` fields:

- `baseline_sb_strategy` is keyed by SB bucket id and uses `shove` / `fold`;
- `baseline_bb_strategy` is keyed by BB bucket id and uses `call` / `fold`;
- the Hero-seat side is required;
- the non-Hero side is optional.

When the non-Hero side is omitted, the builder derives the existing automatic
exact best response to the Hero baseline. When the non-Hero side is present, it
is an explicit fixed comparison profile. The build metadata records
`baseline_villain_source` as `explicit` or `auto_best_response`, and the
scenario SHA-256 in the run manifest captures the exact profile from the input
file. No new manifest field is added.

See [stt_pushfold_format_reference.md](stt_pushfold_format_reference.md) for the
field details, bucket names, and ICM-specific validation rules.

## 7. Provenance and reproducibility

The scenario file is the reproducible unit. If a baseline profile came from an
external source, keep the converted profile in the scenario JSON and keep any
external provenance notes outside the project-specific manifest unless a future
schema explicitly adds such metadata.

The project records descriptive build metadata and a scenario SHA-256 in run
outputs. These records show which scenario file was analyzed; they do not
certify that an external source, external abstraction, or manual conversion was
correct.

## 8. What may and may not be inferred

An imported baseline profile may be used as a reference profile inside the
supplied abstract scenario. Candidate comparisons, fixed-profile values,
`T_deadline`, and `T_detect` diagnostics are then conditional on that scenario
and on the profile maps that were supplied.

An imported baseline profile does not imply:

- equilibrium;
- an optimal strategy;
- solver-grade validation;
- a Nash chart;
- a guarantee;
- profitable play in real games;
- direct real-money advice.

The same caution applies to explicit non-Hero profiles. They are chosen
comparison baselines, not claims that the non-Hero side is playing a best
response unless the scenario author separately justifies that outside this
format.

## 9. Non-goals

The v1 baseline-solution import contract does not add:

- raw solver-export parsing;
- new scenario JSON fields;
- new CLI flags;
- new run-manifest fields;
- solver adapters or external-process calls;
- real-card range parsing;
- card removal;
- chart generation;
- large-scale range solving;
- new detection, response, STT, ICM, or candidate-generation behavior;
- GUI support for importing external solver files.
