# Unified certified-global public workflow

M40 provides one discriminated in-memory workflow over the stable M37, M38,
and M39 certified-global consumers. It is an orchestration contract, not a
new optimizer or poker model.

## Variants and exactly-one-call mapping

| M40 variant | Required nested request | Exactly one public analyzer |
| --- | --- | --- |
| `real_card_aiof_preflop_m37` | `AiofPreflopCertifiedGlobalRequest` | `analyze_aiof_preflop_certified_global` |
| `known_board_real_card_hu_river_m38` | `KnownBoardRealCardHuCertifiedGlobalRequest` | `analyze_known_board_real_card_hu_certified_global` |
| `abstract_three_player_river_m39` | `AbstractThreePlayerCertifiedGlobalRequest` | `analyze_abstract_three_player_certified_global` |
| `known_board_real_card_three_player_river_m39` | `KnownBoardRealCardThreePlayerCertifiedGlobalRequest` | `analyze_known_board_real_card_three_player_certified_global` |

Call
`analyze_unified_certified_global(UnifiedCertifiedGlobalRequest(...))`.
The variant and exact nested-request type must match. The workflow validates
that pair and all nested public limit-object shapes before dispatch. It does
not call a second consumer, a candidate workflow, or a fallback.

## Lossless result and failure schema

On success, the outer status is exactly the native certified status. The
payload holds the complete native result object and its unchanged `to_dict`
projection. That preserves the native M36 result, certificate, work counters,
selected policy, all preparation and analysis identities, and the
consumer-specific complete response evidence. The wrapper does not summarize
or reconstruct any of these records.

The outer result fields are:

- `status`
- `payload`
- `error`
- `failure_evidence`
- `optimizer_work_counters`
- `oracle_work_counters`
- `partial_result`

Every failure has `payload: null` and `partial_result: false`. A native
consumer failure remains the same status and retains the complete native
failure result, exact nested error phase/cause, and completed counters in
`failure_evidence`. An outer pin or output-cap failure after a certified
native run retains the nested status, result identity, and completed counters
but never exposes the successful native payload as a partial result. Nested
failure is never promoted to success.

`exact_unified_certified_global_json` emits sorted-key, compact, strict UTF-8,
one-line JSON.

## Identity and stale-input contract

All identities are lowercase SHA-256 hex. Pins accept either raw hex or the
equivalent `sha256:` prefix.

- `contract_identity` binds the M40 contract and claim.
- `schema_identity` binds the canonical result schema.
- `variant_identity` binds the variant and required nested request type.
- `nested_request_identity` reuses the consumer's semantic canonical request
  identity. M37 has no separate public request identity, so M40 derives a
  domain-separated request identity from M37's canonical analysis identity.
- `workflow_identity` binds contract, schema, variant, nested semantic
  request, workflow version, and outer caps.
- `nested_analysis_identity` is the exact consumer analysis identity.
- `nested_result_identity` is SHA-256 of the native consumer's exact public
  JSON bytes.
- `analysis_identity` binds the workflow, nested analysis, nested bytes, and
  native certified status.

Pin syntax and the contract/schema/variant pins are checked before nested
work. Semantic nested request, workflow, and analysis pins are checked once
the existing consumer has produced its canonical identities; computing them
earlier would duplicate protected consumer semantics. A mismatch is
`STALE_INPUT`, never a clamp or warning. Semantic mapping permutation therefore
keeps identical public bytes, while a meaningful nested input or variant
change changes the relevant identities and bytes.

## Caps and no materialization

`UnifiedCertifiedGlobalLimits` caps records and UTF-8 bytes for the complete
final M40 wrapper. Both caps are caller-lowerable and bounded by immutable
ceilings. Before dispatch, M40 measures a conservative lower-bound success
shape. A very-low cap therefore stops before the selected analyzer, native
aggregate `to_dict`, or native serializer is called.

After a native certified result exists, M40 traverses a lazy projection of
the final public wrapper. It stops at the first exceeded record or byte. It
does not construct a complete outer success dictionary or full outer encoded
byte string during that cap check. The exact `N`/`N-1` boundary is covered by
reviewer-independent counting tests, and an outer failure retains completed
nested counters without exposing partial output.

Nested M36/M37/M38/M39 caps remain authoritative. There is no silent clamp,
sampling, truncation, or partial result.

## Preserved semantics and claim boundary

M40 calls existing public analyzers and does not reproduce their scenario,
objective, response, whole-cell bound, certificate, card, or accounting
logic. Therefore:

- M37 and M38 retain one complete opponent response.
- M39 retains the complete M30 two-opponent non-cooperative Hero-worst
  response correspondence; it is not treated as a one-opponent response.
- M37, M38, and both M39 surfaces retain their full scenario-derived legal
  Hero behavior-policy domains and consumer-specific sound whole-cell bounds.
- Real-card variants retain card/blocker conditioning, exact showdown,
  rake/cap, uncalled excess, and conservation semantics.
- M35/M37/M38/M39 request limits and M36 certificate caps remain in force.

The only claim is the selected variant's identified bounded scalar objective
specified-tolerance certified global maximum. M40 does not claim
cross-variant comparison, equilibrium, profitability, solver-grade scale, or
strategy advice.

M32 finite candidate selection remains a valid R6 workflow. It is not an R8
oracle or certificate. M40 accepts no candidate list, grid, vertices, local
box, warm-start, sampling, or fallback controls.

## R1–R8 cross-module acceptance

The focused crosswalk is
`test_03_r1_to_r8_cross_module_independent_acceptance_projection` in
`tests/test_unified_certified_global_workflow.py`.

| Requirement | M40 gate and retained evidence |
| --- | --- |
| R1 | M37 prepared real-card ranges/blockers and M38/M39 conditioned support are present in native payloads. |
| R2 | The M37 real-card AIoF full-domain certificate is dispatched and retained. |
| R3 | M38 known-board HU river/rake history and certificate are retained. |
| R4 | M39 abstract M31/M30 response history and certificate are retained. |
| R5 | Each native payload keeps supplied baseline, same fixed profile, fresh response, and repeated objective separation. |
| R6 | M32 remains available independently and no candidate/grid field exists in the M40 request. |
| R7 | M39 real-card preparation retains M35 ordered triples, blockers, showdown, rake/cap, uncalled excess, and conservation. |
| R8 | All four paths retain automatic full legal domains, native whole-cell bound contracts, certificates, caps, and no-partial failures. |

The cross-module gate is additive to the independent analytic/exhaustive
oracle suites:

- `tests/test_aiof_preflop_certified_global.py`
- `tests/test_known_board_real_card_hu_certified_global.py`
- `tests/test_three_player_certified_global.py`
- `tests/test_certified_global_optimizer.py`

Those test-owned oracles cover blocker changes, strict interior optima,
singleton and whole-cell bounds, complete response ties and Hero-worst
selection, positive rake, no-benefit, unsupported/no-certificate taxonomy,
and exact output caps. M40's tests compare each direct public analyzer result
to the nested unified result without using the production wrapper as an
oracle.

## Example and lifecycle boundary

Run:

```text
python examples/unified_certified_global_workflow.py
```

The example emits one LF-terminated deterministic M37-through-M40 result.
Two fresh-process byte identity and SHA-256 equality are tested.

M40 is merged on `main` and included in the v0.2.0 release line. Its
independent review, human merge, post-merge verification, and release evidence
live in repository history and the release notes. That lifecycle evidence does
not broaden the claim beyond the selected bounded scalar objective.
