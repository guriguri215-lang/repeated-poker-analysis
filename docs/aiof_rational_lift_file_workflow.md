# AIoF exact rational-lift file workflow v1

`aiof-rational-lift-file-v1` is a strict, bounded run-only JSON adapter over
M13's existing public exact rational-lift strategy API. It is for saved fixtures,
automation, and deterministic machine comparison without constructing Python
dataclasses by hand.

Run the checked-in tiny fixture from the repository root:

```powershell
python scripts/run_aiof_rational_lift_file.py examples/aiof_rational_lift_file_v1.json
```

Success prints one strict JSON line and exits 0. A controlled failure prints one
wrapper JSON line with `output: null` and exits 2. Ordinary controlled results do
not use stderr and do not print tracebacks.

## Strict input contract

Every object rejects missing, unknown, and duplicate keys. Input must be UTF-8
without a BOM. JSON `NaN` and infinities, boolean-as-number values, non-finite
binary64 numbers, and noncanonical rational strings are rejected.

The top-level fields are exactly:

- `format_version`: `aiof-rational-lift-file-v1`;
- `request_id`: a bounded caller label echoed in the result, not a solver identity;
- `sb_range` and `bb_range`: ordered `{label, weight, weight_basis}` arrays using
  the existing explicit class-or-combo grammar and weight-basis semantics;
- `dead_cards`: canonical two-character public cards;
- `game`: stacks, blinds, ante, fee, third-party dead money, and side-pot flag;
- `strategy`: fixed exact algorithm selection and rational claim controls;
- `limits`: all 14 caller-lowerable `AiofStrategyLimits` fields.

Version 1 requires the compact rational LP algorithm, exact exhaustive equity,
`seed: null`, `samples: null`, and both optional diagnostic flags false. It also
requires fee and third-party dead money to be zero and side pot to be false.
`claim_epsilon` and `display_tie_tolerance` use reduced non-negative strings such
as `"0"` or `"1/100"`; decimal spellings such as `"0.01"` are not canonical.

Adapter limits bound input bytes, JSON depth/value count, range entry counts,
dead-card items, and output records/bytes. They may only be lowered by direct API
callers. Solver limits are supplied in the document and may not exceed the M13
hard ceilings. Input bytes are checked before decode/parse, and container counts
are checked before domain tuple construction. Caps are refusal boundaries: the
workflow never clamps, truncates, changes algorithms, or falls back.

## Success result

The wrapper has the exclusive shape `status`, `output`, and `error`. On success,
`status` is `SUCCESS` and `error` is null. The output contains:

- format/request labels and exact algorithm/game/verifier IDs;
- prepared-range, payoff, semantic, and input identities;
- payoff-cell and exact-board-evaluation counts;
- the complete canonical SB shove and BB call profile as rational strings;
- profile and verification identities;
- the fully qualified claim kind, exact objective/verification fields, and exact
  unilateral-gain/value enclosure;
- null oracle and phase-1 diagnostic fields, because v1 does not expose them.

Runtime identity, run identity, platform, path, timestamp, timing, and full LP
tableaux/traces are intentionally excluded. Equivalent content therefore emits
the same JSON bytes on supported Python versions.

## Failure and no-partial rule

The adapter distinguishes parse, invalid-input, cap, strategy,
non-reproducibility, and internal failures. Strategy failures retain the exact
M13 `AiofStrategyStatus` in `nested_status`. Failure output contains only a
bounded phase, newline-free message, and optional nested status. It never exposes
a profile, claim, value, identity, completed payoff-cell/pivot count, trace, or
partial witness.

## Checked fixture and interpretation boundary

The example uses `AsAh` versus `KsKh`, leaves the known board
`2c 3d 4h 5s 7c` live, and marks the other 43 cards dead. With stacks 10/10 and
blinds 0.5/1, it has one compatible payoff cell and one exact board evaluation.
The bounded result has SB shove probability 1, BB call probability 0, profile
value 1, both unilateral gains 0, and the scoped claim
`aiof-rational-lift-game-v1:EXACT_NASH`.

That claim applies only to M13's finite rational-lift game. This workflow is not
a range chart, external-game certificate, large-scale solver, optimal-Hero or
profitability claim, real-money recommendation, Monte Carlo workflow, supplied-
profile analysis, heuristic diagnostic, prepared two-street integration,
candidate/repeated pipeline, manifest/report format, or GUI feature.
