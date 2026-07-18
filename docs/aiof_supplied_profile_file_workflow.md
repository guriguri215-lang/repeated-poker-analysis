# AIoF supplied-profile file workflow v1

`aiof-supplied-profile-file-v1` is a strict, bounded two-phase JSON adapter over
M13's existing public range preparation and supplied-profile ChipEV API. It is
for saved fixtures and deterministic automation when callers need the exact
post-removal combo support before they can supply shove/call probabilities.

Start with the checked-in `inspect` fixture:

```powershell
python scripts/run_aiof_supplied_profile_file.py examples/aiof_supplied_profile_file_v1.json
```

Success prints one strict JSON line and exits 0. A controlled failure prints one
wrapper JSON line with `output: null` and exits 2. Controlled results use no
stderr and never print a traceback.

## Why the workflow has two phases

A class such as `AKs` expands before public-dead-card removal. The surviving SB
and BB supports are then conditioned once on ordered card-disjoint pairs. The
complete combo support therefore cannot safely be inferred from the original
class labels alone.

1. `inspect` validates the full spec, calls the existing public
   `prepare_compatible_ranges`, and returns the prepared identity, canonical
   SB/BB supports, and a complete template whose probabilities are `null`. It
   runs no equity, ChipEV, or best-response analysis and makes no strategy claim.
2. To make a `run` document, retain the exact spec, change `operation` to `run`,
   copy `output.identity` to top-level `template_identity`, copy
   `output.profile_template` to top-level `profile`, and replace every `null`
   with a finite probability in `[0, 1]`. Running that document recomputes the
   identity and support before calling existing `analyze_pushfold` with exact
   exhaustive equity and both fixed-opponent best-response seats.

The template identity binds the format, canonical range semantics, prepared
range identity, support, game, exact controls, and all phase-1 limits. A changed
spec cannot reuse an old profile template.

## Strict document contract

Both operations contain exactly these base fields:

- `format_version`: `aiof-supplied-profile-file-v1`;
- `operation`: `inspect` or `run`;
- `request_id`: a bounded caller label, not a computation identity;
- ordered `sb_range` / `bb_range` arrays of
  `{label, weight, weight_basis}` using explicit class-or-combo syntax;
- `dead_cards`: strict canonical two-character cards;
- fee-zero heads-up `game` values;
- `analysis`: exact exhaustive, zero trace, both response seats, finite
  non-negative deviation tolerance, and null seed/samples;
- `limits`: all nine caller-lowerable `AiofLimits` fields.

`run` additionally requires exact `template_identity` and `profile` objects.
The profile has complete `sb_shove` and `bb_call` row arrays; every row contains
one canonical exact combo and one finite probability. Missing, extra, duplicate,
or noncanonical support is rejected. The adapter never fills missing rows,
normalizes probabilities, clamps values, truncates support, changes algorithms,
or silently falls back.

Every object rejects missing, unknown, and duplicate keys. Input must be UTF-8
without a BOM. JSON `NaN` and infinities, boolean-as-number values, non-finite
binary64 values, unsupported cards/ranges/games, Monte Carlo controls, and
noncanonical response-seat order are rejected.

## Caps and no-partial behavior

Adapter ceilings bound input bytes, JSON depth/value count, range entries,
dead-card items, profile rows, output records, and output bytes. Document limits
also bound exact combo expansion, compatible pairs, exact board evaluations,
cache entries, and trace/sampling resources. Caller-supplied caps can only lower
hard ceilings. Input bytes and container counts are checked before domain
materialization; the core checks compatible-pair and board-evaluation work before
success output is built.

All failures have `output: null` and only a bounded phase, newline-free message,
and optional exact nested `AiofStatus`. Failures never expose a template,
profile, identity, outcome, value, best-response row, completed count, or other
partial payload.

## Run success output

A successful run returns a deterministic bounded projection containing:

- exact algorithm and ChipEV accounting IDs;
- template, prepared-range, and core input identities;
- compatible-pair and exact board-evaluation counts;
- exact outcome counts and finite win/loss/tie probabilities;
- the complete canonical supplied SB shove and BB call rows;
- SB/BB profile values and their conservation sum;
- null exact-mode sampling statistics;
- both fixed-opponent response summaries and every canonical per-own-combo row,
  including reach, action values, best-action correspondence, supplied action
  probability, and raw gain.

Runtime/run identity, absolute path, platform, timestamp, timing, trace, and full
internal object representations are excluded. Equivalent content therefore has
stable strict JSON bytes on supported Python versions.

## Checked fixture and interpretation boundary

The checked fixture uses `AsAh` versus `KsKh`, leaves the known board
`2c 3d 4h 5s 7c` live, and marks the other 43 cards dead. Inspect returns one
combo per seat. Filling SB shove with `1` and BB call with `0` yields SB/BB
profile values `+1/-1`, conservation zero, and zero raw gain for both fixed-
opponent responses with one board evaluation.

This is supplied-profile commitment analysis, not endogenous solving. A
fixed-opponent best response holds the other supplied profile fixed; it is not a
joint equilibrium, Nash certificate, optimal-Hero result, profitability claim,
range chart, raw solver import, real-world range dataset, large-scale solve, or
real-money strategy recommendation. Monte Carlo, heuristic diagnostics,
rational-lift generation, prepared/repeated pipeline integration, manifests,
reports, GUI, and top-level package exports are outside v1.
