# Prepared two-street file workflow v1

`prepared-two-street-file-v1` is a strict, bounded two-phase JSON adapter over
the existing M14 builder and M16 orchestration contracts. It removes the need
to assemble Python dataclasses or calculate public-history and information-set
hashes by hand.

## Inspect

Start from `examples/prepared_two_street_file_v1.json` and run:

```powershell
python scripts/run_prepared_two_street_file.py inspect examples/prepared_two_street_file_v1.json
```

The successful JSON output contains `identity` and `profile_template`. Each
template row has a generated `info_set_id`, its player and observation key, and
ordered legal actions whose probabilities are `null`.

## Run

Copy the original document, change `operation` to `run`, and add exactly these
top-level fields:

- `template_identity`: the complete `identity` object returned by `inspect`;
- `hero_profile`: every Hero template row reduced to `info_set_id` and
  `actions`, with each `null` replaced by a finite probability;
- `villain_profile`: either `null`, or the same complete representation for
  every Villain information set.

Then run:

```powershell
python scripts/run_prepared_two_street_file.py run path\to\completed.json
```

Profiles must cover every applicable information set and every legal action
exactly once. M15 validates probability normalization under its published
tolerance; this workflow never fills missing probabilities or silently
normalizes, clamps, truncates, or drops entries.

## Strict input boundary

Every object rejects unknown, missing, and duplicate keys. UTF-8, input bytes,
JSON depth/value count, public-history length, prepared builder, profile, and
output limits are bounded. Limits are checked before corresponding tuple or
result materialization. A failure has `output: null`, a bounded error, and a
non-zero CLI exit; partial templates or analysis results are not returned.

Histories are arrays of typed `action`, `street_close`, and `chance` events.
The adapter computes the existing M14 public-history IDs. Canonical spec JSON
bytes and the existing M14 semantic hash bind the generated template to the
run input. Whitespace and object-key order do not change that identity.

This format covers abstract heads-up one- or two-street prepared games only.
It does not add real cards, ranges, equity generation, card removal, solver
exports, arbitrary trees, multiway or side-pot play, candidate/pipeline/GUI
integration, equilibrium or Nash claims, optimal-Hero claims, profitability
claims, or real-money advice.
