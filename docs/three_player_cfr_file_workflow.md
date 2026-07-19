# Three-player CFR-style diagnostic file workflow

This strict `three-player-cfr-file-v1` adapter turns the existing isolated M12
diagnostic into a saved-fixture workflow. It is deliberately two-phase:

1. `inspect` validates the recursive tree, complete fixed-Hero policy,
   diagnostic configuration, all M12 core limits, and caller-lowerable workflow
   limits. It returns the ordered O1/O2 action template, canonical tree content
   identity, inspection identity, and an unconfirmed human-attestation template.
2. `run` repeats the complete validation, verifies the inspection identity and
   a complete human-authored `PerfectRecallAttestation`, and then calls the
   existing public diagnostic exactly once.

Start with the bundled inspect document:

```powershell
python scripts/run_three_player_cfr_file.py examples/three_player_cfr_file_v1.json
```

The command emits one strict JSON line, returns exit `0` on workflow success or
exit `2` on a controlled failure, and leaves stderr empty. It writes no files.

## Exact top-level shape

An inspect document has exactly these keys:

```text
format_version, operation, request_id, tree, fixed_hero_policy,
config, core_limits, workflow_limits
```

`format_version` is exactly `three-player-cfr-file-v1`; `operation` is
`inspect`. A run document keeps all inspect fields, changes `operation` to
`run`, and adds exactly `inspection_identity` and `attestation`.

Unknown, missing, or duplicate keys fail closed. Input must be UTF-8 without a
BOM. Invalid UTF-8, `NaN`, `Infinity`, bool-as-number, non-finite binary64
conversion, control characters, and unsupported types are rejected.

## Recursive tree and fixed Hero

`tree` contains a bounded `description` and recursive `root`. Node shapes are:

- terminal: `type`, `node_id`, and utility `H/O1/O2/R`;
- chance: `type`, `node_id`, and ordered `{probability, child}` entries;
- fixed Hero: `type=fixed_hero`, `node_id`, `info_set`, and ordered
  `{action, child}` entries;
- opponent decisions: the same decision shape with `type=opponent_1` or
  `type=opponent_2`.

The adapter preflights JSON size/depth/value count and recursive node/depth/
branch counts before constructing M12 domain objects. The existing public
`tree_content_identity` and `validate_three_player_tree` remain authoritative
for ordered content identity, duplicate node IDs, information-set ownership and
action consistency, repeated-information-set path guards, chance mass, payoff
conservation, and the core caps.

`fixed_hero_policy.info_sets` is an ordered array of complete information-set
rows. Every row contains `info_set` and complete ordered `actions`, each with an
explicit finite `probability`. No policy row or action is silently filled,
normalized by the adapter, clamped, truncated, sampled, or ignored. The existing
M12 tolerance-bound normalization record remains visible only after a successful
run.

## Configuration and limits

`config` explicitly supplies iterations and all four non-negative tolerances.
It also fixes these v1 controls:

```text
compute_deviation_gains = true
include_oracle_rows = false
trace_checkpoint_interval = null
seed = null
```

`request_oracle` may be `true` or `false`. When true, success requires a
complete `MATCH` attachment with zero rows. When false, success requires
`NOT_REQUESTED / none` with zero rows.

`core_limits` supplies every `CfrSafetyLimits` field and may only lower the
current defaults. `workflow_limits` supplies input bytes, JSON depth/values,
tree depth/nodes/branches, policy rows/actions, attestation text, and output
records/bytes, also only at or below the documented v1 ceilings. The direct API
accepts an additional caller-lowerable outer `ThreePlayerCfrFileLimits`
envelope; the effective adapter limits are the lower values.

Node, policy, deviation-plan, oracle-plan/joint/evaluation, and conservative
output record/byte bounds are checked before the diagnostic call. Exact encoded
output is checked again before success is returned. A cap never causes a clamp,
prefix, partial table, or fallback.

## Inspect output and identity

Inspect does not run CFR, regret updates, deviation analysis, or the oracle. Its
success output includes:

- canonical `tree_content_identity`;
- node, terminal, chance, branch, information-set, and action counts;
- ordered O1/O2 information-set action templates;
- the complete fixed-Hero policy echo;
- effective config, core limits, and workflow limits;
- `inspection_identity` using
  `three-player-cfr-inspection-sha256-v1`;
- an attestation template with the tree identity filled and both confirmations,
  verifier, date, and evidence version left `null`.

The inspection SHA-256 binds the format, tree content identity, complete fixed
Hero policy, config, core limits, and effective workflow limits. It intentionally
does not bind path, clock time, platform, Python version, Git state, or request
ID.

Inspect never claims or guesses perfect recall. The caller must review the
specific tree and fill every human field explicitly.

## Run gate and success output

Run first recomputes and compares the full inspection identity. It then requires
`attestation.tree_content_identity` to equal the current tree identity,
`o1_confirmed=true`, `o2_confirmed=true`, and non-empty bounded verifier,
verification date, and evidence version strings. The adapter never generates a
perfect-recall proof and never changes a confirmation to true.

Only after identity, attestation, structure, policy, core, oracle, and output
preflights pass does run call public `run_three_player_cfr_diagnostic` once.
Success requires `DIAGNOSTIC_COMPLETE` for both component and overall status,
all requested iterations completed, deterministic full traversal, no safety-cap
stop, and the required oracle status/coverage.

The deterministic bounded projection contains complete current and average O1/
O2 behavior strategies, utilities, conservation residual, positive-regret
summaries, unilateral deviation gains, tolerances, normalization records,
warnings, and an oracle summary with counts/stability count/warnings but no
rows.

It excludes execution metadata, runtime/run identity, paths, timestamps,
platform, Python and Git details, trace, internal repr, and oracle rows.

## Bundled simultaneous 2x2 fixture

The example is the M20 fixture: O1 chooses `A/B` at `O1_root`; both branches
reach the same O2 information set `O2_root` with `L/R`, so O2 does not observe
O1's action. Hero has no decision. Its inspect identity reports 7 nodes, 4
terminals, and tree content identity
`ae7cde83467e8a7dd156971028706a9e10b9070e33d523b7e6f2d14251463c97`.

After a human fills the attestation, the two-iteration run returns:

```text
component / overall = DIAGNOSTIC_COMPLETE / DIAGNOSTIC_COMPLETE
utility H / O1 / O2 / R = -2.375 / 1.1875 / 1.1875 / 0
unilateral gains O1 / O2 = 0.3125 / 0.3125
oracle = MATCH / complete
pure plans O1 / O2 = 2 / 2
joint profiles = 4
actual profile evaluations = 13
oracle rows = 0
```

The count is independently `2*4 + 2 + 2 + 1 = 13`: two evaluators for each
pure pair, two O1 and two O2 mixed-alternative evaluations, and one direct
average-profile evaluation.

## Failure and no-partial contract

Every result has exactly `status`, `output`, and `error`. Success has output and
`error=null`; every controlled failure has `output=null` plus only bounded,
newline-free `phase`, `message`, and optional nested status. It never exposes a
partial strategy, utility, regret, oracle attachment, identity, count, or
completed-work payload. Unexpected exceptions become a generic
`INTERNAL_FAILURE` without a traceback or exception text.

Stable outer failures distinguish parse, invalid input, cap, identity,
attestation, diagnostic, non-reproducible, and internal failures. Core
non-success and requested oracle non-`MATCH` are file-workflow failures even if
some internal component work completed.

## Interpretation boundary

This adapter preserves the existing deterministic finite-iteration fixed-Hero
three-player CFR-style regret, average-profile, and unilateral-deviation
diagnostic for tiny abstract trees. Oracle `MATCH` is only the existing capped
reference cross-check.

It is not an exact best response, equilibrium or Nash computation, convergence
result, exploitability measure, certificate, proof, optimality result,
solver-grade system, strategy recommendation, profitability claim, or
real-money advice. It does not optimize Hero, test joint/coalition deviations,
auto-attest perfect recall, accept arbitrary graph references, emit trace or
oracle rows, integrate with pipeline/manifest/report/GUI, or change the M12
algorithm, public semantics, status taxonomy, or top-level exports.
