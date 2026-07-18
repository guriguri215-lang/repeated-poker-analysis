# Guarded three-player CFR-style diagnostic workflow

This guide runs one deliberately tiny fixed-Hero fixture through the existing
isolated `repeated_poker.three_player_cfr` submodule. It demonstrates a
deterministic finite-iteration CFR-style regret diagnostic and an explicitly
requested, capped pure-profile reference attachment without changing the M12
algorithm, API, ordering, normalization, identity, caps, or error taxonomy.

Run it from the repository root:

```powershell
python examples/three_player_cfr_diagnostic_workflow.py
```

The script prints one strict JSON-safe line, writes no files, and uses no
network or external data. It omits full strategies, oracle rows, the full
result object, content hashes, paths, credentials, platform details, the full
Python version, and Git commit metadata.

## Fixed-Hero simultaneous 2x2 fixture

Opponent 1 acts at `O1_root` with ordered actions `A/B`. Each branch reaches an
Opponent 2 node with ordered actions `L/R`; both nodes share information set
`O2_root`. O2 therefore does not observe O1's action. This extensive-form tree
is a 2x2 simultaneous representation, not a sequential response by O2.

The terminal utility vectors `(H,O1,O2,R)` are:

| O1 / O2 | L | R |
|---|---|---|
| A | `(-2,1,1,0)` | `(0,0,0,0)` |
| B | `(0,0,0,0)` | `(-4,2,2,0)` |

Hero has no decision, so its complete fixed policy is `BehaviorStrategy({})`.
O1 and O2 remain separate strategic actors. `R` is a non-strategic accounting
residual and chooses no action.

## Manual attestation and content identity

The fixture author directly constructs `PerfectRecallAttestation` and binds it
to the full ordered tree content identity and version
`m20-three-player-public-example-v1`. Both opponent confirmations, verifier,
date, and evidence version are literal human-authored evidence.

The booleans and helper-produced content hash do not automatically prove
perfect recall or general game-tree correctness. Any description, node,
information partition, action order, child, probability, or utility change
changes the identity and invalidates the old attestation. A missing,
unconfirmed, unsupported, or mismatched attestation fails closed as
`UNSUPPORTED_MODEL`.

## Finite-iteration snapshot

The example fixes:

```text
iterations = 2
request_oracle = true
include_oracle_rows = false
seed = none
trace = off
deviation gains = enabled
```

Zero cumulative regret, uniform initialization, deterministic full traversal,
iteration-start strategy construction, tree action order, and existing
reach-weighted behavioural averaging are unchanged. The expected projection is:

```text
component / overall status = DIAGNOSTIC_COMPLETE / DIAGNOSTIC_COMPLETE
requested / completed iterations = 2 / 2
expected utility = H -2.375, O1 1.1875, O2 1.1875, R 0
average-profile unilateral gains = O1 0.3125, O2 0.3125
```

`DIAGNOSTIC_COMPLETE` means the requested finite computation completed. It is
not a claim that the returned average profile is stable: both reported
unilateral gains remain positive. Two iterations are a deterministic
regression snapshot, not a convergence, solution, Nash, or optimality result.

## Complete capped oracle attachment

O1 and O2 each have two full pure plans, so the complete Cartesian table has
four rows. The independent count is:

```text
2 * 4 pure-pair evaluations
+ 2 O1 mixed-alternative evaluations
+ 2 O2 mixed-alternative evaluations
+ 1 direct average-profile evaluation
= 13 profile evaluations
```

With rows excluded from output, predicted and actual output rows are zero. The
two diagonal pure pairs are unilateral-stability rows, and the attachment warns
about multiple such rows and tied utility vectors.

Oracle `MATCH` means only that the independent direct evaluator, pure-plan
mixture calculation, and O1/O2 unilateral-gain calculations agree within the
declared tolerance for this tiny fixed-Hero fixture. The two stable table rows
do not establish joint or coalition stability, select a preferred row, certify
the CFR snapshot, prove convergence, or describe an equilibrium set.

## Caps and no-partial failure

The existing per-player plan, joint-profile, evaluation, and optional output-row
caps are checked before plan lists, Cartesian tables, or output rows are
materialized. Setting `max_oracle_joint_profiles=3` makes this four-row request
`ORACLE_UNAVAILABLE_CAP` with `coverage=none` and `rows=[]`.

Callers must not clamp, truncate, sample, skip, xfail, or fall back to a prefix
and label it complete. A requested oracle that is unavailable, mismatched, or
tolerance-indeterminate makes the overall request non-success even if the CFR
component finished. The example emits stdout only after component, overall,
oracle, coverage, and empty-row checks all pass.

`DiagnosticContractError` distinguishes `INVALID_INPUT`, `UNSUPPORTED_MODEL`,
`CAP_EXCEEDED`, and `NUMERIC_FAILURE`. Oracle statuses additionally distinguish
`ORACLE_UNAVAILABLE_CAP`, `ORACLE_MISMATCH`, and
`INDETERMINATE_TOLERANCE`. These paths, an unexpected status, or any unexpected
exception return nonzero, leave stdout empty, and write only a short stderr
message. Non-finite values are not clamped, converted to null, or replaced by a
finite-looking result.

## Interpretation boundary

This is a deterministic finite-iteration CFR-style regret, average-profile,
and unilateral-deviation diagnostic for a tiny abstract general-sum tree with a
fixed Hero and separate O1/O2 actors. The oracle is a subordinate, explicitly
requested, completely enumerated reference attachment within strict caps.

It is not an exact best response, full three-player poker solver, general-sum
equilibrium or Nash computation, certificate, proof, convergence guarantee,
optimality result, exploitability measure, or solver-grade output. Unilateral
gain varies one opponent at a time and does not test joint or coalition
deviation. The workflow adds no top-level export, JSON/schema/format version,
CLI operation, pipeline, manifest, report, GUI, dependency, strategy
recommendation, profitability claim, or real-money advice.
