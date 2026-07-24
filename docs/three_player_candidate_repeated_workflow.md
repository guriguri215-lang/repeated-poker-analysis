# Exact Three-Player Candidate / Repeated Worked Workflow

## Purpose

`examples/three_player_candidate_repeated_workflow.py` is a deterministic,
public-submodule-only consumer of the existing M30-M32 path:

1. M31 builds and evaluates a tiny abstract one-street three-player river with
   exact chip and rake accounting.
2. M30 computes the complete bounded non-cooperative O1/O2 exact response
   correspondence after Hero is fixed.
3. M32 completely generates a declared finite Hero-shift universe, evaluates
   every candidate through a fresh M31/M30 call, and passes audited scalar
   projections to the existing M27 selection kernel.

Run it from the repository root:

```powershell
python examples/three_player_candidate_repeated_workflow.py
```

The command writes one strict, deterministic JSON line to stdout. It adds no
CLI contract, saved-file schema, pipeline, manifest, report, GUI, dependency,
CI workflow, or top-level package export.

## This is not the CFR-style diagnostic

The two public three-player examples have different contracts:

| Workflow | What completes | What it does not establish |
|---|---|---|
| `three_player_cfr_diagnostic_workflow.py` | A bounded finite-iteration CFR-style diagnostic snapshot plus an opt-in tiny pure-profile oracle | Exact response, convergence, equilibrium, Nash, or a solution |
| `three_player_candidate_repeated_workflow.py` | A bounded complete M31 scenario result, native M30 exact response correspondence for the baseline and every declared candidate, and M32 repeated-value selection rows | A full solver, Hero equilibrium, Nash/equilibrium certificate, or global optimum |

The exact response path remains bounded to the supplied abstract game. Calling
it exact does not turn it into a full poker solver or certify the fixed Hero
policy as an equilibrium.

## Worked fixture

The caller-declared fixture is deliberately tiny:

- one abstract river street;
- strategic players `H`, `O1`, and `O2`;
- non-strategic rake account `R`;
- initial pot `30`;
- initial contributions `H=10`, `O1=10`, `O2=10`;
- rake rate `0`;
- Hero information set `H_root`, with complete baseline
  `check=1, bet=0`;
- complete O1 profile:
  `O1_after_check: check=1`, `O1_after_bet: fold=1`;
- complete O2 profile:
  `O2_after_check: check=1`, `O2_after_bet: fold=1`;
- exact Hero probability shifts `1/2` and `1`, with at most one information set
  edited in a candidate;
- `search_mode=robust_all`;
- `adaptation_mode=simultaneous_o1_o2`;
- horizon `3` and discount `1.0`.

The perfect-recall attestation is human-traceable and bound to the exact tree.
Every O1/O2 information set is a singleton, neither opponent acts twice on a
path, and no opponent forgets a prior private observation or own action. The
public `create_perfect_recall_attestation` helper records both human
confirmations, the verifier text, date, evidence version, tree identity, and
machine structural-evidence identity. This is fixture-specific evidence, not a
general proof that arbitrary input trees have perfect recall.

## Hand calculation

The fixture makes every displayed value easy to check.

### Check line

All players have contributed `10`, so the pot is `30`. O1 and O2 check and the
showdown award gives the whole pot to Hero:

```text
H  = 30 - 10 = 20
O1 =      -10
O2 =      -10
R  =        0
```

### Bet/fold line

Hero's `20` bet is uncalled when both opponents fold. The uncalled amount is
returned, and Hero receives the original `30` pot. The same net vector results:

```text
H/O1/O2/R = 20/-10/-10/0
```

Therefore the baseline, the half-shift candidate, and the full-shift candidate
all have:

```text
pre-adaptation H             = 20
native post-response H worst = 20
```

For horizon `N=3` and discount `1`, baseline total value is `3 * 20 = 60`.
Every candidate also has value `60` for each simultaneous adaptation
opportunity `m=1,2,3,4`, so every total delta is `0`. Both candidates form the
full primary tie set in every row, but the strict positive-uplift rule is not
met. Every row consequently reports:

```text
status = NO_BENEFICIAL_COMMITMENT
selected_candidate_id = null
```

The smaller `1/2` shift is the deterministic display candidate only. Display
tie-breaking does not turn a no-benefit row into a selection.

## Hero safety source

The only Hero safety scalar is each candidate's native, complete M31 exact
response value at:

```text
m31_scenario_response.response.hero_worst
```

The workflow does not substitute any of the following:

- current CFR state or a CFR snapshot;
- the first response witness;
- the pure-profile unilateral-stability subset;
- the separate Hero-min joint-plan / coalition stress diagnostic;
- `hero_best`.

`hero_best` and full response multiplicity remain available in the native M31
object for audit, but the public example prints a small allowlisted projection
rather than reformatting native responses as strategy advice. O1 and O2 remain
separate non-cooperative players. M27's `O1+O2` accounting transport is not a
coalition or transferable-utility claim.

## Output allowlist

The one-line JSON contains only:

- outer success status;
- explicit fixture inputs and attestation trace;
- candidate count;
- baseline exact initial-profile values;
- each candidate ID, exact edit, pre-adaptation Hero value, and native
  `hero_worst` post-response Hero value;
- all four timing rows, including status, delta, selected/null ID, full primary
  ties, and display candidate;
- the safety-source path and explicit false substitution flags;
- bounded scenario, tree, baseline policy, initial profile, response,
  candidate-universe, and M32 run identities.

It intentionally omits full support cells, witness strategies, pure subsets,
and coalition-stress details from stdout. Those remain losslessly nested in the
native successful result.

## Current v1 boundary

The worked workflow demonstrates only:

```text
search_mode = robust_all
adaptation_mode = simultaneous_o1_o2
Hero universe = caller-declared bounded finite shifts
```

`baseline_targeted`, `hybrid`, individual `(m1,m2)` adaptation timing, large or
continuous search, and approximate/CFR fallback are unsupported. The result is
a stationary one-shot repeated-value sensitivity analysis over an assumed
simultaneous switch opportunity. It does not predict O1/O2 learning,
adaptation likelihood, or an actual switch time.

## Interpretation boundary

This example is:

- abstract rather than real-card three-player poker;
- synthetic rather than a real-world range dataset or population calibration;
- bounded rather than a large-scale solver;
- conditional on the complete supplied profiles, finite tree, declared search
  universe, horizon, and discount.

It is not a full solver, Nash/equilibrium certificate, proof, global or
continuous optimum, real-card three-player evaluation, external solver
certification, profitability guarantee, deployment recommendation, or
real-money advice.
