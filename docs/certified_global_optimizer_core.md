# Certified continuous/global Hero commitment optimizer core

This document specifies the M36 in-memory core in
`repeated_poker.certified_global_optimizer`. It solves one bounded contract:

> Over every legal Hero behavior policy in the scenario-derived product of
> simplexes, certify the largest baseline-relative total repeated Hero-EV
> uplift supplied by a conforming scalar response oracle.

The core remains consumer-agnostic. Separate M37-M39 adapters connect the M28
real-card preflop, M29 known-board heads-up river, and M30-M35
abstract/real-card three-player consumers; M40 provides an exactly-one-consumer
wrapper. M36 does not change the existing finite M27/M32 candidate workflows.

## Full continuous domain

`HeroBehaviorScenario` supplies a stable scenario identity and every Hero
information set with all of its legal actions. For an information set \(I\)
with legal actions \(A(I)\), the core constructs

```text
p(I,a) >= 0 for every a in A(I)
sum(a in A(I)) p(I,a) = 1 exactly
```

and searches the Cartesian product of these simplexes. Probabilities are
canonical exact rationals at evaluated points and at cell boundaries.
Continuous means the mathematical domain contains every real point of each
simplex; exact-rational branch points are used to bound that continuum, not to
redefine it as a finite grid.

The API has no `shift_amounts`, candidate library, grid resolution, local
probability bounds, or warm-start neighbourhood. `baseline_policy` is required
because the objective is uplift from that policy. The baseline is also a
feasible initial incumbent, but the root cell remains `[0,1]` for every action
intersected with the exact simplex constraints.

Semantic ordering is canonical: information-set and action input permutations
produce the same domain, policy, run, and output identities.

## Scalar response-oracle boundary

A conforming `CertifiedScalarResponseOracle` has four fixed metadata fields:

- `response_oracle_identity`: lowercase SHA-256 of the response implementation
  and semantic inputs;
- `objective_identity`: lowercase SHA-256 binding the baseline, horizon,
  discount, accounting convention, and other consumer objective inputs;
- `response_semantics`:
  `complete-response-correspondence-hero-worst-only-v1`;
- `bound_contract_version`:
  `whole-cell-scalar-global-upper-bound-v1`.

It implements two methods.

### Exact point evaluation

`evaluate(policy)` returns `ScalarOracleEvaluation`. The record includes:

- the exact policy, response-oracle, and objective identities;
- baseline and candidate total repeated Hero EV;
- exact `uplift = candidate - baseline`;
- an identity for the complete response correspondence;
- complete-response record count and Hero-worst witness count.

The baseline point must have exactly zero uplift. Response ties are not reduced
to a first witness: the consumer binds its complete correspondence and reports
the retained Hero-worst witness multiplicity. The candidate repeated value must
use correspondence-wide `hero_worst` wherever post-response safety enters. A
pure subset, coalition/joint-plan stress result, current CFR state,
`hero_best`, or opponent-EV minimization is not the M36 scalar objective.

### Whole-cell global upper bound

`upper_bound(cell)` returns `ScalarOracleBound`. `upper_bound` must bound the
same uplift for **every** legal behavior policy in the closed cell, not only
sampled points, local stationary points, vertices tested heuristically, or a
finite grid. The record binds:

- cell, response-oracle, and objective identities;
- exact upper-bound value;
- bound contract version and `valid_for_entire_cell=True`;
- deterministic bound identity;
- optionally, one feasible candidate policy.

The optional candidate is only lower-bound guidance. The core checks that it
is inside the cell and evaluates it through the point oracle before it may
become an incumbent. It does not shrink or otherwise change the cell.

The core can validate metadata, exact arithmetic, identities, cell membership,
and that returned bounds dominate the point values it evaluates. It cannot
prove an arbitrary poker consumer's mathematical inequality from callback
code. Soundness is therefore explicitly conditional on the identified oracle
honouring the whole-cell contract. A consumer unable to construct a safe bound
must raise `UnsupportedScalarOracleDomain`; returning a heuristic bound and
calling the result certified is a contract violation.

## Branch-and-bound and soundness

Each search cell stores a lower and upper interval for every action, intersected
with the action's information-set simplex. A cell is feasible exactly when

```text
sum(lower bounds) <= 1 <= sum(upper bounds)
```

for every information set.

The core selects the widest *effective* action interval and splits it at its
exact midpoint. The left child adds `p <= midpoint`; the right child adds
`p >= midpoint`. Both are closed and their union is the parent. Consequently:

1. the root is the full scenario-derived domain;
2. active children replace their parent without losing a legal policy;
3. pruned cells have an upper bound no larger than the incumbent;
4. active cells plus pruned proof cover the complete domain.

Let `L` be the best exact evaluated uplift and let `U(C)` be the conforming
oracle bound for active cell `C`. The certificate uses

```text
global upper = max(L, max(active C) U(C))
absolute gap = global upper - L
relative gap = absolute gap / max(1, abs(L))
```

`L` is a valid global lower bound because its policy is feasible.
Whole-cell oracle validity and the cover invariant make the displayed upper
bound global. Cells are pruned only when `U(C) <= L`.

The status is:

- `CERTIFIED_GLOBAL` when the exact gap is zero;
- `CERTIFIED_EPSILON_GLOBAL` when the gap is nonzero but no larger than the
  requested exact absolute tolerance, or meets a positive requested exact
  relative tolerance.

No other status is success. With zero tolerances, only an exact zero-gap proof
succeeds.

## No-benefit and ties

The supplied baseline is always evaluated, so the incumbent lower bound starts
at zero. If certification completes without a positive evaluated uplift,
`no_beneficial_commitment=True` and all selected-commitment fields are `null`.
An epsilon certificate may prove only that any missed benefit is within the
requested gap; it never inserts an unevaluated or nonpositive commitment into
the selected payload.

Exact ties among evaluated incumbent policies retain all policy identities.
The lexicographically first identity is only the deterministic selected
representation when the value is positive. This display rule does not collapse
the scalar oracle's complete response correspondence.

## Certificate and identity

Every success certificate includes:

- incumbent lower bound and valid global upper bound;
- exact absolute and relative gaps;
- requested absolute and relative tolerances;
- full domain, baseline policy, response oracle, objective, and run identities;
- domain, objective, response, and bound contract versions;
- information-set/action/decision-variable counts;
- cells, nodes, active-cell peak, splits, prunes;
- point/bound/total oracle calls and cache hits;
- bound-record and identity-record counts.

Canonical JSON uses sorted keys, compact separators, UTF-8, and
`allow_nan=False`. Rationals use reduced integer or `numerator/denominator`
text. Noncanonical fractions such as `2/4`, decimal strings, binary floats,
booleans, non-finite values, stale pins, and malformed identities fail closed.
`exact_certified_global_optimizer_json` emits deterministic one-line strict
JSON. The public example is tested in two fresh processes for byte identity.

## Limits and failure atomicity

`CertifiedGlobalOptimizerLimits` covers:

- information sets, actions per information set, and decision variables;
- total cells and processed nodes;
- combined oracle calls and bound records;
- identity records;
- rational numerator/denominator bits;
- output records and bytes.

Every configured value is a positive integer no larger than its immutable hard
ceiling. Values above a hard ceiling are rejected; they are never clamped.
Domain counts are projected before canonical domain materialization. Child
cells, both child bound calls, and their possible point calls are preflighted
before the split is materialized. Bound and identity budgets are checked before
their corresponding record. Output records are counted before serialization,
and output bytes use bounded streaming preflight without accumulating a
partial JSON prefix.

Important fail-closed statuses include:

- `LIMIT_REACHED_NO_CERTIFICATE`;
- `UNSUPPORTED_DOMAIN`;
- `INVALID_INPUT` / `STALE_INPUT`;
- `INVALID_ORACLE_CONTRACT`;
- `ORACLE_FAILURE`;
- `NUMERIC_FAILURE` / `INTERNAL_FAILURE`.

Every failure has `payload=null` and `partial_result=false`. It carries bounded
error metadata and completed work counters, never a selected commitment,
partial leaf set, partial certificate, truncated response, or successful prefix.
There is no fallback, sampling, skipping, xfail, cap clamp, random restart,
local-optimizer certificate, or finite-grid certificate.

## Direct API

```python
from repeated_poker.certified_global_optimizer import (
    CertifiedGlobalOptimizerLimits,
    ExactActionProbability,
    ExactBehaviorPolicy,
    ExactBehaviorRow,
    HeroBehaviorScenario,
    HeroInformationSet,
    optimize_certified_global_hero_commitment,
)

scenario = HeroBehaviorScenario(
    scenario_identity="<lowercase-sha256>",
    information_sets=(
        HeroInformationSet("H:decision", ("check", "bet")),
    ),
)
baseline = ExactBehaviorPolicy(
    rows=(
        ExactBehaviorRow(
            "H:decision",
            (
                ExactActionProbability("check", "1/2"),
                ExactActionProbability("bet", "1/2"),
            ),
        ),
    ),
)

result = optimize_certified_global_hero_commitment(
    scenario,
    baseline,
    conforming_scalar_oracle,
    absolute_gap_tolerance="1/1000",
    relative_gap_tolerance="0",
    limits=CertifiedGlobalOptimizerLimits(max_cells=4095),
)
```

See `examples/certified_global_optimizer_core.py` for a complete analytic
linear fixture. Its bound is independently transparent: the objective is one
action probability minus its baseline value, and the whole-cell upper bound is
that action's interval upper endpoint.

## Claim boundary

An M36 certificate establishes only the specified-tolerance global maximum of
the identified baseline-relative scalar objective over the identified bounded
Hero behavior domain, conditional on the identified oracle's whole-cell bound
contract. It does not establish a Nash or other equilibrium, certify an
external solver, predict opponent learning, establish real-world
profitability, recommend gambling decisions, or support unlimited game size.
