# Real-card AIoF preflop certified global integration

`repeated_poker.aiof_preflop_certified_global` is the M37 consumer adapter
between the M28 heads-up real-card preflop model and the M36 certified global
optimizer. It leaves the existing finite-shift M28 bridge unchanged.

## Exact scope

The adapter certifies the specified-tolerance global maximum of one identified
baseline-relative repeated Hero ChipEV objective. It is not an equilibrium,
Nash chart, ICM model, solver-grade scalability claim, real-world
profitability claim, or strategy recommendation. M37 completes this real-card
preflop consumer integration. Separate M38, M39, and M40 integrations are now
implemented and complete the documented R8 lifecycle; that completion does not
broaden M37's certificate claim.

Only exact exhaustive equity is supported. Fee, third-party dead money, side
pots, Monte Carlo, and nonzero response tolerance fail closed. Every failure
has a null payload and therefore contains no selected policy, certificate,
prepared support, or partial prefix.

## Full legal domain

This is the full legal domain; no declared finite candidate subset replaces
it.

The caller supplies the existing `RangeSpec`, public dead cards, game,
baseline `SuppliedProfile`, and Hero seat. The adapter calls the public
blocker-conditioning preparation contract first. Class entries are expanded
uniformly over the complete class before dead-card removal. The full legal
domain is then derived as follows:

1. take every surviving exact Hero combo in canonical card order;
2. make that combo one Hero information set;
3. attach both legal actions (`shove`/`fold` for SB Hero or `call`/`fold` for
   BB Hero);
4. give every information set the complete nonnegative, sum-to-one simplex;
5. take the product of all those simplexes.

The baseline profile is only the no-commitment comparison and initial feasible
incumbent. It does not restrict any simplex. There are no shift amounts,
candidate lists, maximum shifted-combo counts, grids, local bounds, or
warm-start neighbourhoods in this contract.

The scenario identity binds canonical dead cards, public prepared-range
identity, pre- and post-blocker exact support, exact compatible-pair
probabilities, exhaustive showdown counts, fee-zero game accounting, Hero
seat, and both legal actions. Range, profile, action, and dead-card input
permutations therefore retain the same semantic identities and output bytes.

## Exact point objective

Fix horizon `N`, exact discount `delta`, and adaptation opportunity `m` in
`1..N+1`. Define the nonnegative exact weights

```
W_pre  = sum(t=0..m-2, delta^t)
W_post = sum(t=m-1..N-1, delta^t)
W_all  = W_pre + W_post
```

For Hero behavior policy `p`:

- `b` is baseline Hero EV against the supplied baseline opponent;
- `a(p)` is Hero EV for `p` against that same fixed opponent;
- `l(p)` is Hero EV after a fresh complete opponent best response, using the
  correspondence-wide Hero worst.

The locked total and comparison total are

```
V_lock(p) = W_pre * a(p) + W_post * l(p)
V_base    = W_all * b
uplift(p) = V_lock(p) - V_base
```

This is the M28 repeated convention at one declared adaptation opportunity.
The baseline policy itself means “do not choose a new commitment”, so its M36
point value is exactly `V_base` and its uplift is exactly zero. Every other
policy uses `V_lock`.

All range weights, game amounts, and baseline probabilities arrive through
the existing binary64 dataclasses. Their actual binary64 values are lifted
losslessly with `float.as_integer_ratio()`; the code never parses their display
decimals or uses a tolerance to hide float error. Exact exhaustive integer
win/loss/tie counts, exact class multiplicities, exact blocker-conditioned
joint masses, integer horizon, and canonical rational discount then produce
the certificate coefficients. Unsafe or non-finite input is unsupported.

## Complete response correspondence

Opponent response is calculated fresh from the exact matchup coefficients for
every point. The opponent has one information set per surviving own combo.
Both pure legal actions are evaluated exactly. Every action tied at the exact
best value remains in the factorized complete response correspondence; a
first witness is never substituted.

The response identity binds all factorized rows, exact reach, exact action
contributions, and the policy-specific set of best actions. The
`complete_response_record_count` is the Cartesian cardinality of the complete
factorized best-response correspondence. Because fee-zero heads-up ChipEV is
zero-sum, every member of that exact best-response correspondence attains the
same Hero-worst value, so `hero_worst_witness_count` has the same cardinality.
The rows remain factorized rather than materializing the Cartesian product.

For SB Hero, write `p_i` for shove probability. Hero fold payoff is common to
all BB response rows. Each BB combo chooses the smaller Hero contribution of
call and fold. For BB Hero, each SB combo chooses the smaller Hero contribution
of shove and fold. These are exactly the M28 payoff and Hero-seat conventions.

## Consumer-specific whole-cell proof

For exact affine forms `c(p)` and `L[j,r](p)`, the post-response value is

```
l(p) = c(p) + sum_j min_r L[j,r](p).
```

Let `C` be an M36 closed cell intersected with every two-action simplex. For
every `p` in `C` and every opponent row `j`,

```
min_r L[j,r](p)
    <= min_r max_(q in C) L[j,r](q).
```

Also `c(p) <= max_C c` and `a(p) <= max_C a`. Summing these inequalities and
multiplying by the nonnegative repeated weights gives the valid whole-cell
uplift upper bound

```
U(C) = max(
    0,
    W_pre  * max_C a
      + W_post * (
            max_C c
          + sum_j min_r max_C L[j,r]
        )
      - V_base
).
```

The leading zero covers the special baseline no-commitment point. An affine
maximum is exact: for each information set the active-action interval is

```
max(active.lower, 1 - passive.upper)
..
min(active.upper, 1 - passive.lower),
```

and the appropriate endpoint is chosen from the coefficient sign. Thus both
stored action intervals and the sum-to-one simplex constraint are used. At a
singleton cell every affine maximum is the point value, so `U(C)` converges to
the exact objective. M36 supplies the rational center/split/derived-gap logic;
M37 does not provide an alternate split path.

This proof depends on the actual M37 coefficients and nonnegative repeated
weights. M36 metadata, sampled points, a finite grid, pure vertices, M28
candidate values, random restarts, and local optimizers are not bounds and are
not used as substitutes. A cell whose exact intervals or affine construction
cannot be validated raises `UnsupportedScalarOracleDomain`.

## Public API

`AiofPreflopCertifiedGlobalRequest` binds:

- game, SB/BB ranges, dead cards, baseline profile, and Hero seat;
- horizon, adaptation opportunity, canonical rational discount, and exact-zero
  response tolerance;
- absolute and relative M36 gap tolerances;
- public real-card caps, M37 integration caps, M36 caps, and identity pins.

`prepare_aiof_preflop_certified_global_oracle(request)` exposes the full
scenario, canonical baseline policy, prepared support projection, all consumer
identities, and the conforming point/bound oracle. It raises on invalid
preparation input so tests and other exact consumers can inspect the oracle.

`analyze_aiof_preflop_certified_global(request)` is the fail-closed adapter.
Success is only `CERTIFIED_GLOBAL` or `CERTIFIED_EPSILON_GLOBAL`. Its payload
retains the native M36 result losslessly, plus the exact baseline and selected
consumer point records. `exact_aiof_preflop_certified_global_json(result)`
returns deterministic strict one-line JSON.

## Caps and failure atomicity

Caller caps may be lowered but values above immutable hard ceilings are
rejected rather than clamped. The public range layer preflights range entries,
expanded combos, compatible pairs, and card validity. M37 then preflights:

- per-seat Hero/opponent information sets;
- compatible support and matchup rows;
- exhaustive board evaluations before the matchup iterator;
- coefficient records before coefficient construction;
- cumulative response rows/actions before each point response;
- consumer bound records before each bound;
- records and canonical UTF-8 bytes for the complete final public result,
  including the status, error, payload, native and consumer counters, through a
  deterministic budget-aware traversal before allocating an aggregate success
  result/payload projection or full encoded byte string.

The output traversal has the exact successful shape serialized by
`exact_aiof_preflop_certified_global_json`. A cap equal to the measured record
or byte count succeeds; one less returns
`LIMIT_REACHED_NO_CERTIFICATE`, phase `output`, a null payload, and the already
completed native M36 and consumer work counters. Keys, records, or bytes are
never truncated to fit.

M36 independently preflights cells, nodes, oracle calls, bound records,
rational sizes, identities, and native output. Cap exhaustion never falls back
to sampling, truncation, a partial result, a skipped test, or an uncertified
candidate.

## Example

From a checkout:

```powershell
$env:PYTHONPATH = "$PWD\src"
$env:PYTHONDONTWRITEBYTECODE = "1"
python examples\aiof_preflop_certified_global.py
```

The example uses one blocker-conditioned SB combo and one BB combo, exact
exhaustive evaluation, horizon two, and adaptation opportunity two. It prints
one deterministic certified JSON record.
