# Known-board real-card HU certified-global optimization (M38)

`known_board_real_card_hu_certified_global.py` connects the stable M29
known-board heads-up river/rake model to the M36 certified continuous/global
optimizer. It is intentionally one bounded in-memory consumer.

## Frozen contracts

- contract: `m38-known-board-real-card-hu-certified-global-integration-v1`
- algorithm: `exact-rational-known-board-river-lex-dp-bnb-v1`
- schema: `m38-public-result-schema-v1`
- domain: `m29-full-surviving-hero-action-simplex-product-v1`
- objective: `m29-b-a-l-fixed-adaptation-total-uplift-v1`
- response: `m29-complete-factorized-oop-lex-response-v1`
- bound: `m29-river-terminal-affine-lex-interval-dp-bound-v1`
- certificate claim: `identified-bounded-scalar-objective-global-maximum-only-v1`

Hero is IP and Villain is OOP. The board is strictly five known cards. Extra
dead cards, M13 class/exact weighted ranges, one-time ordered joint
conditioning, M29 combo-to-bucket mappings, seven terminal river lines, fold
rake zero, showdown rake/cap, uncalled excess, and terminal conservation keep
the M29 meanings.

## Full legal domain

M38 asks the M29 tree for every surviving Hero information set and every legal
action. The domain is the Cartesian product of those closed action simplexes,
in canonical information-set/action order. The baseline is a feasible
comparison point only. Shift candidates, a finite candidate list, M27
selection, a grid, pure vertices, a warm-start box, and a local neighbourhood
do not reduce the domain.

## Repeated scalar

For one identified baseline profile:

```
b    = baseline Hero value against the fixed complete baseline Villain profile
a(p) = policy p Hero value against that same fixed Villain profile
l(p) = Hero-worst value over a fresh complete exact Villain best response

W_pre  = sum(t=0..m-2, discount**t)
W_post = sum(t=m-1..N-1, discount**t)
W_all  = W_pre + W_post

uplift(p) = W_pre*a(p) + W_post*l(p) - W_all*b
```

The baseline point is defined to reproduce `W_all*b` and exact zero uplift.
`response_tolerance` is therefore restricted to exact zero. Public binary64
range, profile, game, and rake inputs are lifted with
`float.as_integer_ratio`. Exact combo masses, conditioned probabilities, and
terminal payoffs are then reconstructed rationally from those inputs; every
terminal is checked against the native M29 binary64 tree within M29's numeric
tolerance and must satisfy exact conservation before it enters the oracle.
Subsequent objective, response, cell, cap, and gap arithmetic is rational.

## Complete response without pure enumeration

The M29 river topology factorizes by Villain bucket. For each bucket M38 first
solves `OOP_vs_IP_bet` and `OOP_vs_IP_raise`, then `OOP_first`. Each action
contribution is affine because an M29 terminal path crosses at most one Hero
information set. The point DP maximizes exact Villain value and, across every
tie, retains the Hero-worst and Hero-best values, exact correspondence count,
Hero-worst witness count, conditional best action sets, globally appearing
actions, action variation, and Villain-history provenance. It never samples or
enumerates the full opponent pure-strategy product, even when the M29
materialization cap is lower than that product.

## Sound whole-cell bound

For each M36 cell, M38 first intersects every action interval with its complete
simplex. Exact linear allocation computes the minimum and maximum of each
terminal affine over that intersection. At each downstream Villain
information set, an action remains possible precisely when its Villain upper
interval is at least the largest competing Villain lower interval. The Hero
upper interval is the maximum Hero affine upper value across that superset.
The root repeats the same operation after adding downstream DP intervals.
Summing bucket bounds and the fixed-profile affine bound yields:

```
max(0,
    W_pre * max_C a
  + W_post * sum_bucket lex_interval_DP_upper(bucket, C)
  - W_all * b)
```

For every policy in the cell, the actual best-response action is in the
retained possible-action superset, so this cannot be below the Hero-worst
point value. At a singleton all intervals are exact and the bound calls the
same exact lexicographic DP value, satisfying M36 singleton convergence.
Rake can make tie boundaries discontinuous; keeping every action that can be
optimal anywhere in a non-singleton cell is deliberately conservative and
does not erase that boundary.

The bound does not use a point-only value, metadata, finite candidates,
sampled pure responses, or a first tie witness.

## Failure and output contract

Success has a non-null payload and null error. Every failure has null payload,
one bounded error, `partial_result=false`, and no partial certificate.
Preparation, response, point, bound, M36 search, and output limits fail
closed. Final output caps apply to the entire successful result wrapper.
Their lazy preflight stops before aggregate `payload.to_dict()`,
`result.to_dict()`, or full success encoding when a low cap has already
failed; an output failure retains completed M36/M38 counters.

`exact_known_board_real_card_hu_certified_global_json` emits compact,
sorted-key, strict UTF-8 JSON. No truncation, sampling, fallback, silent clamp,
or partial result is available.

## Claim boundary

The certificate concerns only the global maximum (within the reported exact
gap) of the identified bounded scalar above. It is not a poker equilibrium,
ICM result, solver-grade scale claim, profitability claim, prediction of
adaptation, or strategy advice.

Run the deterministic example:

```console
python examples/known_board_real_card_hu_certified_global.py
```
