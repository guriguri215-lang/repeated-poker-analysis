# Abstract and known-board real-card three-player certified global integration

`repeated_poker.three_player_certified_global` connects the M36 certified
branch-and-bound core to two bounded consumers:

- an abstract M31 three-player river/rake scenario; and
- an M35 strict known-five-card-board real-card three-player river/rake source.

Both consumers derive every surviving Hero information set and every legal
action from the validated scenario. The search domain is the complete product
of those exact action simplexes. A caller cannot supply a candidate list,
shift lattice, grid, vertex subset, local box, or warm-start neighbourhood.
M35's request still contains its historical M32 candidate configuration, but
M39 does not read it when deriving the domain or objective.

## Public API

The public requests are
`AbstractThreePlayerCertifiedGlobalRequest` and
`KnownBoardRealCardThreePlayerCertifiedGlobalRequest`. Use
`analyze_abstract_three_player_certified_global` or
`analyze_known_board_real_card_three_player_certified_global`, then serialize
with `exact_three_player_certified_global_json`.

The two `prepare_*_certified_global_oracle` functions expose the canonical M36
scenario, baseline policy, exact preparation evidence, and conforming point /
whole-cell oracle for audit and independent testing.

```powershell
python examples/three_player_certified_global.py
```

## Objective and complete response

Let `b` be Hero's value under the complete supplied baseline H/O1/O2 profile.
For a locked Hero policy `p`, let `a(p)` be its value against the same fixed
supplied O1/O2 profile. Each non-baseline point then makes a fresh public M31
call. M31 rebuilds the complete reduced payoff rectangle and M30 enumerates
the complete exact two-opponent non-cooperative Nash response
correspondence. `l(p)` is the minimum Hero value over that entire
correspondence.

For horizon `N`, discount `d`, and fixed adaptation opportunity `m`:

```text
W_pre  = sum(d^(t-1), 1 <= t < m)
W_post = sum(d^(t-1), m <= t <= N)
W_all  = W_pre + W_post

uplift(p) = W_pre*a(p) + W_post*l(p) - W_all*b
```

The supplied baseline identity is the no-commitment comparison point and has
exact uplift zero. It is also a feasible incumbent, but it does not restrict
the domain. First witnesses, pure stable subsets, Hero-best, coalition/joint
plan stress, one-opponent response, CFR, or a finite M32 candidate never
replace `l(p)`. Exact response ties and every Hero-worst witness remain in the
nested M31/M30 point record.

## Sound whole-cell upper bound

M31 validates every terminal twice and publishes exact terminal utilities
after uncalled return, fold/showdown rake, optional rake cap, award shares, and
conservation. Let

```text
M = max(exact Hero utility over all validated M31 terminal records).
```

At every Hero policy, against every fixed or responding O1/O2 behavior, the
expected Hero value is a convex combination of terminal utilities and is at
most `M`. All repeated weights are nonnegative. Therefore every policy in
every M36 cell satisfies

```text
uplift(p) <= W_all*M - W_all*b.
```

The consumer returns `max(0, W_all*(M-b))`, where zero includes the baseline
anchor. This deliberately conservative terminal envelope includes every
possible M30 equilibrium, every exact tie surface, and every rake/cap
boundary; it never removes a response because of samples or a local
calculation. It requires no unbounded materialization of the opponent pure
joint product.

For a cell whose simplex intersection determines one exact policy, the bound
calls the same fresh complete M31/M30 point oracle and returns its exact
uplift. This includes one-action rows and singleton intervals, so the
whole-cell contract has exact singleton convergence. The terminal envelope
can be loose. A loose bound may exhaust M36 limits and return
`LIMIT_REACHED_NO_CERTIFICATE`; it is never silently replaced by a local or
finite-candidate optimizer.

## Real-card preservation

The real-card variant uses the M35 validators and exact construction seam
without running the historical finite M32 candidate selector. It preserves:

- strict five-card board and dead-card collision checks;
- exact-combo / 169-class expansion and lossless binary64 weight lift;
- ordered H/O1/O2 compatible-triple conditioning exactly once;
- blocker-conditioned marginals, complete bucket profile, and seven-card
  showdown evaluation;
- two- and three-way winner shares;
- fold rake zero, showdown rake/cap, uncalled excess, and exact
  `H + O1 + O2 + R = 0` terminal conservation.

The complete prepared support, showdown rows, bucket maps, workload, and M35
identities remain in the M39 success payload.

## Exact arithmetic, identities, caps, and failures

Canonical rational text is reduced integer or `numerator/denominator`.
Binary64 inputs such as the historical M32 discount are lifted with
`float.as_integer_ratio()`; they are not reparsed from display decimals.
Nonfinite values, excessive rational growth, malformed structures, or
unsupported domains fail closed.

Canonical SHA-256 identities bind the M30/M31 contracts, scenario, tree,
baseline, complete domain, terminal envelope, repeated objective, all
effective caps, response oracle, and analysis. M39 pins accept raw lowercase
64-hex and the equivalent `sha256:` form. Stale or malformed pins return no
payload. Semantic mapping permutations have byte-identical output; meaningful
changes alter identities and bytes.

Preparation, response, point, bound, aggregate M31 work, M36 cells/nodes/
oracles, rational growth, and output all have caller-lowerable immutable
ceilings. Every failure has `payload=null` and `partial_result=false`.
Completed optimizer and consumer counters remain available.

The record and byte caps measure the complete public success wrapper. A lazy
deterministic traversal counts the exact canonical result before aggregate
success materialization. An exact `N` boundary succeeds, `N-1` fails, and a
very-low cap stops before `payload.to_dict`, result `to_dict`, or full success
serialization. There is no sampling, truncation, fallback, silent clamp,
successful prefix, or partial certificate.

## Claim boundary

A success certifies only the reported specified-tolerance global maximum of
the identified bounded scalar objective, conditional on the documented exact
consumer bound. It does not certify a poker equilibrium, a repeated-game
equilibrium, opponent learning or switch behavior, solver-grade scale,
profitability, collusion resistance, or strategy advice. M39 integration does
not complete the broader R8 or extended-product lifecycle; unified M40
workflow/closeout remains separate.
