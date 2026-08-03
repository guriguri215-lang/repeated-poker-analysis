# Known-board real-card three-player river/rake adapter

## Purpose and boundary

`known-board-real-card-three-player-river-rake-v1` is the bounded in-memory
real-card adapter for the existing M30 exact non-cooperative response, M31
three-player river/rake accounting, and M32 candidate/repeated workflow. It
accepts one exact five-card river board, optional dead cards, separate weighted
Hero/O1/O2 ranges, a complete scenario-native three-player profile, one river
bet target, rake, and the existing finite M32 Hero-shift configuration.

It is not a raw solver-export parser, general poker-tree builder, CLI or saved
schema, unknown-runout evaluator, Monte Carlo method, coalition response,
equilibrium certificate, continuous/global optimizer, profitability claim, or
strategy advice. In particular, M35 implements the added mandatory R7
real-card three-player requirement. Separate M39 and M40 certified-global
workflows are now implemented; they do not turn this finite M32 workflow into a
continuous optimizer.

The module is public as
`repeated_poker.known_board_real_card_three_player_river`; it is intentionally
not re-exported from the package root.

## Chance layer: three-player joint conditioning

The chance layer is the standalone
`prepare_known_board_three_player_support(...)` API. It reuses the M13
`RangeSpec` grammar:

- an exact canonical two-card combo with `EXACT_COMBO_MASS`; or
- one of the canonical 169 classes with `CLASS_TOTAL_MASS`.

Class mass is divided over the class's canonical combos before removal, exactly
as in M13. Board/dead removal never redistributes mass within a range. The
adapter validates that board, dead cards, and every private combo are
collision-free.

For every compatible ordered triple `(h, o1, o2)`, the raw mass is

```text
w(h, o1, o2) = w_H(h) * w_O1(o1) * w_O2(o2).
```

Triples with any shared private card are excluded. The remaining triple support
is normalized once by the total compatible raw mass. The emitted H/O1/O2
conditional marginals are projections of that joint distribution; the betting
chance root is never reconstructed as a product of separately conditioned
marginals. This distinction makes a Hero blocker that changes both opponents,
or an O1/O2-only collision, propagate to the actual showdown and response
weights.

The public `PreparedThreePlayerSupport` retains:

- canonical board, dead, and unavailable cards;
- pre-removal, after-board, and final range identities and counts per seat;
- Cartesian, collision-excluded, and compatible triple counts;
- every compatible canonical triple with exact raw and conditioned mass;
- all three exact conditional combo marginals; and
- a deterministic content identity.

An empty support or zero compatible marginal fails explicitly. There is no
partial support, truncation, resampling, or two-player support reuse.

## Complete behavior profile and optional buckets

Without a supplied `ComboBucketMap`, every surviving combo is its own action
bucket. A strict map may group several combos for action abstraction only.
Every surviving combo must be assigned exactly once, every declared bucket must
be used, and stale/unknown combos fail. Combo ranks, blocker mass, and showdown
rows remain exact even when action buckets are shared.

`RealCardThreePlayerProfile` must cover every surviving bucket and every
decision below:

| player | decision | exact legal actions |
|---|---|---|
| H | `open` | `check`, `bet` |
| O1 | `after_hero_check` | `check` |
| O1 | `vs_hero_bet` | `call`, `fold` |
| O2 | `after_hero_o1_check` | `check` |
| O2 | `vs_hero_bet_o1_call` | `call`, `fold` |
| O2 | `vs_hero_bet_o1_fold` | `call`, `fold` |

Every action key is explicit and every probability is a canonical exact
rational in `[0,1]`; each row sums exactly to one. Missing and off-path rows
are invalid rather than silently filled.

`GeneratedTreeAttestation` supplies human-traceable verifier/date/evidence
fields. The adapter creates the native identity-bound M31 perfect-recall
attestation only after constructing the exact generated tree; it never
self-approves a caller-defined arbitrary tree.

## Exact showdown and M31 accounting

Each distinct surviving combo is evaluated once on the fixed board with the
existing deterministic seven-card evaluator. For each triple, exact ranks
produce the three-way winner set and all heads-up winner sets, including exact
ties.

The fixed M35 tree is:

```text
H check -> O1 check -> O2 check -> three-way showdown
H bet -> O1 call/fold -> O2 call/fold
  both call       -> three-way showdown
  exactly one call -> H versus that caller
  both fold       -> fold terminal
```

The requested `bet_to` is a total-contribution target. All three players begin
with the same positive `initial_contribution`. The native M31 accounting
therefore remains authoritative for:

- fold terminals with zero rake;
- two- and three-way showdowns;
- unmatched excess returned before pot award;
- rake zero or positive, with optional binding cap; and
- exact `H + O1 + O2 + R = 0` conservation at every terminal.

Side pots, all-in semantics, raises, multiple bet sizes, odd-chip allocation,
and other unsupported trees are outside this v1 adapter.

## Exact response, candidates, and repeated values

The canonical profile becomes a complete M31 fixed-Hero policy plus complete
O1/O2 initial profile. The adapter invokes the unchanged public M32 workflow:

1. the baseline is evaluated by a fresh M31/M30 call;
2. every retained finite M32 Hero candidate gets another fresh M31/M30 call;
3. M30 returns the complete non-cooperative O1/O2 exact response
   correspondence, including all ties and witnesses;
4. Hero safety is only `m31_scenario_response.response.hero_worst`; and
5. the unchanged repeated horizon/discount/adaptation-opportunity selector
   compares each candidate with the baseline.

The pure-profile subset and joint-plan/coalition stress remain diagnostics and
never replace the primary response. A first witness or `hero_best` never
replaces `hero_worst`. A no-benefit timing row has no selected commitment.

An external baseline is accepted only as the complete scenario-native
H/O1/O2 profile above. Parsing raw solver exports is not part of M35.

The current candidate universe is still the caller-declared bounded finite M32
shift lattice. It must not be described as R8, a continuous search, or a global
certificate.

## Caps, statuses, and no-partial behavior

Caller-lowerable `KnownBoardRealCardThreePlayerLimits` have immutable ceilings:

| workload | hard ceiling |
|---|---:|
| Cartesian triples | 1,000,000 |
| compatible triples | 32 |
| fixed-board hand evaluations | 96 |
| profile rows | 512 |
| buckets per seat | 32 |
| generated M31 nodes | 500 |
| identity records | 200,000 |
| output bytes | 256,000,000 |
| rational numerator/denominator bits | 1,024 |

The adapter also honors the nested M13/M30/M31/M32 caller-lowerable caps.
Cartesian support is checked before enumeration; compatible count is checked
before support allocation; tree nodes, chance outcomes, information sets,
pure plans, joint profiles, and identity workload are checked before ranking
or tree allocation. Nested M32 performs its own aggregate preflights before
candidate response work.

`KnownBoardRealCardThreePlayerResult` is exclusive:

- `SUCCESS` with one complete payload and no error; or
- an explicit `AiofStatus` with `payload=None` and a bounded error message.

Representative failures include `INVALID_CARD_INPUT`, `INVALID_RANGE`,
`EMPTY_COMPATIBLE_SUPPORT`, `ZERO_COMPATIBLE_MARGINAL`,
`INVALID_STRATEGY`, `CAP_EXCEEDED`, `UNSUPPORTED_MODEL`,
`ACCOUNTING_MISMATCH`, `ORACLE_MISMATCH`, and `NUMERIC_FAILURE`. Nested
failures never expose partial ranks, trees, candidates, responses, or
serialization.

## Deterministic identities

Canonical SHA-256 identities bind:

- conditioned support;
- complete profile and bucket maps;
- board/bet/rake scenario parameters;
- full baseline, including perfect-recall evidence;
- unchanged native M32 output; and
- the complete outer analysis.

The payload also pins the evaluator identity. An optional
`expected_baseline_identity` fails stale rather than running a different
baseline. `exact_known_board_real_card_three_player_json(...)` emits strict
one-line deterministic JSON.

## Worked example

Run:

```powershell
python examples/known_board_real_card_three_player_river_rake.py
```

The fixture uses board `2c 3d 4h 5s 9c`, H=`AsAh`, O1=`KsKh`,
O2=`QsQh`, contributions `10/10/10`, a bet target of `20`, rake `1/10`
without a cap, baseline Hero check, baseline opponents call, shift `1`,
horizon `3`, and discount `1`.

Hero's ace makes a wheel and wins every showdown. The exact hand checks are:

| state | H | O1 | O2 | R |
|---|---:|---:|---:|---:|
| baseline check, pot 30, rake 3 | 17 | -10 | -10 | 3 |
| candidate bet, both baseline-call, pot 60, rake 6 | 34 | -20 | -20 | 6 |
| exact response, both fold, no rake | 20 | -10 | -10 | 0 |

These fixture values validate one bounded case only. They are not a general
proof, global optimum, equilibrium, or real-world recommendation.
