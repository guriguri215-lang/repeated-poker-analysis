# Known-board real-card heads-up river/rake adapter

## Purpose and boundary

`known-board-real-card-hu-river-rake-adapter-v1` is a bounded in-memory
adapter for one already-dealt heads-up river opportunity. Hero is always IP and
Villain is always OOP. The input contains exactly five fixed board cards,
optional extra dead cards, M13 weighted class/exact-combo ranges, complete
action profiles, river sizes and rake, and a finite lattice of Hero probability
shifts.

The adapter produces a conditional comparison over that declared finite
library. It is not an equilibrium solver, a continuous or global optimizer, a
range-chart generator, an unknown-board/runout calculation, a Monte Carlo
method, an external-solver certificate, a profitability claim, or strategy or
real-money advice. It has no JSON/CLI/pipeline integration and is intentionally
not re-exported from the top-level package.

## Card and joint-distribution contract

Board and extra-dead input use the strict M13 card grammar. The board must have
exactly five distinct cards. Board/dead duplication and collision fail closed.
Range class mass is divided among the class's canonical combos before card
removal and is never redistributed after board or dead cards remove combos.

For every surviving ordered Hero/Villain exact-combo pair, raw mass is

```text
w(H, V) = w_H(H) * w_V(V).
```

Pairs whose private cards collide are excluded. All remaining pairs are
conditioned exactly once by their total compatible raw mass. The adapter never
forms chance rows from the product of post-conditioned seat marginals, because
that would discard blocker correlation. A surviving combo with zero compatible
marginal, an empty side, or an empty compatible joint support is a failure.

Each successful joint row keeps both canonical combos, both action-profile
bucket IDs, raw and conditioned mass, both public seven-card `HandRank` values,
and the exact Hero win, Villain win, or chop result. The fixed board is evaluated
once per compatible pair; it is never enumerated as a runout.

## Action buckets and complete profiles

The default mapping assigns every surviving exact combo to a bucket with that
same canonical combo ID. A caller may instead supply a `ComboBucketMap` to make
several combos share one private action abstraction. Declarations and
assignments are strict: every surviving combo occurs exactly once, every
assignment names a declared bucket, no stale or unknown combo is accepted, and
every declared bucket must be nonempty. Sharing a bucket changes only action
probabilities; card mass, blocker provenance, ranks, and showdowns remain on the
individual joint rows and are never averaged.

Hero must provide both complete decisions for every Hero bucket, including
off-path decisions:

| decision | exact action keys |
|---|---|
| `after_oop_check` | `check`, `bet` |
| `vs_oop_bet` | `call`, `fold`, `raise` |

An optional Villain profile must provide all three decisions for every Villain
bucket:

| decision | exact action keys |
|---|---|
| `oop_first` | `check`, `bet` |
| `vs_ip_bet` | `call`, `fold` |
| `vs_ip_raise` | `call`, `fold` |

Every legal action key is explicit, finite, in `[0, 1]`, and each distribution
sums to one within the requested tolerance. Missing keys are not treated as
zero. If no Villain profile is supplied, the adapter obtains the exact DP best-
response correspondence to the baseline Hero profile, selects its deterministic
representative, records `source="auto_best_response"`, and still emits a
complete off-path Villain profile. Baseline `b` is the fixed-profile Hero value
against that representative, not an implicit substitution of the response
correspondence's Hero-worst endpoint.

## Native river tree and accounting

The root is the explicit conditioned joint chance distribution. Every joint row
has the current seven terminal lines: check-check, check-bet-call,
check-bet-fold, bet-call, bet-fold, bet-raise-call, and bet-raise-fold. Shared
buckets share information sets across opponent combo branches while retaining
the actor's own action history, so the native perfect-recall guard remains in
force. The adapter does not convert the request through the abstract scenario
builder.

The output unit is
`net_chips_before_initial_commitments_per_river_opportunity`. Initial
commitments are already in the pot and payoffs are measured immediately before
them. Showdowns apply the supplied rake rate and optional cap. Fold terminals
apply zero rake, and unmatched excess is returned before the winner receives
the matched pot. Every terminal enforces
`Hero EV + Villain EV + house rake = 0`.

## Candidates, exact response, and repeated selection

For every feasible shift amount, candidate generation moves probability from
one source action to one target action at one Hero information set. With
`max_simultaneous_info_sets=2`, all products of feasible shifts at two distinct
information sets are also included. There is no filtering:
`generated_candidate_ids == kept_candidate_ids` and
`filtering_applied=False`. A shared bucket applies its changed distribution to
every combo branch in that bucket.

The adapter retains the native `HeroStrategyCandidate`,
`CandidateComparisonReport`, and `BestResponseResult` objects. Exact response is
always `method="dp"`; the enumerator is used only by independent tiny tests.
Compact correspondences continue to report the exact best-response count,
action variation, off-path sets, Hero worst/best interval, rake extremes, and
both exact and materialized counts. The public projection explicitly sets
`correspondence_materialization_complete`; one representative must never be
read as the full correspondence.

For each candidate the value labels are:

- `b`: baseline Hero against the fixed baseline Villain profile;
- `a`: candidate Hero against that same fixed Villain profile;
- `l_worst` / `l_best`: Hero worst/best over the candidate's exact Villain
  best-response correspondence.

The unchanged M27 `select_automatic_commitments` consumes the genuine native
comparison. It evaluates `m=1..N+1`, uses a strictly positive threshold, keeps
all primary ties, applies the existing secondary order, and emits
`NO_BENEFICIAL_COMMITMENT` when appropriate. An empty candidate universe is a
successful result with all `N+1` rows unselected.

## Hard ceilings and failure atomicity

Caller limits may lower these values but may not raise them; raising one is an
`INVALID_INPUT`, never a silent clamp.

| projected resource | hard ceiling |
|---|---:|
| compatible joint rows / fixed-board evaluations | 2,000 |
| native tree nodes | 24,001 |
| action-profile buckets per seat | 64 |
| Hero information sets | 128 |
| Villain information sets | 192 |
| candidates | 2,000 |
| candidate probability cells | 1,000,000 |
| fixed-profile node visits | 5,000,000 |
| response node visits | 5,000,000 |
| M27 timing rows | 1,000,000 |
| best-response list materialization | 100,000 |

Scalar/grammar and hard-limit configuration are checked first. Range support
and card removal are bounded, compatible support is streamed, mapping/profile
coverage is checked, and every remaining workload is projected. Only after all
caps pass are joint ranks, tree nodes, and candidates materialized. Candidate
identity collisions are rejected before candidate comparison/response work.
There is no truncation, first-N behavior, sampling, automatic grouping,
fallback, skip, or partial payload.

The outer `KnownBoardRealCardHuRiverResult` uses the existing `AiofStatus`.
`SUCCESS` has exactly one complete payload and no error; every failure has
`payload=None` and one sanitized error. A failure after several internal
candidate evaluations still exposes no prefix result.

## Identities and deterministic projection

Canonical JSON plus SHA-256 separates board, prepared joint, profile mapping,
tree, baseline, candidate, response, and analysis identities. An optional
`expected_baseline_identity` pins the baseline before candidate analysis.
Board, dead, range-entry, mapping-row, profile-row/action, and shift ordering are
semantic permutations: they do not change strict JSON output bytes. Card/range,
mapping/profile, rake, or size changes affect the baseline; shift-universe
changes affect candidates and analysis; horizon/discount/threshold changes
affect analysis only. Public `to_dict()` values are JSON-safe and reject NaN or
Infinity.

## Tiny in-memory example

This one-pair fixture uses the default one-combo-per-bucket mapping and a
supplied complete Villain profile. Empty `shift_amounts` intentionally asks for
no candidate commitments.

```python
from repeated_poker.aiof_cards import RangeEntry, RangeSpec, WeightBasis
from repeated_poker.known_board_real_card_hu_river import (
    ActionProbability,
    KnownBoardRealCardHuRiverRequest,
    RiverActionProfile,
    RiverProfileRow,
    analyze_known_board_real_card_hu_river,
)


def dist(**values):
    return tuple(ActionProbability(action, value) for action, value in values.items())


hero = RiverActionProfile((
    RiverProfileRow("AsAh", "after_oop_check", dist(check=0.5, bet=0.5)),
    RiverProfileRow(
        "AsAh", "vs_oop_bet", dist(call=0.5, fold=0.25, raise=0.25)
    ),
))
villain = RiverActionProfile((
    RiverProfileRow("KsKh", "oop_first", dist(check=0.5, bet=0.5)),
    RiverProfileRow("KsKh", "vs_ip_bet", dist(call=0.5, fold=0.5)),
    RiverProfileRow("KsKh", "vs_ip_raise", dist(call=0.5, fold=0.5)),
))
request = KnownBoardRealCardHuRiverRequest(
    board=("2c", "3d", "4h", "5s", "9c"),
    hero_range=RangeSpec((
        RangeEntry("AsAh", 1.0, WeightBasis.EXACT_COMBO_MASS),
    )),
    villain_range=RangeSpec((
        RangeEntry("KsKh", 1.0, WeightBasis.EXACT_COMBO_MASS),
    )),
    baseline_hero_profile=hero,
    baseline_villain_profile=villain,
    rake_rate=0.05,
    rake_cap=3.0,
    shift_amounts=(),
)
result = analyze_known_board_real_card_hu_river(request)
assert result.payload is not None
assert result.payload.provenance.compatible_pair_count == 1
assert result.payload.joint_rows[0].showdown_result == "hero"
assert result.payload.automatic_selection.rows[0].status == (
    "NO_BENEFICIAL_COMMITMENT"
)
```

For nontrivial libraries, explicitly supply shared bucket maps before default
one-to-one buckets exceed a hard ceiling. The adapter never invents that
abstraction for the caller.
