# Exact real-card AIoF candidate/repeated bridge

`repeated_poker.aiof_preflop_candidate_repeated` is a bounded in-memory bridge
between the existing real-card supplied-profile evaluator and the existing
automatic repeated-value selector. It answers a narrow conditional question:

> Within this finite, explicitly declared library of exact-combo probability
> shifts, which Hero candidate has the largest repeated total net ChipEV for
> each assumed opponent adaptation opportunity?

It does not solve for the baseline profile, search a continuous strategy space,
or establish equilibrium, optimality, opponent learning, profitability, or
strategy advice.

## Supported model and public API

Construct `AiofPreflopCandidateRepeatedRequest` and call
`analyze_aiof_preflop_candidate_repeated`. The request contains:

- one `HeadsUpChipEvGame`, with fee, third-party money, and side pots all zero;
- explicit SB and BB `RangeSpec` values plus public dead cards;
- one complete `SuppliedProfile` over the prepared exact-combo supports;
- `hero_seat="sb"` or `"bb"`;
- a unique tuple of strictly positive shift amounts and
  `max_shifted_combos` equal to one or two;
- the finite horizon, discount, tolerance, minimum total uplift, and
  caller-lowerable M13/bridge caps.

Only `EquityAlgorithm.EXACT_EXHAUSTIVE` with `seed=None` and `samples=None` is
supported. Monte Carlo or sampling controls return `UNSUPPORTED_MODEL`; the
bridge never falls back to a different algorithm. SB Hero's active action is
shove and BB Hero's active action is call.

```python
from repeated_poker.aiof_preflop_candidate_repeated import (
    AiofPreflopCandidateRepeatedRequest,
    analyze_aiof_preflop_candidate_repeated,
)

result = analyze_aiof_preflop_candidate_repeated(
    AiofPreflopCandidateRepeatedRequest(
        game=game,
        sb_range=sb_range,
        bb_range=bb_range,
        dead_cards=dead_cards,
        baseline_profile=complete_profile,
        hero_seat="sb",
        shift_amounts=(0.05, 0.10),
        max_shifted_combos=2,
        horizon=100,
        discount=0.99,
    )
)
if result.payload is not None:
    payload = result.payload.to_dict()
```

Callers must check both `result.status` and `result.payload`. Success has a
complete payload and no error. Every failure has `payload=None` and a short
error; work completed before a later candidate failure is not returned.

## Exact candidate universe

The baseline profile is validated as a complete exact copy over the prepared
post-removal supports. No missing row is filled, and no probability is
normalized, clamped, or redistributed. For every Hero exact combo and every
declared shift `d`, the bridge generates each feasible directed move:

```text
fold -> active, if p + d <= 1
active -> fold, if p - d >= 0
```

One candidate contains one shift, or (when enabled) two shifts on two distinct
exact combos. A class label is never shifted as a unit. The bridge does not
drop, rank, Pareto-filter, or truncate candidates. An empty feasible universe
is successful and yields exactly `N + 1` `NO_BENEFICIAL_COMMITMENT` rows.

For shifts `d_1, ..., d_k`, the reported full two-action strategy-space
distance is

```text
L1 = 2 * sum(d_i).
```

Candidate order, IDs, and serialized results are deterministic and independent
of range-entry, profile-row, dead-card, or shift-amount insertion order.

## Workload projection and refusal caps

Caps are checked before candidate/profile materialization and before any real-
card analysis. Let `n_i` be the feasible directed shift count for Hero combo
`i`, and `S = sum_i n_i`. The exact candidate count is

```text
C = S                                      (one shifted combo)
C = S + (S^2 - sum_i n_i^2) / 2           (up to two shifted combos)
```

If one exact supplied-profile analysis needs `E` board evaluations and the
opponent prepared support has size `R`, the bridge checks:

```text
(C + 1) * E       <= max_total_board_evaluations
(C + 1) * R       <= max_response_rows
C * (N + 1)       <= max_timing_rows
C                 <= max_candidates
```

The existing per-analysis M13 range, compatible-pair, exact-board, dead-card,
cache, and trace ceilings also remain authoritative. Exceeding any cap returns
`CAP_EXCEEDED`; the request is not sampled, clamped, partially evaluated, or
silently reduced.

## Value and response semantics

All values are heads-up fee-zero net chip deltas measured immediately before
mandatory posts. For every baseline/candidate exact analysis the bridge checks
zero-sum conservation. With Hero's baseline value `b`, a candidate's value
against the fixed supplied opponent `a`, and the opponent's exact best-response
value, the post-response Hero value is

```text
l = - opponent_best_response_value.
```

Fee-zero zero-sum makes the conservative and optimistic Hero response values
equal. The bridge still retains the native `UnilateralBestResponse`: every
opponent exact-combo row, action value, reach probability, and full best-action
tie is serialized. It does not materialize the Cartesian product of pure
opponent strategies.

The automatic selector evaluates every candidate for all assumed adaptation
opportunities `m = 1, ..., N + 1`. Before `m` it uses `a`; from `m` onward it
uses `l`; the comparison baseline is `b`. `m=N+1` means no adaptation within
the modeled horizon. An assumed `m` is not a prediction of opponent behavior.

## Identity and determinism

The baseline SHA-256 identity binds the game/accounting contract, both prepared
range identities and exact supports, dead cards, compatible-pair count, Hero
seat, complete canonical baseline profile, and exact algorithm. A caller may
provide `expected_baseline_identity` to reject stale inputs.

Each candidate ID binds the full baseline identity, Hero seat, and complete
canonical shift descriptor. The analysis identity additionally binds the full
candidate universe, generation settings, repeated settings, caps, algorithm,
coverage, and workload projection. Identities exclude timestamps, absolute
paths, runtime metadata, and insertion order.

## Interpretation boundary

The result is exhaustive only over the declared finite exact-combo shift
library. It is not a preflop chart, an endogenous or external-game solution, a
global or local continuous optimum, a Nash or repeated-game equilibrium, an
opponent-adaptation model, an ICM/tournament result, a profitability guarantee,
or real-money advice. Candidate selection should always be read together with
the reported baseline identity, coverage, caps, workload, native response rows,
and repeated configuration.
