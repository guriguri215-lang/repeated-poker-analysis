# Real-card AIoF public workflow

This guide shows one deliberately tiny, deterministic path through the existing
M13 public module APIs. It starts from explicit real-card combinations, verifies
one known board exactly, evaluates a supplied heads-up push/fold profile in
fee-zero ChipEV, and then runs the exact rational strategy wrapper for the same
declared game.

Run the worked example from the repository root:

```powershell
python examples/aiof_real_card_workflow.py
```

The script prints one strict JSON-safe summary line. It writes no files, uses no
network or external data, and omits runtime identity, absolute paths, credentials,
and full dataclass dumps.

## Fixture and range meaning

The worked fixture is:

- SB exact combo: `AsAh`.
- BB exact combo: `KsKh`.
- Known board: `2c 3d 4h 5s 7c`.
- The other 43 cards are explicit dead cards.

`RangeEntry`, `RangeSpec`, and `WeightBasis` distinguish exact-combo mass from
169-class total mass. A class such as `AKs` expands uniformly across its four
suited combinations before conditioning; an exact combo such as `AsAh` names one
specific two-card holding. The grammar is intentionally explicit: it is not a
range chart parser, shorthand such as `AA+` is unsupported, and no raw solver
range is imported or certified.

Dead cards remove conflicting exact combinations without silently redistributing
mass inside a class. SB and BB range masses are then conditioned once on ordered,
card-disjoint combo pairs. This card-removal and compatibility step is part of
the input identity, so changing a combo, board/dead card, mass, or algorithm
changes the identified computation.

## Phase 1: exact equity and supplied-profile ChipEV

The primary worked path selects `EquityAlgorithm.EXACT_EXHAUSTIVE`. With 43 dead
cards, exactly five board cards remain, so the hand oracle has one trial and one
board evaluation:

```text
trials=1, wins=1, losses=0, ties=0, board_evaluations=1
```

The key rule is that exact and deterministic Monte Carlo are non-interchangeable. Exact requests use
`seed=None` and `samples=None`; deterministic Monte Carlo requires an explicit
seed/sample contract and reports sampling statistics. The API never silently
falls back from one algorithm to the other, and this worked example does not use
Monte Carlo.

The ChipEV game has stacks 10/10, blinds 0.5/1, ante 0, fee 0, third-party dead
money 0, and no side pot. Values are net chip deltas from immediately before
mandatory posts. The supplied profile shoves `AsAh` with probability 1 and folds
`KsKh` to the shove with probability 1. The resulting values are SB `+1` and BB
`-1`, whose sum is zero.

`analyze_pushfold(..., best_response_seats=("sb", "bb"))` reports an exact
fixed-opponent response for each requested seat. It holds the other seat's
supplied complete combo profile fixed. It is not a joint equilibrium result, a
range chart, an optimal-Hero claim, or a profitability statement.

## Phase 2: exact rational-lift result

`generate_rational_lift_strategy` is a fail-closed wrapper for the declared
`aiof-rational-lift-game-v1`. The worked input has one compatible payoff cell
and one exact board evaluation. Its independently verified Fraction witness is:

```text
profile_value=Fraction(1), g_sb=Fraction(0), g_bb=Fraction(0)
```

The JSON summary renders these exact fractions as strings (`"1"`, `"0"`,
`"0"`) and fully qualifies the claim as
`aiof-rational-lift-game-v1:EXACT_NASH`. That label is scoped only to the finite
rational-lift game constructed by this module. It is not a Nash certificate for
an external poker game, not a solver-grade chart, not proof that Hero is optimal
in a broader game, and not a real-money recommendation.

## Caps, identity, and failure behavior

The example lowers the relevant range, combo-pair, board-evaluation, and solver
caps to the tiny fixture. Production APIs also enforce hard ceilings before
large payoff, board, tableau, trace, oracle, or sampling work is materialized.
Caps are refusal boundaries: callers must not clamp, truncate, skip, or silently
change the requested algorithm to obtain a result.

The underlying result objects separate semantic/input identities from runtime
identity. Runtime identity may differ between supported Python versions while
the exact profile and Fraction witness remain the same. The example intentionally
does not print unstable runtime or run identities.

Every phase is checked before stdout is emitted. An unexpected status returns a
nonzero exit code, writes a short error to stderr, and emits no partial payload
or strategy claim. A failure result must have no success payload; never interpret
completed payoff cells, pivots, traces, or earlier phase objects as a successful
answer.

## Scope boundary

This path is limited to heads-up preflop, fee-zero ChipEV, no third-party dead
money, and no side pot. It uses only existing public names from
`repeated_poker.aiof_cards`, `aiof_equity`, `aiof_chip_ev`, and `aiof_strategy`.
It adds no JSON file format, CLI operation, top-level package export, external
solver/data dependency, prepared two-street integration, candidate/repeated
pipeline, manifest/report contract, GUI feature, or large-scale solving path.
