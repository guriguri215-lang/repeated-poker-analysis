# 02. Research and Implementation Plan

## Objective

Analyse a one-hand poker solution when comparable spots recur and opponents can observe, learn, and adapt. The first deliverable is not a story about reputation. It is a reproducible analyser that reports:

- the baseline one-hand solution and its EVs;
- each fully fixed Hero candidate policy;
- Villain’s exact best-response correspondence to that policy;
- Hero EV under adverse, favourable, and reference best-response tie rules;
- an economic deadline for Villain adaptation and a separate estimate of how quickly the commitment can be identified; and
- a run manifest tying the assumptions, input files, code version, and results together.

## Mathematical commitments

### 1. Do not conflate finite repetition with reputation

If the final opportunity is common knowledge, each spot is independent, monitoring is complete, and both players are fully rational, backward induction normally makes repeated play of the one-hand equilibrium the relevant finite-horizon prediction. Therefore, an immediate losing call is not automatically justified merely because the number of repetitions is `N`.

The project distinguishes four analysis modes.

| Mode | Meaning | What can be claimed |
|---|---|---|
| `known_finite` | The number of remaining comparable opportunities `N` is common knowledge. | A payoff and sensitivity analysis. Do not make a reputation-equilibrium claim without additional assumptions. |
| `uncertain_horizon` | Every opportunity has a specified stopping hazard. | A stationary continuation value can be analysed through the implied discount factor. |
| `infinite_discounted` | Infinite opportunities with `0 < delta < 1`. | Repeated-equilibrium candidates can be tested under explicit public-history and strategy-space assumptions. |
| `adaptive_opponent` | Villain follows a specified learning or update rule. | A behavioural prediction, not automatically a game-theoretic equilibrium. |

`N` means the number of opportunities for a comparable spot, not necessarily the number of dealt hands. A separate occurrence-rate input is needed to convert opportunities into physical hands.

### 2. A Hero lock is a commitment assumption

To lock Hero is to impose a mixed strategy at **every Hero information set** in the target tree: check, fold, bet, call, raise, and every other legal Hero decision. It represents an external commitment device, a public policy, or a sufficiently credible reputation that prevents Hero from deviating.

If Hero may freely abandon the fixed strategy later, a node-lock result alone is not an equilibrium. Use the following names precisely.

| Name | Definition | Role |
|---|---|---|
| Baseline solution `sigma0` | The unconstrained one-hand equilibrium or imported reference strategy. | Comparison baseline. |
| Minimum-Villain-EV candidate `piH_minV` | A Hero policy that lowers Villain EV while Villain remains at baseline `sigmaV0`, subject to explicit constraints. | A candidate-generation mode matching the original idea. |
| Villain response correspondence `BR_V(piH)` | Every Villain strategy that maximises Villain EV against fully fixed `piH`. | The response after the commitment. |
| Best candidate commitment `piH_leader` | The selected policy from a finite candidate library, evaluated against `BR_V(piH)` under a stated tie rule. | The main recommendation in the first version. |
| Repeated-equilibrium candidate | A full profile whose deviation incentives have been tested at every relevant public history. | Call it an equilibrium only when those tests pass. |

The original procedure—find `piH_minV`, lock Hero, calculate Villain’s response, then compare Hero EV—is a useful candidate generator. It is not by itself a proof of a repeated-game equilibrium.

If the Hero lock is credible, there is no additional free Hero optimisation after Villain responds. The process does not automatically return to the ordinary one-hand equilibrium. It is a commitment problem. If the Hero lock is not credible, then the usual two-player equilibrium analysis is required instead.

### 3. Candidate library rather than a premature global optimiser

The first version need not optimise over every continuous strategy probability. It should construct a finite and inspectable candidate library, including:

1. candidates that lower Villain’s baseline EV;
2. systematic perturbations of Hero’s baseline action frequencies; and
3. candidates that retain positive Hero post-response EV while increasing the distance in **observable** behaviour from baseline.

For each candidate, report post-response Hero EV, Villain EV, observable distribution distance, and adaptation timing. Present a Pareto frontier rather than forcing all objectives into a single score. The final choice depends on how credible and visible a commitment can be.

### 4. Rake is a non-zero-sum payoff component

The house does not choose actions, but action-dependent rake and a rake cap make the two-player game non-zero-sum. Each terminal must satisfy:

```text
u_H + u_V + rake = 0
```

This matters for baseline-solving algorithms. An algorithm designed only for zero-sum equilibrium guarantees cannot be assumed to solve an action-dependent-rake game correctly. Once Hero is fixed, however, Villain’s problem is still a one-player expected-utility maximisation problem.

## River analysis: inputs and exact response

### Inputs

The interchange format is JSON and/or CSV. The project does not directly read a proprietary solver’s internal file format.

1. `SpotSpec`: board, pot, stacks, positions, full action tree, and rake rules.
2. `RangeSnapshot`: combo weights for both players after card removal; each conditional range must normalise correctly.
3. `StrategySnapshot`: action probabilities for every information set and legal action.
4. `BaselineSolution` (optional): an imported baseline solution with EV, solver name/version, convergence information, and export time. A small internally solved game may supply the baseline instead.
5. `RepeatedSpec`: `N`, `delta` or stopping hazard, public observables, occurrence rate, and opponent-adaptation model.

Solver exports that cannot legally be distributed are never committed to the repository.

### Exact Villain response after a full Hero lock

The main product is an analysis tool outside existing poker solvers. An external solver may provide `sigma0`, but it is not called to resolve every candidate.

After every Hero information set is fixed, Villain still has all legal decisions in the tree: check, fold, bet, call, raise, and so on. The tool enumerates legal private-card combinations, chance outcomes, and fixed Hero action probabilities, then calculates continuation values for every Villain action at every Villain information set.

Under the ordinary assumptions of a finite extensive-form game with perfect recall and expected-utility maximisation, at least one pure Villain best response exists against a fixed Hero behavioural strategy. This remains true in a non-zero-sum game. A mixed best response is also valid whenever Villain is indifferent among actions.

This means that CFR is not required as the primary calculation. CFR may return a mixed average strategy because of ties, symmetry, or finite-iteration approximation. The exact response engine should instead use dynamic programming or one-player sequence-form optimisation over the full Villain action tree.

### Best-response ties are material

If several Villain actions have equal Villain EV, the mixture among them can change Hero EV. The response is therefore a correspondence, not a single unexplained policy. Every candidate report must include:

- `BR_V(piH)`: indifferent information sets and their optimal actions;
- `EV_H_worst`: Hero EV under the Villain-optimal response that minimises Hero EV;
- `EV_H_best`: Hero EV under the Villain-optimal response that maximises Hero EV; and
- `EV_H_reference`: Hero EV under a declared reference tie rule, for example the optimal mixture closest to the baseline frequencies.

The default decision criterion is robust: accept a candidate only when `EV_H_worst` exceeds the baseline by the required tolerance. A report may also display optimistic and baseline-continuity values.

At a zero-reach Villain information set, the ex-ante strategy does not identify a unique response. The input must provide either an off-path rule or a small tremble/exploration probability.

### River workflow

1. Validate ranges, card removal, legal actions, terminal rake, and imported baseline values.
2. Build the finite Hero candidate library under legal, distance, and short-run-loss constraints.
3. Run the original compatible search: hold `sigmaV0` fixed and seek candidates that lower Villain EV.
4. For every candidate, calculate the full `BR_V(piH)` correspondence without calling an external solver.
5. Compare the Hero EV interval with the baseline and retain the robustly profitable candidates.
6. Evaluate repetition and adaptation for retained candidates.
7. Sweep rake, cap, sizing, baseline-solver error, Villain optimisation quality, and observation noise.

## Adaptation deadline and identification time

The placeholder name `M` is replaced by two distinct measures.

Let `b` be Hero’s value per opportunity under the baseline, `a` be Hero’s value while locked but before Villain adapts, and `l` be Hero’s value after Villain adapts. If Villain switches at opportunity `m`, then:

```text
V_lock(m) = sum(t=1..m-1, delta^(t-1) * a)
          + sum(t=m..N,   delta^(t-1) * l)

V_base    = sum(t=1..N, delta^(t-1) * b)

T_deadline = max { m in [1, N] : V_lock(m) >= V_base }
```

`T_deadline` is the **adaptation deadline**: the latest opportunity at which Villain must adapt for the locked policy to remain at least as valuable as baseline. In a stateful model, the implementation uses conditional values by history rather than constant `a` and `l`.

The intuition that a larger departure from baseline is easier to identify needs a distinct observational model. Define `T_detect`, the **identification time**, from observable action distributions only. A first approximation uses expected evidence per opportunity `D_obs` and a decision threshold `lambda`:

```text
T_detect = ceil(lambda / D_obs)
```

Showdown frequency, hidden cards, observation noise, exploration, and occurrence rate must be inputs or sensitivity variables. A practical screening rule is:

```text
T_detect <= T_deadline
```

The first quantity is an economic deadline; the second is a behavioural-identification estimate. They must never be presented as the same thing.

## STT SB-vs-BB Push/Fold

The initial interpretation is preflop: everyone folds to the small blind, SB chooses fold or shove, and BB chooses fold or call. If a literal flop BvB spot is intended, it becomes a later game-tree extension.

Inputs include the payout table, all stacks, blinds, antes, card distribution, and any required rake rule. Utility is prize EV, not chip EV. The first backend is explicit ICM, with a Future-ICM or tournament-simulation backend designed as a later replaceable extension.

The same fixed-Hero analysis applies. With a fixed BB call policy, enumerate card removal and compare SB shove/fold prize EV by SB information set. With a fixed SB policy, compare BB call/fold. Ties are reported as a response set with robust and optimistic Hero values.

## Planned architecture

```text
repeated-poker-analysis/
  README.md
  docs/                 # specifications, decision records, input documentation
  examples/             # small distributable inputs and expected outputs
  src/repeated_poker/
    domain/             # Range, Strategy, Spot, Rake types and validation
    games/river/        # river action trees and terminal payoff rules
    games/stt/          # SB-vs-BB Push/Fold and prize-value interface
    exact_response/     # full Villain action tree after Hero is fixed
    toy_solver/         # tiny baseline games only; not a full NLHE solver
    imports/            # baseline-solution and LockSpec import/export
    repeated/           # continuation values, T_deadline, T_detect
    reporting/          # manifests, tables, plot-ready data
    cli/
  tests/
    unit/ integration/ regression/ property/
```

Implementation status note: the current implementation intentionally keeps a flat module layout (one module per concern directly under `src/repeated_poker/`, with CLI and GUI scripts under `scripts/`) instead of the subpackage split above. The flat layout is a deliberate choice while modules stay reviewable in isolation; the split remains the long-term shape and will be revisited when module size or coupling forces it.

Generated results, commercial solver exports, and private hand histories are not committed. Every run writes a `run_manifest.json` containing input hashes, code revision, dependency lock, seed, utility units, rake rule, horizon assumptions, adaptation model, and solver/optimiser status. (Status: implemented as a `manifest` block embedded in the analysis exports -- scenario file SHA-256, format version, package version, best-effort git commit, UTC timestamp, and effective parameters -- rather than a separate `run_manifest.json`; dependency lock and seed are not applicable to the standard-library-only, deterministic analysis.)

## Development phases

| Phase | Deliverable | Exit criterion |
|---|---|---|
| 0. Specification | Input schema, utility units, rake semantics, and defined terminology. | The nuts-chop example can be calculated by hand. |
| 1. Exact response core | Non-zero-sum river tree, full Hero lock, Villain response correspondence, and exact values. | Tests cover zero rake, cap boundaries, probability normalisation, and Hero EV ranges under Villain ties. |
| 2. Repeated value | Known finite, uncertain horizon, and discounted modes; `T_deadline`; observable-distance calculations. | Enumeration matches analytical examples and explains every acceptance or rejection. |
| 3. Range and baseline import | Normalised snapshots, CSV/JSON adapters, and LockSpec. | Broken inputs are rejected and a run is reproducible from its manifest. |
| 4. STT Push/Fold | ICM prize-value backend, SB/BB response correspondence, and repeated evaluation. | Symmetry, extreme-stack, and known ICM cases pass. |
| 5. Extensions | Future ICM, occurrence rates, richer adaptation, and multi-player games. | New assumptions are explicit and comparisons preserve the earlier benchmark suite. |

The first implementation must stop after the small nuts-chop-with-rake benchmark and its tests. Do not combine actual range imports, STT, and learning models in the first change.
