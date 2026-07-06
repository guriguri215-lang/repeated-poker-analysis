# Public Observables And Adaptation Interpretation

## Purpose

This page defines the public-observation contract used by `T_detect` diagnostics
and the narrow convention under which a detection-time diagnostic may be compared
with `T_deadline`.

It does not add a new detection method, scenario field, solver model, or
opponent-learning model. Results remain diagnostics under explicit modelling
assumptions.

## Public Observation Contract

A terminal public observation is:

```text
public root-to-terminal action sequence
+ optional public reveal labels supplied by the scenario builder
```

The action sequence records the public Hero and Villain actions on the path to a
terminal. Optional reveal labels are builder annotations for abstract labels that
the model treats as public at that terminal.

`TerminalReveals` is the internal v1 representation of those terminal reveal
annotations:

```text
Mapping[terminal_id, None | tuple[str, ...]]
```

It is sufficient for v1 and is not a scenario JSON field. Scenario authors do
not add `TerminalReveals` directly to input files.

## Terminal Reveal Semantics

Fold terminals reveal the public action path only. No private bucket is revealed,
so the terminal reveal entry is `None`.

Call or showdown-style terminals also reveal the public action path. If the
builder says abstract showdown labels are public, the terminal reveal entry is
the tuple supplied by that builder. The detection code uses that tuple exactly as
supplied; it does not infer real cards, ranges, blockers, or card removal.

For `showdown_reveal`, every terminal has an internal reveal entry. `None` means
there is no public private-label reveal at that terminal.

## River And STT Examples

River scenarios and STT push/fold scenarios use the same public-observation
concept.

For river scenarios:

- Fold terminals have no reveal.
- Showdown or call-style terminals may carry builder-supplied abstract hand or
  bucket labels when that scenario builder marks those labels as public.

For STT push/fold scenarios:

- SB fold terminals have no reveal.
- BB fold-after-shove terminals have no reveal.
- Call terminals reveal `(sb_bucket_id, bb_bucket_id)` in the abstract bucket
  model, because the all-in showdown labels are public within that model.

## Detection Methods And Observation Models

`local_v0` is a local frequency diagnostic. It compares observable Hero action
distributions at the changed Hero information set or sets. It is conditional on
reaching those information sets, does not include tree reach probability, and
does not combine evidence across public paths. Use it for the question: how
large is the local action-frequency change once this decision point is reached?

`reach_weighted_v1` builds public observation distributions for one complete
abstract hand/opportunity in the model from root-to-terminal path probabilities
under the fixed baseline Villain profile. It compares baseline Hero (`P0`) with
candidate Hero (`P1`) over a public observation channel. Because public reach
through the tree is included, this is the preferred method when comparing
`T_detect` with `T_deadline` in comparable-opportunity units.

The v1 observation models are:

- `actions_only`: use only the public action sequence.
- `showdown_reveal`: use the public action sequence plus the builder-supplied
  reveal tuple when a terminal has one.

`showdown_reveal` is opt-in and includes only public labels that the builder has
explicitly supplied.

The candidate pre-filter may also use `reach_weighted_v1` when a detection
minimum is requested. It uses this same public-observation contract before the
candidate-comparison stage and interprets the threshold as a minimum finite
`t_detect_hands`; `None` means no signal under the selected observation model
and is not pruned by that filter.

## Comparable Opportunities And Physical Hands

`T_deadline`, `T_detect`, and `t_detect_estimated_opportunities` are expressed
in comparable abstract opportunities. Those opportunities are not necessarily
physical dealt hands in a wider game or session.

Two occurrence probabilities are intentionally separate:

- `detection_occurrence_probability_per_opportunity` is local-v0 only. It
  converts local observations at the changed information set into comparable
  opportunities.
- `comparable_spot_occurrence_probability_per_physical_hand` is an optional
  report configuration key. It is a cross-spot population frequency for how
  often the comparable abstract spot occurs per physical dealt hand.

The physical-hand conversion writes `t_detect_estimated_physical_hands` only
when both `t_detect_estimated_opportunities` and the comparable-spot physical
hand probability are present. It is computed after detection, as:

```text
ceil(t_detect_estimated_opportunities
     / comparable_spot_occurrence_probability_per_physical_hand)
```

This probability is not a single-tree reach probability. For `reach_weighted_v1`,
within-spot reach is already inside the one-hand public observation
distribution. For `local_v0`, the local observation-to-opportunity conversion
comes first, and the physical-hand conversion comes second.

The optional physical-hand conversion is not a scenario JSON field, not a new
detection method, not an opponent-learning model, not a real-world forecast of
when a person adapts, and not a profitability guarantee.
The CLI flag
`--detection-comparable-spot-occurrence-probability-per-physical-hand` exposes
only this report-side conversion and records the supplied value in the run
manifest.

## Threshold-Observer Adaptation Convention

`T_detect` is primarily a detectability diagnostic. It may be connected to
adaptation timing only under this idealized threshold-observer convention:

1. The same abstract spot repeats independently over the horizon.
2. The observer sees the chosen public observation channel.
3. The observer compares the baseline distribution `P0` with the candidate
   distribution `P1`.
4. `P1` is treated as known for the likelihood-ratio diagnostic.
5. A positive log-likelihood threshold is chosen outside the solver.
6. Detection occurs at the reported `T_detect` abstract hand/opportunity.
7. Adaptation is immediate and deterministic at that abstract hand/opportunity.
8. Post-detection play is represented by the same exact-response mode used for
   `T_deadline`.

Under only those assumptions, `detected_adaptation_is_at_least_baseline` is a
modelled adaptation timing diagnostic: if this threshold observer adapts at the
estimated time, does locked Hero EV remain at least baseline in the timing row
used by the report?

This convention is not a claim about human behaviour and not a real
learning-speed estimate.

## What May And May Not Be Inferred

It is appropriate to say that a result is a diagnostic under the selected public
observation channel, threshold, horizon, baseline profile, candidate profile, and
exact-response convention.

It is not appropriate to read `T_detect` as a behavioural prediction, a model of
memory, a model of psychology, or an estimate of when a real opponent will
adapt. Passing a timing diagnostic does not promise real-world poker results.

Because `reach_weighted_v1` uses no private labels unless the builder marks them
as public, it may be slower than an observer with extra information. Because it
treats `P1` as known, it may be faster than an observer that must first estimate
the alternative. It is therefore neither an upper nor a lower bound on real
detection time.

## Non-Goals

This v1 contract does not implement:

- Real opponent learning, Bayesian updating, memory, priors, or false-positive
  modelling.
- A replacement structured object for `TerminalReveals`.
- New detection methods beyond `local_v0` and `reach_weighted_v1`.
- New scenario JSON fields, or CLI flags that change detection math.
- Real-card evaluation, card removal, or chart generation.
- Future-ICM, FGS, or tournament simulation.
- Gambling, bankroll, staking, financial, legal, or real-money advice.
