# Assumptions and Limitations

## Purpose of this document

- This document states what the project assumes and what it does not claim.
- It is intended to prevent overinterpretation of outputs.
- It is **not legal, financial, gambling, or bankroll advice**.

## Project status

- Experimental research / learning tool.
- Built by a non-professional solver developer. Here "non-professional" is not
  self-deprecation; it simply means this is not a warranted, production-grade
  product.
- Developed with AI assistance.
- Outputs should be treated as diagnostic and exploratory, not as decisions.

## Game-model assumptions

- The tool works on small abstract two-player games.
- It is not a full poker solver.
- Rake makes the two-player payoff accounting non-zero-sum.
- The house / rake account is accounting only; it is **not a strategic player**
  and chooses no actions.
- The exact response is solved by backward induction over Villain information
  sets by default; the v0 pure-strategy enumerator remains available as
  `method="enumerate"` for small trees, guarded by a `max_pure_strategies`
  limit that also caps how large a tied best-response correspondence the
  default method materialises.
- Perfect recall is an input contract. The current structural guard rejects a
  repeated information set on a single path, but it is not a proof of complete
  game-tree correctness.
- Utilities are net hand payoffs under the model (with zero rake the game is
  zero-sum; rake makes it non-zero-sum).

## Baseline profile import assumptions

- Baseline profiles from external sources are chosen comparison inputs, not
  solver certification.
- The v1 import boundary is the existing scenario-native mixed strategy map
  format documented in
  [baseline_solution_import_format.md](baseline_solution_import_format.md).
- The project does not validate that an external source's abstraction matches
  the scenario. The scenario author is responsible for matching actions,
  bucket ids, ranges, payoff model, rake or ICM assumptions, and information-set
  meanings.
- Raw solver export parsing, real-card range import, card removal, and
  large-scale range solving remain non-goals.

## ICM assumptions

- The ICM backend implements Malmuth-Harville Independent Chip Model prize
  equity.
- ICM maps a stack vector and a payout vector to modelled tournament prize EV.
- It ignores position, blind increases, future hands, skill differences, and
  table dynamics.
- Its outputs are not real tournament predictions, not real-money advice, and
  not push/fold charts.
- Future-ICM, FGS, and tournament-simulation backends are later extensions, not
  part of the current ICM backend.

## STT push/fold assumptions

- The STT implementation covers a preflop SB-vs-BB push/fold spot after all
  other players have folded.
- SB has only `shove` or `fold`; BB has only `call` or `fold` against a shove.
- Showdown results are supplied as abstract bucket probabilities. The tool does
  not evaluate real cards, parse real ranges, or apply card removal.
- Terminal values are modelled tournament prize EV deltas from ICM. They are
  not chip EV, not real tournament predictions, not real-money advice, and not
  push/fold charts.
- The terminal accounting residual is the bystander prize EV delta. It may be
  negative even though river-chip rake remains non-negative.
- `T_deadline` and `T_detect` on STT scenarios repeat the same abstract spot as
  a sensitivity assumption. They do not simulate blind increases, eliminations,
  future stack evolution, or tournament dynamics.
- Limping, min-raising, non-all-in sizing, side pots, partial blind posting,
  Future-ICM, FGS, and tournament simulation are not part of STT v1.

## Hero commitment assumptions

- Candidate strategies represent fixed Hero mixed strategies (a commitment).
- Current candidates are simple probability shifts at single Hero information
  sets; they are not an exhaustive strategy search.
- A candidate being robustly profitable under the implemented criterion does
  **not** mean it is a true repeated-game equilibrium.
- The tool helps search for commitment candidates and produce diagnostics; it
  does not prove a full equilibrium.

## Villain response assumptions

- Villain's best response is computed inside the supplied finite tree only.
- Multiple best responses may exist (ties are real and reported).
- The worst / best Hero EV across Villain best responses is a diagnostic
  interval, not a single predicted outcome.
- The model does not predict human mistakes, psychology, table image, or
  meta-game adjustment.

## T_deadline assumptions

- `T_deadline` is a sensitivity analysis over an assumed switching opportunity.
- It assumes per-opportunity values are constant within each regime
  (baseline / locked-before-adaptation / after-adaptation).
- It does not estimate when Villain actually adapts.
- It should not be read as a psychological threshold.

## T_detect assumptions

- `T_detect` is a rough diagnostic of an expected detection-time scale, not a
  real opponent-learning model.
- The shared public-observation contract and the only supported adaptation
  timing interpretation are defined in
  [public_observables_and_adaptation.md](public_observables_and_adaptation.md).
- The default `local_v0` model is based on local observable event distributions.
  It is conditional on reaching the candidate's information set and observing an
  action there.
- The opt-in `reach_weighted_v1` model builds a per-hand public observation
  distribution from root-to-terminal path probabilities, using the baseline
  Villain profile and either baseline Hero or candidate Hero. One observation is
  one hand, so `t_detect_hands` is directly comparable to `T_deadline`
  opportunities.
- `reach_weighted_v1` supports `actions_only` and `showdown_reveal`. The reveal
  model uses only builder-supplied public showdown annotations; fold terminals do
  not reveal private buckets.
- Because `reach_weighted_v1` does not use private buckets unless they are
  publicly revealed, it can be slower than a real observer with more
  information. Because it assumes the candidate distribution `P1` is known
  exactly, it can also be faster than a real observer that must estimate the
  alternative. It is therefore neither an upper nor a lower bound on real
  detection time.
- A `t_detect_hands` value of `None` means no signal under the chosen observation
  model, not safety and not real-world undetectability.
- Candidate pre-filtering can use `reach_weighted_v1` as a diagnostic pruning
  option before candidate comparison. Its minimum threshold applies only to
  finite `t_detect_hands`; `None` is not filtered out.
- It does not model real learning, memory, or statistical sophistication.
- KL-based estimates depend on the chosen log-likelihood threshold.
- `T_detect` may be compared with `T_deadline` only under the documented
  idealized threshold-observer convention. Under that convention,
  `detected_adaptation_is_at_least_baseline` asks what happens if adaptation is
  immediate at the estimated detection time; it is not a behavioural prediction.

## Output interpretation warnings

- Positive EV in the model **does not guarantee profitable poker play**.
- No output should be treated as direct real-money advice.
- The model can be wrong if the abstraction, ranges, rake, or action set are
  wrong.
- Results should be checked with hand calculations and small examples first.
- Large deviations from baseline may be easier to detect, but detectability is
  only approximated here.

## Current non-goals

- Full poker solver
- Raw solver export parsing or real-card range import
- Large-scale range solving
- Cross-spot reach detection and real opponent-learning models (within-spot
  reach-weighted `T_detect` v1 is implemented)
- Human opponent modeling
- Strategy recommendation for real-money play
- Future-ICM / FGS / tournament-simulation backend
- STT limp / raise / non-all-in sizing, side-pot, or real-card evaluation
- Publicly hosted web service (the GUIs are local-only prototypes)
- New GUI features (the five local editor/analyze prototypes are frozen;
  bug fixes only)
- Commercial paid product

CLI runners, file exports (JSON / CSV / Markdown), and local-only GUI
prototypes exist and are therefore no longer non-goals; they remain thin
input/output layers over the same small-tree analysis core.

## Responsible publication notes

- Future articles should present this as a learning / research project.
- Avoid claims like "profitable poker strategy," "solver-grade," or
  "guaranteed exploit."
- Include limitations prominently in any public writing.

## Checklist before public release

- README links to this document.
- The MVP walkthrough is up to date.
- Example outputs are reproducible.
- Tests pass.
- No private paths, tokens, personal emails, or solver exports are committed.
- Claims are consistent with the implemented features.
