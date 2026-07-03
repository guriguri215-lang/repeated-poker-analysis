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

- `T_detect` is based on local observable event distributions.
- The current local candidate detection is conditional on reaching the
  candidate's information set and observing an action there.
- It does not model full tree reach probability.
- It does not model real learning, memory, or statistical sophistication.
- KL-based estimates depend on the chosen log-likelihood threshold.

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
- Real solver range import
- Large-scale range solving
- Full tree reach detection
- Human opponent modeling
- Strategy recommendation for real-money play
- STT / ICM / push-fold implementation
- CLI / file output / web app
- Commercial paid product

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
