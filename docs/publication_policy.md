# Publication Policy

## Purpose

- This repository may be made public as an
  experimental research / learning project.
- The goal is to share the modelling approach, code, examples, and limitations.
- It is not a commercial solver, paid product, or real-money strategy engine.

## License

- The code is released under the MIT License.
- The license permits reuse, modification, distribution, and private use,
  subject to preserving the license and copyright notice.
- The software is provided as-is, without warranty.
- See `LICENSE`.

## How to describe the project publicly

Recommended phrasing:

- "An experimental Python toolkit for small abstract repeated-poker analysis."
- "A research / learning project about Hero commitment candidates, exact
  response diagnostics, `T_deadline`, and `T_detect` diagnostics."
- "A small MVP for exploring modelling assumptions, not a full poker solver."

Avoid phrasing:

- "Profitable poker strategy"
- "Solver-grade"
- "Guaranteed exploit"
- "Real-money recommendation"
- "Commercial poker solver"

## Publication boundaries

- Do not present outputs as gambling, bankroll, financial, legal, or real-money
  advice.
- Do not imply that positive model EV guarantees profitable poker play.
- Do not imply that `T_detect` predicts real opponent learning.
- Do not imply that `T_deadline` predicts actual human adaptation timing.
- Link to `docs/assumptions_and_limitations.md` when writing public
  explanations.

## Before changing repository visibility

- Run `python scripts/check_mvp.py`.
- Run `python -m pytest -q`.
- Review `docs/public_readiness_checklist.md`.
- Confirm no private paths, emails, tokens, or solver exports are committed.
- Confirm README, MVP walkthrough, examples guide, and assumptions document are
  consistent.
