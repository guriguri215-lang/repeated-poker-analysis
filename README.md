# Repeated Poker Analysis

This directory contains the design documents for a new Claude Code project. Its purpose is to analyse poker as a repeated game: commitment, opponent adaptation, and the difference between a one-hand solution and a strategy used repeatedly in comparable spots.

The existing project at `poker sim $EV\繰り返しゲームの解析` is reference material only. Nothing in that project is changed by this work.

## Current design decisions

| Topic | Decision |
|---|---|
| First target | A river spot with ranges and rake. This is where the repeated-game core will be validated. |
| Second target | Preflop SB-vs-BB Push/Fold in an STT. The earlier phrase “flop BvB” is interpreted as preflop because the described spot begins after everyone folds to the small blind. |
| Game model | Two strategic players plus a non-strategic house rake account. Rake makes the game non-zero-sum; it does not by itself create a third strategic player. |
| Hero lock | Hero’s mixed strategy is fixed at every Hero information set in the target tree, including check, fold, bet, call, and raise decisions where legal. |
| Villain response | Villain retains every legal action. The tool calculates Villain’s exact best-response set to the fully fixed Hero strategy. |
| Baseline equilibrium | An existing solver may optionally provide the baseline solution. The first version does not embed or control that solver. |
| Analysis form | A fixed-Hero response is a commitment analysis, not automatically a repeated-game equilibrium. Known finite repetition, uncertain horizon, and discounted infinite repetition are reported separately. |
| Implementation | Start a clean standalone project rather than extending the earlier prototype. |
| Quality bar | Mathematical specifications, input validation, hand-calculated benchmarks, reproducible run manifests, and tests are required from the beginning. |

The original idea—find Hero strategies that lower Villain’s EV while Villain initially remains at the baseline strategy, then evaluate Villain’s response—is retained as a candidate generator. It is not the only criterion. Each candidate is evaluated after Villain’s response, with explicit treatment of best-response ties.

## Project document

- [02_research_and_implementation_plan.md](02_research_and_implementation_plan.md) — mathematical model, response correspondence, timing measures, inputs and outputs, and development phases.

## Current working state

- GitHub registration, Git for Windows installation, and Git author configuration are complete.
- The local project root is this `repeated-poker-analysis` directory.
- The first program and its tests will be created locally before a GitHub repository is created.
- No Git repository, remote, or GitHub repository is created by these documents.

The GitHub and Claude Code operating procedure is intentionally kept outside this initial repository content. The first commit will contain only this README, the implementation plan, and the initial program with its tests.

## Decisions to fix before implementation expands

1. Confirm that the STT target is preflop SB-vs-BB Push/Fold rather than a literal flop spot.
2. Define the baseline-solution import format, if an external solver is used.
3. Decide whether the first STT value backend is ICM only or must include a Future-ICM / tournament-simulation backend.
4. Define the public observables and the opponent adaptation model before interpreting a commitment result as a behavioural prediction.

## Development

The first program lives in `src/repeated_poker/` with worked inputs in
`examples/` and tests in `tests/`. It is a self-contained, finite exact
best-response analyser; it does not call any external solver.

Install the development dependencies (pytest) into a virtual environment:

```
python -m venv .venv
# Windows:        .venv\Scripts\activate
# macOS / Linux:  source .venv/bin/activate
pip install -e ".[dev]"
```

Run the test suite from the project root:

```
pytest
```

Run the worked examples:

```
python examples/nuts_chop_river.py
python examples/value_bluff_river.py
```

The exact enumerator materialises the entire Villain pure-strategy space and is
intended for small abstract trees only; it is guarded by a configurable
`max_pure_strategies` limit and is not yet the scalable, range-based response
engine described in the implementation plan.

### Detection time (`T_detect`)

`repeated_poker.detection` provides a v0 detection-time estimate
(`calculate_detection_time`, `calculate_candidate_local_detection`). It compares
two observable event distributions (for example, action frequencies) with the
total variation distance and the KL divergence in nats, then converts the
divergence into a required number of observations via a log-likelihood
threshold.

`T_detect` is a sensitivity analysis based on observable event distributions. It
is not a psychological model, not a real learning-speed estimate, and not a full
opponent-adaptation model. It is separate from `T_deadline`: `T_deadline` is an
economic adaptation deadline, while `T_detect` is a behavioural-identification
estimate. Strategy-space L1 distance and observable-distribution distance are
different concepts and must not be conflated.

`build_candidate_analysis_report` can optionally include a per-candidate local
`T_detect`: pass `baseline_hero_strategy` together with
`detection_log_likelihood_threshold` (and optionally
`detection_occurrence_probability_per_opportunity`). Each row then carries the
detection distances and two distinct detection-vs-deadline reads:

- `t_detect_is_no_later_than_t_deadline` is a pure time comparison
  (`estimated_opportunities <= t_deadline`). It does **not** mean Hero is
  economically safe: `t_deadline` is only the latest passing opportunity, and
  Hero EV need not be monotone in the switching opportunity.
- `detected_adaptation_is_at_least_baseline` is the economic read. It maps the
  estimated detection opportunity onto the adaptation-deadline timing rows
  (clamped to the `m = N+1` never-adapts row beyond the horizon) and reports
  whether Hero is at least at baseline EV if Villain adapts exactly then.

This local model is conditional on reaching the candidate's information set,
ignores tree reach probability, and does not guarantee real opponent learning
or adaptation.

### Markdown summary

`format_candidate_analysis_markdown` can render a human-readable Markdown
summary from a `CandidateAnalysisReport`. It is presentation-only: it does not
change analysis results, and it does not write files (it returns a string).

### Candidate pre-filter

`filter_candidates` is a lightweight pre-comparison pruning helper for generated
candidates (by allowed information set, strategy-space L1 distance, or a local
detection minimum). It does not replace `compare_candidates` or
`select_candidates`. The detection-based filter uses local observable
distributions and does not model tree reach probability or real opponent
learning.

### Analysis pipeline

`run_candidate_analysis_pipeline` wires candidate generation, optional
pre-filtering, fixed-profile comparison, analysis reporting, and optional
Markdown rendering into a single call for a small abstract game. It is an
orchestration helper, not a new solver; it does not write files and adds no
CLI.

### MVP walkthrough

See [docs/mvp_walkthrough.md](docs/mvp_walkthrough.md) for an end-to-end
explanation of the current minimum viable workflow and how to read the pipeline
output.

### Assumptions and limitations

See [docs/assumptions_and_limitations.md](docs/assumptions_and_limitations.md)
for the modelling assumptions, interpretation limits, and responsible-publication
notes.
