# Repeated Poker Analysis

This is an experimental research / learning project for analyzing small abstract
poker subgames as repeated-game commitment problems. It focuses on candidate
Hero commitment strategies, exact Villain responses in small finite trees,
`T_deadline`, `T_detect`, and readable summaries.

## What this project is

- A small Python toolkit for abstract repeated-poker analysis.
- A way to generate and evaluate candidate Hero commitment strategies.
- A tool for exact-response diagnostics on small finite trees.
- A project with explicit assumptions, limitations, examples, and MVP checks.

## What this project is not

- Not a full poker solver.
- Not a real-money strategy recommendation tool.
- Not gambling, bankroll, financial, or legal advice.
- Not a guarantee of profitable play.
- Not yet connected to real solver ranges or large-scale range solving.

## Fastest way to run the MVP

```powershell
python scripts/check_mvp.py
```

- This runs the test suite and key examples.
- For a guided explanation, see `docs/mvp_walkthrough.md`.
- For example order, see `docs/examples_guide.md`.
- For assumptions and limitations, see `docs/assumptions_and_limitations.md`.

## Current design decisions

| Topic | Decision |
|---|---|
| First target | A river spot with ranges and rake. This is where the repeated-game core will be validated. |
| Second target | Confirmed: preflop SB-vs-BB Push/Fold in an STT (fixed 2026-07-05). The earlier phrase "flop BvB" is interpreted as preflop because the described spot begins after everyone folds to the small blind. |
| STT value backend | The first STT value backend is ICM only. Future-ICM and tournament-simulation backends are later, separately designed extensions (fixed 2026-07-05). |
| Game model | Two strategic players plus a non-strategic house rake account. Rake makes the game non-zero-sum; it does not by itself create a third strategic player. |
| Hero lock | Hero's mixed strategy is fixed at every Hero information set in the target tree, including check, fold, bet, call, and raise decisions where legal. |
| Villain response | Villain retains every legal action. The tool calculates Villain's exact best-response set to the fully fixed Hero strategy. |
| Baseline solution import contract | Baseline-solution import v1 is the existing scenario-native mixed strategy map format. External-source profiles may be converted outside this project into the current baseline strategy fields; the project does not parse or certify raw solver exports. |
| Public observables / adaptation interpretation | Public observations are the public action path plus optional builder-supplied reveal labels. `T_detect` can be compared with `T_deadline` only as a diagnostic under the idealized threshold-observer convention; real opponent-learning and behavioural prediction remain unsupported. |
| Analysis form | A fixed-Hero response is a commitment analysis, not automatically a repeated-game equilibrium. Known finite repetition, uncertain horizon, and discounted infinite repetition are reported separately. |
| Implementation | Start a clean standalone project rather than extending the earlier prototype. |
| Quality bar | Mathematical specifications, input validation, hand-calculated benchmarks, reproducible run manifests, and tests are required from the beginning. |

The original idea - find Hero strategies that lower Villain's EV while Villain initially remains at the baseline strategy, then evaluate Villain's response - is retained as a candidate generator. It is not the only criterion. Each candidate is evaluated after Villain's response, with explicit treatment of best-response ties.

## Project document

- [02_research_and_implementation_plan.md](02_research_and_implementation_plan.md) - mathematical model, response correspondence, timing measures, inputs and outputs, and development phases.
- [docs/baseline_solution_import_format.md](docs/baseline_solution_import_format.md) - v1 baseline-solution import boundary over existing scenario-native profile fields.

## Current working state

- The project is tracked in Git and developed through small pull requests.
- The current MVP includes candidate generation, candidate pre-filtering, exact response diagnostics for small trees, `T_deadline`, `T_detect` (`local_v0` by default, `reach_weighted_v1` opt-in), analysis reports, Markdown summaries, a high-level pipeline API, and an experimental STT SB-vs-BB push/fold ICM scenario path.
- The main end-to-end entry point is `run_candidate_analysis_pipeline`.
- The quickest local sanity check is `python scripts/check_mvp.py`.
- The project remains experimental and intended for small abstract games; see the assumptions and limitations document before interpreting outputs.

## Decisions to fix before implementation expands

No open blocking decisions are currently tracked here. The baseline-solution
import format is fixed conservatively as existing scenario-native mixed strategy
maps; see [docs/baseline_solution_import_format.md](docs/baseline_solution_import_format.md).

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

For a quick local sanity check that runs the test suite and the key examples in
one command, use:

```
python scripts/check_mvp.py
```

Run the worked examples:

```
python examples/nuts_chop_river.py
python examples/value_bluff_river.py
```

The exact Villain response is computed by lexicographic backward induction
over Villain information sets by default (`solve_exact_response(...,
method="dp")`), whose cost is linear in the tree size. The v0 enumerator is
kept as `method="enumerate"`: it materialises the entire Villain pure-strategy
space, is guarded by a configurable `max_pure_strategies` limit, and serves as
the small-tree oracle in the equivalence tests. Both methods compute Villain's
best response to a fixed Hero strategy; neither is an equilibrium computation.

### Detection time (`T_detect`)

`repeated_poker.detection` provides the default `local_v0` detection-time
estimate (`calculate_detection_time`, `calculate_candidate_local_detection`). It
compares two observable event distributions (for example, action frequencies)
with the total variation distance and the KL divergence in nats, then converts
the divergence into a required number of observations via a log-likelihood
threshold.

See [docs/public_observables_and_adaptation.md](docs/public_observables_and_adaptation.md)
for the shared public-observation contract and the narrow threshold-observer
adaptation convention.

`T_detect` is a sensitivity analysis based on observable event distributions. It
is not a psychological model, not a real learning-speed estimate, and not a full
opponent-adaptation model. It is separate from `T_deadline`: `T_deadline` is an
economic adaptation deadline, while `T_detect` is a detectability diagnostic.
Strategy-space L1 distance and observable-distribution distance are different
concepts and must not be conflated.

`build_candidate_analysis_report` can optionally include per-candidate
`T_detect`: pass `baseline_hero_strategy` together with
`detection_log_likelihood_threshold`. The default `local_v0` method can also use
`detection_occurrence_probability_per_opportunity`. The opt-in
`reach_weighted_v1` method additionally takes the tree, baseline Villain
strategy, and an observation model (`actions_only` or `showdown_reveal`), then
builds per-hand public observation distributions from root-to-terminal path
probabilities. Each row then carries detection fields and two distinct
detection-vs-deadline reads:

- `t_detect_is_no_later_than_t_deadline` is a pure time comparison
  (`estimated_opportunities <= t_deadline`). It does **not** mean Hero is
  economically safe: `t_deadline` is only the latest passing opportunity, and
  Hero EV need not be monotone in the switching opportunity.
- `detected_adaptation_is_at_least_baseline` is the economic read. It maps the
  estimated detection opportunity onto the adaptation-deadline timing rows
  (clamped to the `m = N+1` never-adapts row beyond the horizon) and reports
  whether Hero is at least at baseline EV if Villain adapts exactly then.

The default local model is conditional on reaching the candidate's information
set and does not include tree reach probability. The opt-in v1 model includes
within-spot reach in its per-hand observation distribution. Neither model is a
real opponent-learning model, and neither should be read as a behavioural
prediction outside the documented threshold-observer convention.

### Markdown summary

`format_candidate_analysis_markdown` can render a human-readable Markdown
summary from a `CandidateAnalysisReport`. It is presentation-only: it does not
change analysis results, and it does not write files (it returns a string).

### Candidate pre-filter

`filter_candidates` is a lightweight pre-comparison pruning helper for generated
candidates (by allowed information set, strategy-space L1 distance, or a local
detection minimum). It does not replace `compare_candidates` or
`select_candidates`. The detection-based filter remains tied to the local
observable-distribution model; v1 filtering is left for a later task.

### Analysis pipeline

`run_candidate_analysis_pipeline` wires candidate generation, optional
pre-filtering, fixed-profile comparison, analysis reporting, and optional
Markdown rendering into a single call for a small abstract game. It is an
orchestration helper, not a new solver; it does not write files and adds no
CLI.

### JSON scenario input

An abstract river spot can be described in a JSON file and turned into a
`GameTree` plus pipeline inputs with `load_river_scenario_json` and
`build_river_steal_game_from_scenario` (see
`examples/scenarios/nuts_chop_steal_bet98.json` and
`python scripts/run_river_scenario.py <scenario.json>`).

The scenario JSON format is currently version `"1"`. New files should declare it
with a top-level `"format_version": "1"`; the field is optional for backward
compatibility, so a file without it is treated as `"1"`. Unknown versions (and a
numeric `1`, `null`, a bool, or an empty string) are rejected. The format is
still experimental and may get a v2, so the version is recorded in the build
metadata and in the analysis / validation / batch outputs.

The input has three mutually exclusive modes:

- **single-hand mode**: a top-level `showdown` and `baseline_hero_strategy`.
- **abstract Hero range mode**: a `hero_range` of weighted hands, each with its
  own `showdown` and `baseline_strategy` (see
  `examples/scenarios/abstract_range_steal_bet98.json`).
- **abstract Hero/Villain range matrix mode**: a `hero_range` (without per-hand
  `showdown`), a `villain_range`, and exactly one matchup matrix keyed by
  `[hero_id][villain_id]` -- either a `showdown_matrix` of discrete
  `chop`/`hero`/`villain` results (see
  `examples/scenarios/range_matrix_steal_bet98.json`) or an `equity_matrix` of
  Hero pot shares before rake in `[0, 1]` (see
  `examples/scenarios/range_equity_steal_bet98.json`).

By default the baseline Villain is derived automatically as the exact best
response to the baseline Hero strategy. Any mode may instead pin it explicitly
with an optional top-level `baseline_villain_strategy`
(`{ villain_info_set: { action: probability } }`). An explicit baseline Villain
is a **chosen comparison profile, not an equilibrium claim**: it need not be a
best response to baseline Hero, and it asserts no optimality or profitability --
it only fixes the reference profile the baseline value and the candidate
comparison are measured against. The build records the origin as
`baseline_villain_source` (`explicit` or `auto_best_response`), and the scenario
SHA-256 in the run manifest captures the exact input. The GUI/form editors do
not carry this field and reject scenarios that use it, so edit it directly in the
JSON. See `docs/scenario_format_reference.md` section 3a for the full field spec.

To run a scenario all the way through the candidate-analysis pipeline and print
the Markdown summary, use `run_river_scenario_analysis` or
`python scripts/run_river_scenario_analysis.py <scenario.json>`. That script can
also save the result to files with `--output-json`, `--output-markdown`, and
`--output-csv` (each creates missing parent directories and overwrites an
existing file); without them it prints to stdout only. By default the JSON may
contain `Infinity` for a non-finite value; pass `--strict-json` for
RFC 8259-compatible JSON that maps non-finite floats to `null` (recommended when
the output is read by JavaScript `JSON.parse`).

To compare several scenarios at once, use `run_batch_scenario_analysis` or
`python scripts/run_scenario_batch.py <dir-or-files>`, which runs the same
single-scenario analysis on each input (a directory's `*.json` in filename order,
or the given files in order) and prints one comparison row per scenario. It also
takes `--output-json`, `--output-csv`, `--output-markdown`, and `--strict-json`,
plus `--continue-on-error` to record failing scenarios instead of stopping. The
batch CSV and Markdown are meant for comparing scenarios side by side: the
Markdown report has an overview (total / ok / error counts), a comparison table
(model kind, horizon, candidate counts, and the top-ranked candidate columns),
and a short notes section, while the CSV stays machine-friendly. The
`top_candidate_*` columns are only populated when ranking is enabled (for example
`--rank-by t_deadline`), and `--strict-json` affects only the JSON export. The
batch runner is an analysis/reporting helper over the existing pipeline, not a
new solver model.

Every analysis run carries a **run manifest** for reproducibility: the SHA-256
of the scenario file (`null` when the run started from an in-memory scenario),
the scenario `format_version`, the package version, the git commit of the
package source (best effort; `null` when git or a checkout is unavailable), a
UTC timestamp, and the effective parameters (horizon, discount, response mode,
tolerances, and so on). The manifest appears as a `manifest` object in the JSON
exports (per scenario, plus a batch-level manifest with the requested overrides
in batch JSON), as a single `# run_manifest: {...}` comment line before the
header row in the CSV exports (skip `#` lines when parsing, for example pandas
`comment="#"`), and as a `### Run manifest` section in the Markdown exports.
It is descriptive metadata only and changes no analysis result; see
`docs/scenario_format_reference.md` for the field list.

To check that a scenario JSON is well formed *before* running any analysis, use
`validate_river_scenario_inputs` or
`python scripts/validate_river_scenario.py <dir-or-files>`. It loads, parses, and
builds the game for each input (a directory's `*.json` in filename order, or the
given files in order) and prints one row per scenario with its `scenario_id`,
derived `model_kind`, information-set / terminal counts, and an `ok`/error flag.
Unlike the analysis and batch runners it stops at the parser/build level: it does
*not* generate candidates, run the exact-response solver, or run the analysis
pipeline. A bad file reports a short `error: ...` line instead of a Python
traceback; pass `--continue-on-error` to record failing files and keep going,
and `--output-json` (optionally with `--strict-json`) to save the rows as JSON.

### STT push/fold ICM scenario input

The STT path uses a separate JSON format, `stt_pushfold-1`, for preflop
SB-vs-BB push/fold spots in a single-table tournament. It is intentionally
separate from the river format because the payoff backend is ICM prize EV delta,
not chip EV or river pot share.

To run the bundled 2x2 abstract bucket example:

```bash
python scripts/run_stt_pushfold_analysis.py examples/stt_pushfold_2x2.json
```

The STT runner shares the same candidate-analysis pipeline and supports
`--output-json`, `--output-markdown`, `--output-csv`, and `--strict-json`.
Analysis values are modelled tournament prize EV deltas from
Malmuth-Harville ICM. They are not real tournament predictions, not real-money
advice, and not push/fold charts. The repeated layer assumes the same abstract
spot is repeated for sensitivity analysis; it is not a tournament simulation,
Future-ICM, or FGS model.

The STT terminal triple still uses the core field name `house_rake` for API
compatibility, but in STT reports that third slot is a bystander prize EV delta
and may be negative. See `docs/stt_pushfold_format_reference.md` for the full
field specification and validation rules.

To start from a working file instead of an empty one, generate a starter
scenario with `create_scenario_template` or
`python scripts/create_scenario_template.py --kind <kind>` (use `--list-kinds`
to see the kinds). It prints the JSON to stdout, or saves it with `--output`
(`--force` to overwrite). Every template includes `"format_version": "1"` and is
validated at the parser/build level by default. The generated templates are
abstract toy examples, not strategic recommendations: edit them, then re-check
with `python scripts/validate_river_scenario.py <file>`. Example:

```bash
python scripts/create_scenario_template.py --list-kinds
python scripts/create_scenario_template.py --kind range-matrix-equity-betting-tree --output reports/template.json
python scripts/validate_river_scenario.py reports/template.json
```

To fill in the common fields without editing JSON by hand, use the interactive
wizard `python scripts/wizard_create_scenario.py`. It starts from a template and
asks for the scenario id, description, rake, initial commitment, bet size,
repeated horizons / discount, and output path; anything passed as a flag
(`--kind`, `--output`, ...) is not asked. It validates before writing and refuses
to overwrite without `--force`. Range buckets and matrices keep the template's
toy values, so edit those in the JSON afterwards. This is the precursor to a
future GUI/form input layer.

To run the whole path in one go, use the guided workflow
`python scripts/wizard_run_scenario.py`. It either creates a scenario from a
template (`--kind`, saved to `--scenario-output`) or analyses an existing one
(`--scenario PATH`), validating it, running the analysis, printing a short
summary, and optionally saving the result with `--output-json` /
`--output-markdown` (`--strict-json` for RFC 8259 JSON). It sequences the
existing wizard / validation / analysis / export pieces, adding no new model.

```bash
python scripts/wizard_run_scenario.py --scenario examples/scenarios/nuts_chop_steal_bet98.json
python scripts/wizard_run_scenario.py --kind single-hand --scenario-output reports/my.json --non-interactive
```

Scope of the abstract range modes in v1:

- Matchup outcomes are given directly as abstract inputs: a discrete
  `showdown_matrix`, or an `equity_matrix` of Hero pot shares (for example
  precomputed by an external tool). There is no real card or hand evaluation, so
  it does not parse real cards, hand ranges, or solver exports.
- By default the JSON action tree is limited to OOP `check`/`bet` and IP
  `call`/`fold`, and an OOP `check` resolves immediately to a check-check
  showdown.

Matrix-mode scenarios may also add an optional **river betting tree v1** via a
`betting_tree` object (`oop_bet_size`, `ip_bet_after_check_size`,
`ip_raise_size`; see `examples/scenarios/range_equity_betting_tree_bet98.json`).
This adds an IP stab after an OOP check (with an OOP call/fold response) and one
IP raise versus an OOP bet (with an OOP call/fold response). In betting-tree mode
each Hero bucket supplies `baseline_strategies` for both decision points instead
of the simple `baseline_strategy`. It is still one river street only: no
re-raise, no multiple sizes per node, no nested betting trees, and no street
transitions, even though the core `GameTree` itself allows arbitrary action
labels.

### MVP walkthrough

See [docs/mvp_walkthrough.md](docs/mvp_walkthrough.md) for an end-to-end
explanation of the current minimum viable workflow and how to read the pipeline
output.

The original nuts-chop steal example is covered by a regression test and
summarized in the MVP walkthrough.

### Assumptions and limitations

See [docs/assumptions_and_limitations.md](docs/assumptions_and_limitations.md)
for the modelling assumptions, interpretation limits, and responsible-publication
notes.

### Examples guide

See [docs/examples_guide.md](docs/examples_guide.md) for a guide to the example
scripts and the recommended order for running them.

### Scenario format reference

See [docs/scenario_format_reference.md](docs/scenario_format_reference.md) for
the full JSON scenario format: every top-level field, each input mode and its
required fields, the information-set naming, payoff conventions, and validation
troubleshooting.

See [docs/stt_pushfold_format_reference.md](docs/stt_pushfold_format_reference.md)
for the separate STT SB-vs-BB push/fold JSON format and its ICM prize-EV
accounting conventions.

See [docs/baseline_solution_import_format.md](docs/baseline_solution_import_format.md)
for the shared v1 contract that treats existing river and STT baseline strategy
fields as scenario-native profile imports, not raw solver-export imports.

### GUI/form input design

See [docs/gui_input_design.md](docs/gui_input_design.md) for the design of the
GUI/form input layer over the existing CLI workflow (screens, MVP scope,
validation and results UX, implementation phases, and the per-mode details of
the current prototypes). Local-only prototype GUIs exist for all five scenario
modes (see the table below). The GUI-independent
building blocks live in `repeated_poker.scenario_form`: `SingleHandScenarioForm`,
`HeroRangeScenarioForm`, `ShowdownMatrixScenarioForm`, `EquityMatrixScenarioForm`,
and `BettingTreeScenarioForm` (each with `*_form_from_dict` / `*_form_to_dict` /
`validate_*_form`), form <-> JSON bridges with field-level validation messages
covering all five scenario modes (single-hand, Hero-range-only, discrete
showdown-matrix, equity-matrix, and river betting-tree).

The local GUI prototypes (standard library only, bound to `127.0.0.1`) are:

| Scenario mode | Command | Port | Editor | Analyze |
|---|---|---|---|---|
| single-hand | `python scripts/serve_single_hand_gui.py` | 8000 | yes | yes |
| Hero-range-only | `python scripts/serve_hero_range_gui.py` | 8001 | yes | yes |
| showdown-matrix | `python scripts/serve_showdown_matrix_gui.py` | 8002 | yes | yes |
| equity-matrix | `python scripts/serve_equity_matrix_gui.py` | 8003 | yes | yes |
| river betting-tree | `python scripts/serve_betting_tree_gui.py` | 8004 | yes | yes |

All five prototypes share the same shape and limits. Each uses `--port`
(defaulting to the value above) and `--host` (default `127.0.0.1`), is built on
the standard library only (no framework), and makes no external calls. Each
loads a scenario JSON of its mode into an editable form, validates it, saves
only after the form validates and a parser/build round-trip succeeds (with an
overwrite checkbox and a strict-JSON option), and runs the analysis from the
current form values via a local `/api/analyze` endpoint (horizon and discount
overrides plus a Markdown toggle; the result shows the candidate counts, the
resolved horizon/discount, and the Markdown summary). Everything stays
local-only and abstract: graphing, real-card equity, external solver imports,
result persistence, public serving, and any new solver or model are out of
scope. The GUI surface is currently frozen (bug fixes only) while the research
core is prioritised; per-mode editor details live in the design document
linked above.

To exercise the form models from the command line (a lower-level alternative to
the GUIs), run
`python scripts/inspect_scenario_form.py <scenario.json>`. It detects the mode,
runs that mode's form `from_dict` / `validate` / `to_dict`, and re-parses and
rebuilds the result, printing a short report (mode, form class, validation, and
round-trip status). It is an inspect-only developer utility: it never edits the
scenario and adds no analysis.

To write the round-tripped scenario back out (a stand-in for a future GUI
"save"), run `python scripts/roundtrip_scenario_form.py <scenario.json>
[--output PATH|-] [--force] [--strict-json]`. It loads the form, and only when
the form validates cleanly and the `to_dict` output re-parses and rebuilds, emits
that JSON -- to `--output PATH` (refusing to overwrite without `--force`) or, by
default or with `--output -`, to stdout as JSON only. `--strict-json` reuses the
same RFC 8259 serialiser as the analysis exporters.

To edit fields and save (the smallest "form edit -> save" flow, single-hand mode
only), run `python scripts/edit_scenario_form.py <scenario.json> --set
FIELD=VALUE [...] [--output PATH|-] [--force] [--strict-json]`. Each `--set`
updates one flat `SingleHandScenarioForm` field (for example `bet_size=50`,
`horizons=10,20`, `shift_amounts=0.25,0.5,1.0`, `rake_cap=none`), with dotted
aliases such as `rake.rate` and `baseline.call`; the result is only written when
the edited form validates and round-trips. Non-single-hand scenarios, unknown
fields, and bad values are rejected with a clean `error:` message. The three form
CLIs compose: `edit` -> stdout -> `inspect` / `roundtrip`.

### Public readiness

See [docs/public_readiness_checklist.md](docs/public_readiness_checklist.md)
before changing repository visibility from private to public.

### License and publication policy

This project is released under the MIT License. See [LICENSE](LICENSE) and
[docs/publication_policy.md](docs/publication_policy.md) for the publication
posture and wording guidelines.

### MVP check script

Run `python scripts/check_mvp.py` before opening a PR or sharing the MVP. It runs
the test suite and the key examples. The script uses only the Python standard
library and has no network or file-output side effects; the commands it runs may
read the local git commit while building run manifests (never writing
version-control state).

### Ranking report rows

`rank_candidate_rows` can sort analysis report rows by diagnostic criteria (for
example, post-response worst-case Hero EV difference, baseline-Villain EV, L1
distance, `T_deadline`, or local `T_detect`). It is not automatic strategy
selection and does not claim optimality.
