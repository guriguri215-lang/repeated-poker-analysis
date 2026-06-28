# GUI/form input design

This is a design document only. No GUI code, web app, server, framework, or new
dependency is introduced here. It plans how a future GUI/form input layer would
let a user who does not know the JSON format still build a scenario, validate it,
run the analysis, and export the result. It builds on the existing JSON scenario
format (see [scenario_format_reference.md](scenario_format_reference.md)) and the
existing command-line workflow; it is the screen-level design of that workflow,
not a new solver or game-theory model.

## 1. Purpose

- Let a non-JSON user create, validate, analyse, and export a scenario through a
  guided form instead of hand-editing JSON.
- Turn the existing CLI workflow into a screen flow: the same steps
  (create / pick -> validate -> analyse -> export), exposed as forms and panels.
- Decide, before any GUI implementation, what belongs in an MVP GUI and what is
  deferred, so the first implementation stays small.
- Keep JSON as the source of truth: the GUI is an editor/runner on top of the
  format, not a replacement for it.

## 2. Non-goals

- The GUI implementation itself (this document is design only).
- Choosing a web app, server, or UI framework, or adding any dependency.
- A real-card parser (parsing real hole cards / boards into ranges).
- Importing solver outputs or proprietary formats (no external solver import).
- Being a commercial poker solver.
- Bankroll, gambling, or real-money advice. The analysis produces abstract model
  outputs only.

## 3. Current CLI workflow mapping

The GUI screens are derived from the existing scripts and APIs. Nothing new is
required on the engine side for the MVP.

| CLI piece | Role today | GUI screen / action |
| --- | --- | --- |
| `scripts/create_scenario_template.py` | Emit a starter template for a kind | Start / mode selection -> "new from template" |
| `scripts/wizard_create_scenario.py` | Prompt for common top-level fields | Scenario setup + game parameter forms |
| `scripts/wizard_run_scenario.py` | Create-or-pick, validate, analyse, export | The overall screen flow (the workflow backbone) |
| `scripts/validate_river_scenario.py` | Parser/build validation, no analysis | Validation panel |
| `scripts/run_river_scenario_analysis.py` | Run the analysis pipeline | Analysis run panel + Results summary |
| `report_export` (`write_analysis_json` / `write_analysis_markdown`) | Save results | Export panel |

## 4. Proposed GUI screens

A first cut of the screens, in roughly the order a user would move through them:

- Start / mode selection: new scenario from a template kind, or open an existing
  scenario JSON.
- Scenario setup: `scenario_id`, `description`, and the chosen mode/kind
  (`format_version` shown read-only as `"1"`).
- Rake / pot / sizing: rake rate and cap, initial commitment (hero / villain),
  bet size (hidden in betting-tree mode).
- Range buckets: hero (and villain, in matrix modes) buckets with weights and
  baseline strategies.
- Showdown / equity matrix editor: the Hero x Villain grid for
  `showdown_matrix` or `equity_matrix`.
- Betting tree settings: the betting-tree sizes (deferred; see MVP scope).
- Candidate generation / repeated settings: `shift_amounts`, `horizons`,
  `discount`.
- Validation panel: run parser/build validation and show pass/fail with
  per-field messages.
- Analysis run panel: trigger the analysis (with horizon / discount overrides).
- Results summary: candidate counts and the key per-candidate diagnostics.
- Export panel: Markdown / JSON export, with a strict-JSON option.

## 5. MVP GUI scope

The first GUI should be deliberately small:

- Modes in the MVP: `single-hand` and `hero-range`, plus exactly one matrix mode
  (`range-matrix-showdown` or `range-matrix-equity`; pick one to start).
- Betting-tree mode is deferred. As an interim step it may be offered only as a
  read-only template starting point (the user edits the JSON by hand), not a full
  editor.
- The batch runner is deferred; the GUI MVP analyses one scenario at a time.
- Strict JSON export is a single checkbox on the export panel.
- Detection / filtering / ranking options are deferred or shown as advanced/
  collapsed fields with safe defaults.

## 6. Field grouping

Forms group the JSON fields so a user fills related values together. Grouping
follows [scenario_format_reference.md](scenario_format_reference.md).

- Basic fields: `scenario_id`, `description`, `format_version` (read-only `"1"`),
  mode/kind.
- Game parameters: rake rate / cap, initial commitment (hero / villain), bet
  size(s).
- Range fields: hero buckets, villain buckets, weights, baseline strategies.
- Matrix fields: `showdown_matrix` or `equity_matrix` (exactly one).
- Analysis fields: `shift_amounts`, `horizons`, `discount`, and (deferred /
  advanced) detection, filtering, and ranking options.
- Export fields: Markdown, JSON, strict JSON.

## 7. Validation UX

- Run the same parser/build validation the CLI uses; do not run analysis until
  validation passes.
- Field-level validation mirrors the format rules: rake rate in `[0, 1]`, cap
  non-negative or empty, commitments non-negative, bet size positive, discount in
  `(0, 1]`, horizons positive integers.
- Range checks: bucket weights sum to one; hand ids unique (and hero/villain ids
  disjoint in matrix modes).
- Matrix completeness: the matrix must cover exactly the Hero x Villain id grid;
  highlight missing or unknown cells.
- Error display: show a clear message next to the offending field, plus a summary
  in the Validation panel. Reuse the format's existing error messages where
  possible rather than inventing new wording.

## 8. Results UX

The Results summary shows the analysis outputs already produced by the pipeline:

- candidate count (generated), and kept / excluded / eligible counts.
- post-response Hero EV (worst-case) and its difference from baseline.
- `T_deadline` per candidate.
- `T_detect` / detected-adaptation fields when detection output is present.
- A persistent caveat that these are abstract model outputs
  (not real-money advice) and do not guarantee profitable play.

## 9. Data flow

- GUI state -> scenario JSON -> validation -> analysis pipeline -> report export.
- JSON remains the source of truth: the GUI reads and writes the same scenario
  JSON the CLI uses, so a scenario can move between the GUI, the CLI, and a hand
  edit without loss.
- The GUI must be able to import an existing scenario JSON and export the current
  form state as JSON at any time.

### Scenario form model (single-hand and hero-range)

A first slice of this data flow already exists as a small, GUI-independent layer
in `repeated_poker.scenario_form`. Each supported mode has a flat dataclass plus
a form <-> JSON bridge and a field-level validator that returns
`FormValidationMessage` items for display instead of raising:

- single-hand: `SingleHandScenarioForm` with `single_hand_form_from_dict` /
  `single_hand_form_to_dict` / `validate_single_hand_form`.
- Hero-range-only: `HeroRangeScenarioForm` (a list of `HeroRangeHandForm`
  buckets) with `hero_range_form_from_dict` / `hero_range_form_to_dict` /
  `validate_hero_range_form`. The validator uses per-hand field names such as
  `hands[0].hand_id` and `hands[1].weight`, and the form is the precursor to a
  range bucket editor.
- Showdown-matrix (v1): `ShowdownMatrixScenarioForm` (lists of
  `HeroMatrixBucketForm` and `VillainMatrixBucketForm`, plus a
  `showdown_matrix` grid) with `showdown_matrix_form_from_dict` /
  `showdown_matrix_form_to_dict` / `validate_showdown_matrix_form`. Hero buckets
  carry a baseline call/fold split but no per-hand `showdown` (the outcome comes
  from the matrix); Villain buckets are weighted ids only. The validator uses
  field names such as `hero_buckets[0].hand_id`, `villain_buckets[1].weight`, and
  `showdown_matrix[hero_id][villain_id]`, checks Hero/Villain ids are disjoint,
  and reports matrix completeness (missing/unknown rows and cells). This is the
  precursor to the matrix editor. The discrete `equity_matrix` flavour and the
  betting-tree mode are not yet covered.

Every `from_dict` reuses the existing JSON parser (so no parsing is duplicated),
and a valid form's `to_dict` output is accepted by the parser and the game
builder. The equity-matrix and betting-tree modes are future work.

## 10. Implementation phases after this doc

The implementation phases below are deliberately incremental, so each step is
small and testable:

- Phase 1: a static form prototype or a CLI-backed local UI mock (no engine
  changes), to validate the screen flow.
- Phase 2: single-hand mode input/edit, wired to validation.
- Phase 3: range bucket editor (hero, then villain).
- Phase 4: matrix editor (showdown or equity).
- Phase 5: analysis result viewer (Results summary).
- Phase 6: export integration (Markdown / JSON / strict JSON).
- Phase 7: batch and advanced options (betting tree, detection / filtering /
  ranking).

## 11. Open questions

- Local desktop GUI vs a simple local web UI (no external server).
- Whether to adopt a framework later, and which, kept out of scope here.
- How to handle large matrices (entry, display, and validation performance).
- Whether the GUI should allow raw JSON editing alongside the forms.
- How to handle future scenario format changes (a `"2"` format version) in the
  GUI without breaking existing `"1"` files.
