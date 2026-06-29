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

### Scenario form model (supported modes)

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
  precursor to the matrix editor.
- Equity-matrix (v1): `EquityMatrixScenarioForm`, the other matrix flavour. It
  reuses the same `HeroMatrixBucketForm` / `VillainMatrixBucketForm` buckets, with
  an `equity_matrix` grid whose cells are the Hero pot share before rake (a finite
  number in `[0, 1]`), via `equity_matrix_form_from_dict` /
  `equity_matrix_form_to_dict` / `validate_equity_matrix_form`. The bucket,
  disjoint-id, and matrix-completeness checks are shared with the showdown-matrix
  form; only the cell rule differs (a number in `[0, 1]`, with field name
  `equity_matrix[hero_id][villain_id]`).
- Betting-tree (v1): `BettingTreeScenarioForm` for the river one-street betting
  tree, via `betting_tree_form_from_dict` / `betting_tree_form_to_dict` /
  `validate_betting_tree_form`. It reuses `VillainMatrixBucketForm` and adds a
  `BettingTreeSizingForm` (`oop_bet_size` / `ip_bet_after_check_size` /
  `ip_raise_size`) and `HeroBettingTreeBucketForm`, whose two Hero decision
  points are split into flat probability fields (`after_oop_check`: check/bet;
  `vs_oop_bet`: call/fold/raise). `matrix_type` (`"showdown"` / `"equity"`)
  selects how `matrix` cells are read, reusing the shared matrix-grid check. The
  validator adds the size rules (`ip_raise_size` > `oop_bet_size`), the
  `bet_size` == `oop_bet_size` match, and the two per-bucket distributions. It is
  the precursor to the betting-tree settings screen; arbitrary / nested tree
  editing stays out of scope.

Every `from_dict` reuses the existing JSON parser (so no parsing is duplicated),
and a valid form's `to_dict` output is accepted by the parser and the game
builder. All five JSON scenario modes now have a form model.

`repeated_poker.detect_scenario_form_mode(data)` returns the mode label for a
scenario dict (one of `repeated_poker.SCENARIO_FORM_MODES`), so a loader can pick
the right form without duplicating the mode rules. Before any GUI exists, the
inspect-only CLI `scripts/inspect_scenario_form.py <scenario.json>` uses it to
load a scenario through its form model and report the mode, validation messages,
and round-trip parse/build status -- a quick way to exercise the form layer end
to end from the command line.

`scripts/roundtrip_scenario_form.py <scenario.json> [--output PATH|-] [--force]
[--strict-json]` is the writer counterpart: a stand-in for the GUI "save" step. It
loads the form and, only when it validates cleanly and the `to_dict` output
re-parses and rebuilds, writes that JSON to a file (refusing to overwrite without
`--force`) or to stdout. It reuses the same strict-JSON serialiser as the analysis
exporters, so a GUI save and a CLI save would share one format.

`scripts/edit_scenario_form.py <scenario.json> --set FIELD=VALUE [...]
[--output PATH|-] [--force] [--strict-json]` is the smallest "form edit -> save"
flow, restricted to single-hand mode: it loads the `SingleHandScenarioForm`,
applies `--set` edits to flat fields (with dotted aliases like `rake.rate` /
`baseline.call`), and writes only when the edited form validates and round-trips.
It approximates the GUI's per-field edit and save path. The three form CLIs
(inspect, roundtrip, edit) reuse one loader, mode detection, safe writer, and
strict-JSON serialiser, and compose as `edit -> inspect / roundtrip`.

`scripts/serve_single_hand_gui.py [--host 127.0.0.1] [--port 8000]` is the first
actual GUI: a local-only browser prototype of the single-hand edit-and-save flow,
built on the standard library (`http.server` plus inline HTML / CSS / vanilla
JavaScript -- no framework or dependency). It serves a form page (`GET /`) and a
small JSON API (`POST /api/load`, `/api/validate`, `/api/save`) that reuse the
same `SingleHandScenarioForm` helpers, field value parsing, safe writer, and
strict-JSON serialiser as the CLIs. It is deliberately tiny and single-hand only:
the screens, MVP scope, and phases above describe where a fuller GUI would go
(other modes, the matrix editor, analysis, and export), all still future work.
Safety: it binds to `127.0.0.1`, makes no external calls, reads/writes only
user-supplied paths, refuses to overwrite without the overwrite box, and returns
short error messages rather than tracebacks.

The prototype also runs the analysis locally from the current form (`POST
/api/analyze`, the **Analyze** button): it validates the form and calls
`run_river_scenario_analysis`, returning the candidate counts (generated / kept /
excluded), the resolved horizon and discount, and the Markdown summary (shown as
plain text). Optional `horizon` / `discount` overrides are validated (a positive
int / a finite positive number) and `render_markdown` must be a boolean. This is
the "analyze and view a summary" step of the screen flow; charts, export beyond
the existing save, and the other modes remain future work.

A small UX-polish pass exposes those analyze options in the page -- a horizon
override and a discount override (blank for the scenario default) and a "render
Markdown summary" toggle -- and separates the analysis result (scenario id,
horizon/discount, and the generated/kept/excluded counts on one line, plus the
summary `<pre>`) from the status line and validation messages. It is still a
local prototype and single-hand only, with no graphing or multi-mode editing.

`scripts/serve_hero_range_gui.py [--host 127.0.0.1] [--port 8001]` is the next
mode's editor: a local-only browser prototype of the Hero-range-only
load / edit-buckets / validate / save flow. It serves a page (`GET /`) with the
top-level fields plus a table of weighted Hero buckets (Add hand / Remove per
row) and the same `POST /api/load`, `/api/validate`, `/api/save` API, reusing
`HeroRangeScenarioForm` / `validate_hero_range_form` / `hero_range_form_to_dict`,
the shared loader and safe writer, and the same safety rules (localhost only, raw
`format_version`, boolean save options, `textContent` display, no tracebacks). It
rejects non-Hero-range scenarios. It does not run the analysis pipeline yet, and
the matrix and betting-tree editors and graphing remain future work.

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
