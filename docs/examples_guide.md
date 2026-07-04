# Examples Guide

## Purpose

- Explain what each example under `examples/` demonstrates.
- Help readers choose the right example to run first.
- Clarify that the examples are small abstract demonstrations, not real poker
  recommendations.

## Recommended order

1. `examples/nuts_chop_river.py` - start here: the smallest benchmark, where
   every showdown is a chop, so the accounting is easy to check by hand.
2. `examples/candidate_library.py` - generate candidates and compare them
   against a baseline Villain.
3. `examples/candidate_filters.py` - prune candidates before comparison and read
   the exclusion reasons.
4. `examples/analysis_pipeline.py` - the most important example for an MVP check:
   it wires every stage together end to end.
5. `examples/ranking.py` - sort the pipeline's report rows by diagnostic criteria.
6. `examples/markdown_summary.py` - render an existing report as Markdown.
7. `examples/analysis_report.py` - build the consolidated report directly, at a
   lower level than the pipeline.
8. `examples/value_bluff_river.py` - a contrasting non-chop spot.

The order moves from the simplest hand-checkable tree, through candidate
generation / filtering, to the full pipeline, and finally to presentation and
contrast examples. For a quick MVP sanity check, `analysis_pipeline.py` is the
single most important example to run.

## Example reference

### `examples/nuts_chop_river.py`

- **What it demonstrates**: the true nuts-chop river benchmark, where any legal
  holding chops at showdown, plus the exact Villain best response to a fixed
  Hero strategy.
- **Why it matters**: it is small enough to verify by hand, so it anchors trust
  in the engine.
- **What to look for**: the best-response EVs and the rake accounting; with zero
  rake the payoffs are zero-sum.
- **Limitations**: a single tiny abstract tree, not a real board or range.

### `examples/value_bluff_river.py`

- This is the former nuts-chop example, now separated as a value/bluff example.
- Shows a non-chop, value/bluff-like abstract spot where showdowns are not
  always tied.
- Useful as a contrast to the true nuts-chop benchmark.
- It is not the main repeated-game MVP example.

### `examples/candidate_library.py`

- Shows candidate generation and comparison against a baseline Villain strategy.
- Useful to understand candidates before running the full pipeline.

### `examples/candidate_filters.py`

- Shows pre-comparison pruning of generated candidates.
- Explains the kept/excluded split and the English exclusion reasons.

### `examples/analysis_report.py`

- Shows consolidated report building directly.
- More detailed but lower-level than the pipeline.

### `examples/markdown_summary.py`

- Shows presentation-only Markdown rendering of an analysis report.
- Does not change analysis results.

### `examples/analysis_pipeline.py`

- The recommended MVP entry point.
- Wires generation, filtering, comparison, report, detection, and the Markdown
  summary into one call.
- The best example for an end-to-end sanity check.

### `examples/ranking.py`

- Sorts the pipeline's analysis report rows by diagnostic criteria (for example,
  post-response worst-case Hero EV difference or local `T_detect`).
- Diagnostic / presentation only: it does not auto-select a candidate and makes
  no optimality claim.

## How to run examples

The examples need both `src` and `examples` on the Python path (some reuse the
nuts-chop tree and baseline strategies defined in other example modules).

```powershell
python examples/nuts_chop_river.py
python examples/candidate_library.py
python examples/candidate_filters.py
python examples/analysis_pipeline.py
python examples/ranking.py
python examples/markdown_summary.py
python examples/analysis_report.py
python examples/value_bluff_river.py
```

## How to interpret common output fields

- **generated**: number of candidates produced before filtering.
- **kept**: candidates that survived the pre-filter and are compared.
- **excluded**: candidates removed by the pre-filter.
- **exclusion reasons**: English codes explaining why a candidate was removed
  (for example `l1_distance_exceeds_limit`, `required_observations_below_limit`).
- **fixed_hero_ev**: Hero EV when the candidate is locked and Villain keeps the
  baseline strategy (no best response yet).
- **post_response_hero_ev_worst**: Hero EV after Villain plays its worst-for-Hero
  exact best response to the locked candidate.
- **t_deadline**: the latest opportunity at which Villain may adapt while the
  locked policy stays at least as valuable as baseline (or `-` when none).
- **t_detect_estimated_opportunities**: estimated opportunities before the
  candidate is statistically distinguishable from baseline at its information
  set (only when an occurrence probability is supplied).
- **detected_adaptation_is_at_least_baseline**: whether Hero is at least at
  baseline EV at the estimated detection timing (the economic read).
- **observation_distance**: the observable-distribution (total-variation)
  distance between the baseline and candidate Hero action distributions at the
  changed information set(s) (the maximum over sets for a multi-shift candidate).
  It is an observable distance, distinct from the strategy-space `l1_distance`.
- **is_ev_observation_deadline_pareto_candidate**: whether the candidate is on the
  trade-off Pareto frontier over `post_response_hero_ev_worst` (higher better),
  `observation_distance` (lower better), and `t_deadline` (higher better). It is a
  descriptive trade-off surface over all candidates, not a selection filter and
  not an equilibrium or optimality claim.
- **shifts** (JSON) / **info_sets**, **num_shifts** (CSV): the per-information-set
  breakdown of a multi-shift candidate (`candidate_generation.max_simultaneous_info_sets = 2`).
  Single-shift candidates keep the scalar `info_set` / `source_action` /
  `target_action` / `shift_amount` fields; multi-shift candidates leave those
  `null` and carry the combination here (and in `candidate_id`).

## What examples are not

- They are not real hand recommendations.
- They are not proof of profitable play.
- They are not full solver outputs.
- They do not import real ranges.
- They do not model full opponent adaptation.

## JSON scenario input

Instead of a hand-written Python example, an abstract river spot can be loaded
from a JSON file. See `examples/scenarios/nuts_chop_steal_bet98.json` for the
BET=98 nuts-chop steal case. For the full field-by-field specification of every
mode, see [scenario_format_reference.md](scenario_format_reference.md).

The scenario JSON format is currently version `"1"`. New files should include a
top-level `"format_version": "1"` (all the bundled samples now do); the field is
optional for backward compatibility, so a file without it is read as `"1"`, while
any unknown version is rejected. The format is still experimental and may get a
v2. The resolved version is
surfaced in the build metadata and in the analysis, validation, and batch outputs
(JSON / CSV / Markdown / stdout), so you can tell at a glance which format a
result came from.

The input has three mutually exclusive modes:

- **single-hand mode**: a top-level `showdown` result and a single
  `baseline_hero_strategy` at the `IP_vs_bet` information set (the nuts-chop
  sample above).
- **abstract Hero range mode**: a `hero_range` list of weighted hands, each with
  its own `hand_id`, `weight`, `showdown`, and `baseline_strategy`. See
  `examples/scenarios/abstract_range_steal_bet98.json` for a two-hand range (one
  hand chops and folds at baseline, one hand wins at showdown and calls). A
  chance node draws the Hero bucket; Villain shares one `OOP_river` information
  set across all buckets (it does not see Hero's hand), while Hero gets a
  per-hand `IP_vs_bet::<hand_id>` information set.
- **abstract Hero/Villain range matrix mode**: a `hero_range` (without per-hand
  `showdown`), a `villain_range`, and exactly one matchup matrix keyed by
  `[hero_id][villain_id]`. See `examples/scenarios/range_matrix_steal_bet98.json`
  (a `showdown_matrix` of discrete results) and
  `examples/scenarios/range_equity_steal_bet98.json` (an `equity_matrix`). A
  chance node draws the `(hero, villain)` pair with probability
  `hero_weight * villain_weight`; the outcome of each pair comes from the matrix.

The difference between the two range modes is the showdown source. Hero-range-only
mode fixes one `showdown` per Hero bucket, so Villain has no private hand and
shares a single `OOP_river` information set. Matrix mode gives Villain its own
weighted buckets and resolves each Hero/Villain matchup through a matrix; Villain
then knows its own bucket (a per-bucket `OOP_river::<villain_id>` information set,
shared across Hero buckets), while Hero still knows only its own bucket
(`IP_vs_bet::<hero_id>`, shared across Villain buckets).

Matrix mode accepts either matrix, but not both:

- `showdown_matrix`: each cell is a discrete `chop`/`hero`/`villain` result.
- `equity_matrix`: each cell is Hero's **pot share before rake** in `[0, 1]`,
  where `1.0` awards Hero the whole raked pot, `0.5` is a chop, and `0.0` awards
  it all to Villain. This lets you feed an equity precomputed by an external tool
  directly into the game; the rake is still taken from the awarded pot first, so
  `hero_ev + villain_ev + house_rake == 0` holds at every terminal.

These are *abstract* ranges given directly as weights and matchup outcomes; they
are **not** a real card or hand-range parser.

### River betting tree v1

By default matrix mode uses the simple action tree (OOP `check`/`bet`, IP
`call`/`fold`, with an OOP `check` resolving immediately to a check-check
showdown). Matrix mode can instead opt into a fuller one-street betting tree by
adding a `betting_tree` object with three sizes:

- `oop_bet_size`: OOP's bet when it bets first.
- `ip_bet_after_check_size`: IP's stab after an OOP check.
- `ip_raise_size`: IP's **total** committed chips once a raise versus the OOP bet
  is called (not the raise increment), and it must exceed `oop_bet_size`.

See `examples/scenarios/range_equity_betting_tree_bet98.json`. The action tree
becomes:

```
OOP check -> IP check (check-check showdown)
          -> IP bet   -> OOP fold / OOP call (showdown)
OOP bet   -> IP fold / IP call (showdown)
          -> IP raise -> OOP fold / OOP call (showdown)
```

So an OOP `check` no longer has to be an immediate showdown: there is now one IP
stab after the check and one IP raise versus the OOP bet. The information sets
add Hero points `IP_after_OOP_check::<hero_id>` and `IP_vs_OOP_bet::<hero_id>`
and Villain points `OOP_first::<villain_id>`, `OOP_vs_IP_bet::<villain_id>`, and
`OOP_vs_IP_raise::<villain_id>`. In betting-tree mode each Hero bucket supplies
`baseline_strategies` for both Hero decision points (`after_oop_check` with
`check`/`bet`, and `vs_oop_bet` with `call`/`fold`/`raise`) instead of the simple
`baseline_strategy`. Betting-tree mode is supported with matrix mode only
(it requires `hero_range` + `villain_range` + a matrix).

What the abstract range modes do **not** do in v1:

- The `equity_matrix` is a directly supplied Hero pot share, not a computed
  equity: there is no card or hand evaluation behind it.
- Without a `betting_tree`, the action tree is fixed to OOP `check`/`bet` and IP
  `call`/`fold`, and an OOP `check` resolves immediately to a check-check
  showdown.
- The `betting_tree` is one river street only: no re-raise, no multiple sizes per
  node, no nested/arbitrary betting trees, and no street transitions, even though
  the core `GameTree` supports arbitrary action labels.
- No real card parsing, hand-range syntax, hand evaluation, or solver imports.

All three modes work with the same helper scripts and runner below.

There are two helper scripts for a scenario file:

- `scripts/run_river_scenario.py` is a quick sanity check: it prints the terminal
  EVs, the baseline and locked-call responses, the candidate count, and the
  `T_deadline` table, without running the candidate-analysis pipeline.
- `scripts/run_river_scenario_analysis.py` runs the full candidate-analysis
  pipeline (generation, comparison, reporting) and prints the Markdown summary,
  plus an optional ranking section.

```powershell
python scripts/run_river_scenario.py examples/scenarios/nuts_chop_steal_bet98.json
python scripts/run_river_scenario_analysis.py examples/scenarios/nuts_chop_steal_bet98.json
```

The analysis script accepts `--horizon`, `--discount`, `--rank-by`, `--top-k`,
and `--no-markdown`. It can also save the result to files with `--output-json`,
`--output-markdown`, and `--output-csv`:

```powershell
python scripts/run_river_scenario_analysis.py examples/scenarios/range_equity_betting_tree_bet98.json --output-json reports/result.json --output-markdown reports/result.md --output-csv reports/result.csv
```

Each output flag creates missing parent directories and overwrites an existing
file, and the script prints a short `saved ... to <path>` line per file. The JSON
payload contains the scenario id, the selected horizon/discount, the build
metadata, the candidate counts, the filter result (kept ids and excluded
candidates with reasons), the full `analysis_report.to_dict()`, the Markdown
summary, and the ranking (when `--rank-by` is given); it is written with
`json.dumps(indent=2)`. By default Python's standard `json` module may serialise
a non-finite KL divergence as `Infinity`, which is not strict RFC 8259 JSON and
which JavaScript `JSON.parse` rejects. Pass `--strict-json` (or `strict=True` to
`write_analysis_json` / `write_batch_json`) to emit RFC 8259-compatible JSON
instead, which maps every non-finite float (`inf` / `-inf` / `nan`) to `null`;
prefer it when the output is consumed by `JSON.parse`. `--strict-json` affects
only the JSON output, not Markdown or CSV. The CSV has one row per candidate with
the main selection/deadline/detection columns. By convention these go under
`reports/`, which is git-ignored. `--output-markdown` forces Markdown generation,
so it overrides `--no-markdown` for the saved file (`--no-markdown` then only
suppresses the stdout summary).

From Python the same run is available as `run_river_scenario_analysis`, and the
exporters are `write_analysis_json` / `write_analysis_markdown` /
`write_analysis_csv`:

```python
from repeated_poker import run_river_scenario_analysis, write_analysis_json

result = run_river_scenario_analysis(
    "examples/scenarios/nuts_chop_steal_bet98.json"
)
print(result.markdown_summary)
write_analysis_json(result, "reports/result.json")
```

### Comparing several scenarios

The single-scenario runner above analyses one file. To compare several at once,
use the batch runner, which runs the same per-scenario analysis on each input and
emits one comparison row per scenario:

```powershell
python scripts/run_scenario_batch.py examples/scenarios
python scripts/run_scenario_batch.py examples/scenarios/nuts_chop_steal_bet98.json examples/scenarios/abstract_range_steal_bet98.json --rank-by t_deadline --output-json reports/batch.json --output-csv reports/batch.csv --output-markdown reports/batch.md
```

A single directory argument reads its `*.json` in filename order; multiple
arguments are read in the given order. The batch CSV/Markdown are meant for
comparing scenarios side by side: they carry one row per scenario (scenario id,
model kind, horizon, discount, candidate counts including
`minimum_villain_ev_candidates`, the top-ranked candidate's id / `t_deadline` /
worst-case post-response Hero-EV difference / detected-adaptation flag, and any
`error`). The Markdown report adds an overview (total / ok / error counts) and a
short notes section; the CSV stays machine-friendly (`true`/`false`, empty cells
for missing values, raw float strings). The batch JSON also embeds each
successful scenario's full result. The `top_candidate_*` columns are only
populated when ranking is enabled (for example `--rank-by t_deadline`), and the
detected-adaptation flag only when detection output exists; otherwise those cells
show `-` in Markdown. `--continue-on-error` records a failing scenario's error on
its row instead of stopping, and `--strict-json` affects only the JSON export
(not the Markdown or CSV). From Python this is `run_batch_scenario_analysis` with
`write_batch_json` / `write_batch_csv` / `write_batch_markdown`. Like the rest of
the scenario tooling, the batch runner is an analysis/reporting helper over the
existing pipeline, not a new solver model.

### Validating scenarios before analysis

When a scenario JSON grows complex it helps to confirm it is well formed *before*
running any analysis. The validation runner loads, parses, and builds the game
for each input but stops there -- it does **not** generate candidates, run the
exact-response solver, or run the analysis pipeline, so it is fast and never
prints a Python traceback:

```powershell
python scripts/validate_river_scenario.py examples/scenarios
python scripts/validate_river_scenario.py examples/scenarios/nuts_chop_steal_bet98.json examples/scenarios/range_equity_betting_tree_bet98.json
python scripts/validate_river_scenario.py examples/scenarios --output-json reports/validation.json --strict-json
```

A single directory argument reads its `*.json` in filename order; multiple
arguments are read in the given order. Each row reports the `source_path`, an
`ok` flag, the `scenario_id`, the derived `model_kind` (for example `single_hand`,
`range`, or `range_matrix:equity+betting_tree`), the Hero/Villain information-set
counts, the terminal count, and the chance-outcome count. A bad file prints a
short `error: <type>: <message>` line instead of a traceback; `--continue-on-error`
records the failing file and keeps going, and `--output-json` (optionally with
`--strict-json`) saves the rows as JSON. From Python this is
`validate_river_scenario_inputs` with `write_validation_json`. Use this to catch
input mistakes; use `run_river_scenario_analysis` / `run_scenario_batch.py` once
the inputs validate to actually analyse and compare candidates.

### Generating a starter scenario

To avoid writing a scenario JSON from scratch, generate a starter template with
`create_scenario_template` or `scripts/create_scenario_template.py`. Each
template is an abstract toy example (not a strategic recommendation), includes
`"format_version": "1"`, and is validated at the parser/build level by default
(pass `--no-validate` to skip). It prints to stdout, or saves with `--output`
(refusing to overwrite an existing file unless `--force` is given):

```powershell
python scripts/create_scenario_template.py --list-kinds
python scripts/create_scenario_template.py --kind range-matrix-equity-betting-tree --output reports/template.json
python scripts/validate_river_scenario.py reports/template.json
```

The kinds are `single-hand`, `hero-range`, `range-matrix-showdown`,
`range-matrix-equity`, and `range-matrix-equity-betting-tree`. Generated files
are meant to be edited and re-validated; for the full field specification see
[scenario_format_reference.md](scenario_format_reference.md). From Python this is
`create_scenario_template(kind)` (see `available_scenario_template_kinds()`).

For a guided version of the same thing, run the interactive wizard:

```powershell
python scripts/wizard_create_scenario.py
python scripts/wizard_create_scenario.py --kind single-hand --output reports/my_scenario.json
```

The wizard starts from a template and prompts for the common top-level fields
(scenario id, description, rake, initial commitment, bet size, repeated horizons
/ discount, and the output path); an empty answer keeps the template default, and
anything passed as a flag is not asked. It validates before writing and will not
overwrite an existing file without `--force`. Range buckets and matrices keep the
template's toy values, so edit those in the JSON afterwards. The wizard is the
precursor to a future GUI/form input layer, which is not implemented yet.

### Guided end-to-end workflow

To walk the whole path (create or pick a scenario, validate, analyse, and
optionally export) in one command, use the guided workflow:

```powershell
python scripts/wizard_run_scenario.py --scenario examples/scenarios/nuts_chop_steal_bet98.json
python scripts/wizard_run_scenario.py --kind single-hand --scenario-output reports/my_scenario.json --non-interactive
python scripts/wizard_run_scenario.py --scenario examples/scenarios/nuts_chop_steal_bet98.json --output-json reports/result.json --strict-json --output-markdown reports/result.md
```

It has two modes: *create-and-run* (`--kind`, building a starter scenario via the
wizard helpers and saving it to `--scenario-output`) and *existing-file*
(`--scenario PATH`, which it does not modify). Both validate at the parser/build
level, run `run_river_scenario_analysis`, print a short summary (scenario id,
validation ok, horizon / discount, candidate counts), and can save the analysis
with `--output-json` (optionally `--strict-json`) and `--output-markdown`. It
only sequences the existing wizard / validation / analysis / export pieces, so it
adds no new solver or game-theory model, and errors are reported as a short
`error: ...` line rather than a traceback.

To stop at the building blocks instead, build the game and feed it into the
pipeline yourself:

```python
from repeated_poker import (
    build_river_steal_game_from_scenario,
    generate_shift_candidates,
    load_river_scenario_json,
)

build = build_river_steal_game_from_scenario(
    load_river_scenario_json("examples/scenarios/nuts_chop_steal_bet98.json")
)
candidates = generate_shift_candidates(
    build.tree, build.baseline_hero_strategy, build.shift_amounts
)
```

`build.tree`, `build.baseline_hero_strategy`, and `build.baseline_villain_strategy`
are also ready to pass into `run_candidate_analysis_pipeline`. This v1 input is
an abstract spot only; it does not parse real cards, hand ranges, or solver
exports.

## Related docs

- [MVP Walkthrough](mvp_walkthrough.md)
- [Assumptions and Limitations](assumptions_and_limitations.md)
