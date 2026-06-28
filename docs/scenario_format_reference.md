# Scenario Format Reference

This is the reference for the JSON scenario input consumed by
`load_river_scenario_json` / `river_scenario_from_dict` and built into a game by
`build_river_steal_game_from_scenario`. It is meant to be a single place that
describes every top-level field, every input mode, and the conventions the
analysis relies on. It is the specification followed by the scenario template
generator (`scripts/create_scenario_template.py`) and for later input-usability
work (a form/GUI input layer).

## 1. Scope and status

- Scenario format version: `"1"`.
- The format is experimental and may change in a future `"2"`.
- `format_version` is optional for backward compatibility: a file without it is
  treated as `"1"`. New files should include `"format_version": "1"`.
- This is an abstract river-game scenario format. It is
  **not a full poker solver** format: it does not parse real cards, hand ranges,
  board textures, or solver exports. Matchup outcomes (discrete results or
  equities) are supplied directly as abstract inputs.
- Validate a file before analysis with the validation CLI, for example:

  ```bash
  python scripts/validate_river_scenario.py examples/scenarios/nuts_chop_steal_bet98.json
  python scripts/validate_river_scenario.py examples/scenarios
  ```

  Validation runs at the parser/build level only (see section 7).

- To start from a working file instead of writing one by hand, generate a
  template that conforms to this reference:

  ```bash
  python scripts/create_scenario_template.py --list-kinds
  python scripts/create_scenario_template.py --kind range-matrix-equity-betting-tree --output reports/template.json
  python scripts/validate_river_scenario.py reports/template.json
  ```

  Generated templates are abstract toy examples (not strategic recommendations),
  always include `"format_version": "1"`, and are meant to be edited and
  re-validated. For a guided version that prompts for the common top-level
  fields, use the interactive wizard
  `python scripts/wizard_create_scenario.py`.

## 2. Top-level fields

| Field | Required | Modes | Type | Meaning / constraints |
| --- | --- | --- | --- | --- |
| `format_version` | optional | all | string | Scenario format version. Only `"1"` is accepted; a missing field defaults to `"1"`. A numeric `1`, `null`, a bool, an empty string, or any unknown version is rejected. |
| `scenario_id` | required | all | string | Non-empty identifier for the scenario. |
| `description` | optional | all | string | Free text; defaults to `""`. |
| `rake` | required | all | object | `{ "rate": number in [0, 1], "cap": number >= 0 or null }`. `cap` is optional (no cap when omitted or `null`). |
| `initial_commitment` | required | all | object | `{ "hero": number >= 0, "villain": number >= 0 }`: chips already committed before the river decision. |
| `bet_size` | required (simple modes) / optional (betting-tree mode) | all | number > 0 | The OOP bet size for the simple action tree. In betting-tree mode it defaults to `betting_tree.oop_bet_size`; if given it must equal it. |
| `showdown` | required (single-hand) | single-hand | string | One of `"chop"`, `"hero"`, `"villain"`. Forbidden in range/matrix modes. |
| `baseline_hero_strategy` | required (single-hand) | single-hand | object | `{ "IP_vs_bet": { "call": p, "fold": p } }`; probabilities must be non-negative and sum to 1. Forbidden in range/matrix modes. |
| `hero_range` | required (range modes) | Hero-range-only, matrix | array | Weighted Hero buckets (see section 4). |
| `villain_range` | required (matrix modes) | matrix | array | Weighted Villain buckets (see section 4). |
| `showdown_matrix` | one matrix required (matrix modes) | matrix | object | `[hero_id][villain_id]` -> `"chop"` / `"hero"` / `"villain"`. Mutually exclusive with `equity_matrix`. |
| `equity_matrix` | one matrix required (matrix modes) | matrix | object | `[hero_id][villain_id]` -> Hero pot share before rake in `[0, 1]`. Mutually exclusive with `showdown_matrix`. |
| `betting_tree` | optional | matrix only | object | Adds the one-street betting tree (see the betting-tree mode below). |
| `candidate_generation` | optional (required by the analysis runner) | all | object | `{ "shift_amounts": [number > 0, ...] }` (see section 6). |
| `repeated` | optional | all | object | `{ "horizons": [int >= 1, ...] or null, "discount": number in (0, 1] }` (see section 6). |

The input mode is inferred from which fields are present, and the modes are
mutually exclusive. Mixing fields from different modes (for example a top-level
`showdown` together with a `hero_range`, or a `villain_range` without a matrix)
is rejected.

## 3. Modes

### single-hand mode

- **Required**: `showdown`, `baseline_hero_strategy` (at `IP_vs_bet`).
- **Forbidden**: `hero_range`, `villain_range`, `showdown_matrix`,
  `equity_matrix`, `betting_tree`.
- **Information sets**: Hero `IP_vs_bet`; Villain `OOP_river` (single, no suffix).
- **Actions**: OOP `check` / `bet`; IP (facing the bet) `call` / `fold`. An OOP
  `check` resolves immediately to a check-check showdown (no IP action after a
  check).
- **Sample**: [`examples/scenarios/nuts_chop_steal_bet98.json`](../examples/scenarios/nuts_chop_steal_bet98.json).

### Hero-range-only mode

- **Required**: `hero_range`, where each hand has `hand_id`, `weight`,
  `showdown`, and `baseline_strategy` (`call` / `fold`).
- **Forbidden**: top-level `showdown` / `baseline_hero_strategy`,
  `villain_range`, any matrix, `betting_tree`.
- **Information sets**: Hero gets a per-hand `IP_vs_bet::<hero_id>`; Villain
  shares one `OOP_river` across all Hero buckets (it does not observe Hero's
  hand).
- **Current limitation**: there is no Villain range here, so Villain cannot
  condition on its own bucket; for Villain buckets use matrix mode.
- **Sample**: [`examples/scenarios/abstract_range_steal_bet98.json`](../examples/scenarios/abstract_range_steal_bet98.json).

### range matrix + showdown_matrix mode

- **Required**: `hero_range` (no per-hand `showdown`), `villain_range`, and a
  `showdown_matrix`.
- **Matrix shape**: `showdown_matrix[hero_id][villain_id]` is `"chop"` /
  `"hero"` / `"villain"`, and must cover exactly the Hero x Villain id grid.
- **Bucket visibility**: Hero knows its own bucket but not Villain's; Villain
  knows its own bucket but not Hero's.
- **Information sets**: Hero `IP_vs_bet::<hero_id>` (shared across Villain
  buckets); Villain `OOP_river::<villain_id>` (shared across Hero buckets).
- **Sample**: [`examples/scenarios/range_matrix_steal_bet98.json`](../examples/scenarios/range_matrix_steal_bet98.json).

### range matrix + equity_matrix mode

- **Required**: `hero_range` (no per-hand `showdown`), `villain_range`, and an
  `equity_matrix`.
- **Equity meaning**: `equity_matrix[hero_id][villain_id]` is the Hero **pot
  share before rake** in `[0, 1]`, where `1.0` awards Hero the whole raked pot,
  `0.5` is a chop, and `0.0` awards it all to Villain.
- There is **no real card equity calculation**: the equity is an abstract input,
  for example precomputed by an external tool.
- Information sets and bucket visibility match showdown-matrix mode.
- **Sample**: [`examples/scenarios/range_equity_steal_bet98.json`](../examples/scenarios/range_equity_steal_bet98.json).

### range matrix + betting_tree mode

- **Required**: matrix mode (`hero_range` + `villain_range` + one of
  `showdown_matrix` / `equity_matrix`) plus a `betting_tree`. Each Hero hand
  carries `baseline_strategies` with two decision points instead of the simple
  `baseline_strategy`:
  - `after_oop_check`: `check` / `bet`,
  - `vs_oop_bet`: `call` / `fold` / `raise`.
- **`betting_tree` fields** (all numbers > 0):
  - `oop_bet_size`: the OOP bet size.
  - `ip_bet_after_check_size`: the IP stab size after an OOP check.
  - `ip_raise_size`: the **total** chips each player commits once IP's raise is
    called (not the raise increment); must be greater than `oop_bet_size`.
- **Action tree** (per matchup):
  - OOP acts first: `check` / `bet`.
  - After an OOP `check`: IP `check` / `bet`; if IP bets, OOP `call` / `fold`.
  - After an OOP `bet`: IP `call` / `fold` / `raise`; if IP raises, OOP
    `call` / `fold`.
- **Information sets**: Hero `IP_after_OOP_check::<hero_id>` and
  `IP_vs_OOP_bet::<hero_id>`; Villain `OOP_first::<villain_id>`,
  `OOP_vs_IP_bet::<villain_id>`, and `OOP_vs_IP_raise::<villain_id>`.
- **Not supported** (future extensions): re-raise, multiple bet sizes per node,
  arbitrary nested betting trees, street transitions, and side pots.
- **Sample**: [`examples/scenarios/range_equity_betting_tree_bet98.json`](../examples/scenarios/range_equity_betting_tree_bet98.json).

## 4. Ranges and buckets

- Each `hero_range` / `villain_range` entry is `{ "hand_id": str, "weight":
  number > 0, ... }`.
- `hand_id` must be a non-empty string and unique within its range. In matrix
  mode, Hero and Villain ids must also be disjoint from each other (so matrix
  keys and any future cross-references stay unambiguous).
- Weights within a range must sum to 1.
- A chance node draws the Hero bucket (Hero-range-only mode) or the
  `(hero, villain)` pair with probability `hero_weight * villain_weight` (matrix
  modes).
- Observation model: Hero knows its own bucket but not Villain's, and Villain
  knows its own bucket but not Hero's. This is why each side's information sets
  are keyed by that side's id and shared across the other side's buckets.
- Information-set names by mode:
  - single-hand: `IP_vs_bet`, `OOP_river`.
  - Hero-range-only: `IP_vs_bet::<hero_id>`, shared `OOP_river`.
  - matrix (simple tree): `IP_vs_bet::<hero_id>`, `OOP_river::<villain_id>`.
  - matrix + betting tree: `IP_after_OOP_check::<hero_id>`,
    `IP_vs_OOP_bet::<hero_id>`, `OOP_first::<villain_id>`,
    `OOP_vs_IP_bet::<villain_id>`, `OOP_vs_IP_raise::<villain_id>`.

## 5. Payoff conventions

- EV is hand-level net profit (chips won or lost relative to the start of the
  river decision), not a pot-size fraction.
- Rake is computed from the full showdown pot, then subtracted before the
  remaining pot is awarded or split, bounded by `rake.cap` when set. With a
  positive rake the game is **non-zero-sum** between Hero and Villain (the house
  takes a share).
- An uncalled bet is returned: when a player folds to a bet/raise, the uncalled
  amount goes back and the other player wins the folding player's committed
  chips. No rake is taken on a fold.
- `showdown_matrix` gives a discrete `chop` / `hero` / `villain` result per
  matchup; the remaining pot after rake is awarded or split accordingly.
- `equity_matrix` gives the Hero pot share **before rake** in `[0, 1]`. The rake
  is computed from the full pot and subtracted, then the remaining pot is split
  by that share (so Hero effectively receives its share of the after-rake pot).
  There is no real card evaluation behind these numbers.

## 6. Candidate generation and repeated-game fields

- `candidate_generation.shift_amounts`: a list of positive numbers. Each amount
  generates shift candidates that move probability mass between actions at a Hero
  information set, relative to the baseline strategy. The analysis runner needs
  at least one shift amount; the parser accepts a scenario without
  `candidate_generation`, but the analysis runner then has nothing to generate.
- `repeated.horizons`: a list of integers `>= 1`. The analysis runner uses the
  maximum as the default horizon unless one is passed on the command line.
- `repeated.discount`: a number in `(0, 1]` (defaults to `1.0`).
- `T_deadline` and `T_detect` are **analysis outputs**, not input solver
  guarantees: they describe when a locked policy stops being at least as good as
  baseline, and when a deviation becomes statistically distinguishable. They are
  not promises of profitable play.

## 7. Validation and troubleshooting

Run the validation CLI to confirm a file parses and builds before running any
analysis:

```bash
python scripts/validate_river_scenario.py examples/scenarios
python scripts/validate_river_scenario.py path/to/scenario.json --continue-on-error
```

Validation loads, parses, and builds the game (parser/build level). It does
**not** run candidate generation, the exact-response solver, or the full
candidate analysis pipeline, so it is a fast structural check, not a full
candidate analysis.

Common errors and their causes:

- `unsupported format_version`: `format_version` is not `"1"` (a numeric `1`,
  `null`, a bool, or an empty string also fails this check).
- weights not summing to 1: a `hero_range` / `villain_range`'s weights do not add
  up to 1.
- missing matrix entries: a `showdown_matrix` / `equity_matrix` does not cover
  exactly the Hero x Villain id grid (missing or unknown ids).
- both `showdown_matrix` and `equity_matrix` present: provide exactly one.
- `betting_tree` used outside matrix mode: it requires `hero_range` +
  `villain_range` + a matrix.
- invalid baseline strategy probabilities: an action distribution has unknown
  actions or does not sum to 1.
- unknown information set: `baseline_hero_strategy` references an information set
  the tree does not have.

## 8. Minimal examples

A single-hand scenario (trimmed):

```json
{
  "format_version": "1",
  "scenario_id": "nuts_chop_steal_bet98",
  "rake": { "rate": 0.05, "cap": 4.0 },
  "initial_commitment": { "hero": 1.0, "villain": 1.0 },
  "bet_size": 98.0,
  "showdown": "chop",
  "baseline_hero_strategy": { "IP_vs_bet": { "call": 0.0, "fold": 1.0 } },
  "candidate_generation": { "shift_amounts": [1.0] },
  "repeated": { "horizons": [10, 100], "discount": 1.0 }
}
```

An equity-matrix matchup cell looks like
`"equity_matrix": { "hero_strong": { "villain_bluff": 1.0, "villain_value": 0.0 } }`.

Full samples (do not copy partial files; start from these):

- [`examples/scenarios/nuts_chop_steal_bet98.json`](../examples/scenarios/nuts_chop_steal_bet98.json) (single-hand mode)
- [`examples/scenarios/abstract_range_steal_bet98.json`](../examples/scenarios/abstract_range_steal_bet98.json) (Hero-range-only mode)
- [`examples/scenarios/range_matrix_steal_bet98.json`](../examples/scenarios/range_matrix_steal_bet98.json) (showdown matrix)
- [`examples/scenarios/range_equity_steal_bet98.json`](../examples/scenarios/range_equity_steal_bet98.json) (equity matrix)
- [`examples/scenarios/range_equity_betting_tree_bet98.json`](../examples/scenarios/range_equity_betting_tree_bet98.json) (matrix + betting tree)

See also [`docs/examples_guide.md`](examples_guide.md) for how to run the
analysis and batch tooling on these files.
