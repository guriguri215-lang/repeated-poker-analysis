# Repeated Poker Analysis

Repeated Poker Analysis is a Python research prototype for exact fixed-strategy
response analysis and bounded Hero-commitment experiments in small poker models.
It accepts explicit finite models, compares a supplied baseline with candidate or
continuous Hero commitments, and produces response, repeated-value, timing, and
audit outputs.

> **Status: Alpha research prototype (v0.2.1).** The documented small-model
> workflows are implemented and heavily tested, but the package is not a full
> poker solver, its APIs may change, and its numerical claims have not been
> independently validated against a production solver.

## Problem

Locking one player's policy and asking how the other player responds is a
commitment problem, not automatically an equilibrium calculation. Repeating the
same spot also does not by itself create a reputation equilibrium. This project
makes those boundaries explicit: it represents a finite game, fixes Hero's
policy, computes the supported opponent response, and reports repeated-value and
detectability diagnostics under declared assumptions.

The intended users are researchers and developers studying game theory, poker
abstractions, exact-response algorithms, or bounded optimization. It is not
intended for real-money decision making.

## What this project is

It is a small-model analysis toolkit with the following implemented boundaries:

| Capability | Status | Implemented boundary |
|---|---|---|
| Two-player fixed-Hero response | Implemented | Exact Villain best-response correspondence in a supplied finite tree, using backward induction by default and complete enumeration as a small-tree oracle. |
| Candidate commitment analysis | Implemented | Generate, filter, compare, rank, and conditionally select a declared finite Hero candidate library. |
| Repeated-game diagnostics | Implemented / experimental interpretation | Known-horizon or discounted value comparisons, T_deadline, and local or reach-weighted T_detect diagnostics. |
| Abstract river and STT inputs | Implemented | Versioned JSON for small river models with rake and a separate abstract SB-vs-BB push/fold ICM path. |
| Prepared one-/two-street analysis | Experimental | Strict bounded inspect/run workflows over caller-supplied finite prepared trees and profiles. |
| Real-card adapters | Implemented for documented models | Exact-combo AIoF preflop and known-five-card-board river/rake workflows with card-removal checks and explicit workload caps. |
| Three-player analysis | Implemented for bounded contracts | Exact non-cooperative O1/O2 response for documented tiny models; a separate finite-iteration CFR-style diagnostic remains diagnostic only. |
| Certified global commitment objective | Implemented for conforming bounded oracles | Exact-rational branch-and-bound over the scenario-derived Hero simplex product; success certifies only the identified scalar objective at the reported tolerance. |
| CLI, reports, and local GUI | Implemented | Python scripts, JSON/CSV/Markdown exports, and five loopback-only browser editors/runners for the abstract river scenario modes. |

Implementation evidence is in src/repeated_poker, tests, examples, and
.github/workflows/ci.yml. Detailed workflow contracts are linked under
[Documentation](#documentation).

## What this project is not

- It does not compute a general GTO, Nash, repeated-game, or tournament
  equilibrium.
- It is not a large-scale range solver and does not claim solver-grade
  performance.
- It does not parse or certify raw commercial-solver exports. External profiles
  must first be converted into a documented scenario-native strategy map.
- It does not predict human learning, psychology, table image, or when an
  opponent will actually adapt.
- It does not implement Future ICM, FGS, tournament simulation, arbitrary
  multi-street no-limit trees, or a public hosted service.
- It does not use an LLM, external model provider, remote data service, or paid
  API.
- It does not provide gambling, bankroll, financial, legal, or real-money
  strategy advice, and a positive model EV is not a profitability guarantee.

## Current implementation

The main abstract two-player path is:

1. Parse and validate a versioned JSON scenario or construct a GameTree through
   the Python API.
2. Build the finite game, complete baseline Hero policy, and baseline Villain
   comparison policy.
3. Generate and optionally filter a finite Hero candidate library with
   `run_candidate_analysis_pipeline`.
4. Evaluate the fixed baseline and compute a fresh exact Villain response to
   each fixed Hero candidate.
5. Calculate repeated-value, T_deadline, and optional T_detect diagnostics.
6. Return Python dataclasses and optional JSON, CSV, or Markdown reports with a
   run manifest.

An explicit `baseline_villain_strategy` is a caller-chosen comparison profile,
not an equilibrium claim.

The certified-global workflows are a separate branch. They derive the full
legal Hero behavior-policy simplex product for one supported consumer, evaluate
an exact point objective and a sound whole-cell upper bound, and run the bounded
branch-and-bound core. They return either a certificate for the identified
scalar objective or a controlled no-certificate result; they do not silently
fall back to a grid, sample, local optimizer, or finite candidate list.

### Architecture

~~~mermaid
flowchart LR
    A["JSON scenario or Python request"] --> B["Schema and model validation"]
    G["Local GUI or CLI"] --> A
    B --> C["Finite game and supplied baseline"]
    C --> D["Finite candidate pipeline"]
    D --> E["Fixed-profile evaluation"]
    E --> F["Exact opponent response"]
    F --> H["Repeated-value, T_deadline, and T_detect diagnostics"]
    C --> I["Supported full Hero simplex domain"]
    I --> J["Point and whole-cell oracle"]
    J --> K["Certified branch-and-bound"]
    K --> L["Certificate or controlled no-certificate status"]
    H --> M["Python objects, JSON, CSV, and Markdown"]
    L --> M
~~~

This diagram describes the current implementation. The original research and
planned-architecture document is useful design history, but it is not a current
feature-status list.

## Requirements

| Requirement | Current state |
|---|---|
| Python | 3.10 or newer |
| Runtime dependencies | None beyond the Python standard library |
| Development dependency | pytest 7 or newer through the dev extra |
| Verified operating systems | Ubuntu in CI; Windows in the publication review |
| Not independently verified | macOS |
| External tools | Git is optional and used only for best-effort commit metadata in run manifests |
| Environment variables | None required |
| Network or API keys | None required |
| GPU | Not used |
| Paid service | Not required |

CI runs the MVP validation on Python 3.10 and 3.13. This review also reproduced
the full suite on Windows with Python 3.12. Pure-Python implementation does not
by itself establish untested platform support.

## Quickstart

### Fastest way to run the MVP

Clone the repository and install it from source:

~~~powershell
git clone https://github.com/guriguri215-lang/repeated-poker-analysis.git
cd repeated-poker-analysis
python -m venv .venv

# Windows PowerShell
.venv\Scripts\Activate.ps1

# macOS / Linux instead
# source .venv/bin/activate

python -m pip install -e .
python examples/nuts_chop_river.py
~~~

The example prints a JSON-like exact-response record. The bundled fixture should
include these values:

~~~text
"villain_max_ev": -0.8
"ev_h_worst": -0.8
"expected_house_rake_worst": 1.6
"num_villain_pure_strategies": 36
~~~

For the full validation path:

~~~powershell
python -m pip install -e ".[dev]"
python scripts/check_mvp.py
~~~

A successful run ends with:

~~~text
All 9 MVP checks passed.
~~~

The MVP script runs the complete pytest suite before the worked examples, so it
can take several minutes.

## Minimal Python example

This high-level API loads the bundled abstract river JSON and runs the candidate
pipeline:

~~~python
from repeated_poker import run_river_scenario_analysis

result = run_river_scenario_analysis(
    "examples/scenarios/nuts_chop_steal_bet98.json"
)
counts = result.pipeline_result.filter_result.summary_counts

print(result.scenario_id)
print(counts.total, counts.kept, counts.excluded)
~~~

Expected output for the bundled file:

~~~text
nuts_chop_steal_bet98
1 1 0
~~~

The package-root API focuses on the original abstract two-player workflow.
Specialized real-card, three-player, file-adapter, and certified-global APIs live
in their documented repeated_poker submodules. Public API stability is not yet
guaranteed; versioned file formats and explicit status objects are the stronger
compatibility boundaries.

## CLI and local GUI

The project does not install a console-script entry point. Run the scripts from a
checkout and use --help for the exact options. The main abstract runner is:

~~~powershell
python scripts/run_river_scenario_analysis.py --help
python scripts/run_river_scenario_analysis.py examples/scenarios/nuts_chop_steal_bet98.json --output-json reports/result.json --output-markdown reports/result.md --output-csv reports/result.csv --strict-json
~~~

Create a minimal versioned scenario template with:

~~~powershell
python scripts/create_scenario_template.py --help
~~~

Inputs are versioned JSON files. Results are written to stdout unless an output
path is requested. Argparse usage errors and controlled input failures return a
nonzero exit status; successful runs return 0.

### Extended workflow command index

These are bounded adapters, not additional solver claims. Read the linked
contract before interpreting an output:

| Workflow | Command | Boundary |
|---|---|---|
| [Real-card AIoF](docs/aiof_real_card_workflow.md) | `python examples/aiof_real_card_workflow.py` | `aiof-rational-lift-game-v1`; fixed-opponent response; not a range chart. |
| [AIoF rational-lift file](docs/aiof_rational_lift_file_workflow.md) | `python scripts/run_aiof_rational_lift_file.py examples/aiof_rational_lift_file_v1.json` | Two-phase, no-partial exact rational-lift runtime adapter; not an external-game solver or real-money tool. |
| [AIoF supplied-profile file](docs/aiof_supplied_profile_file_workflow.md) | `python scripts/run_aiof_supplied_profile_file.py examples/aiof_supplied_profile_file_v1.json` | Two-phase and no-partial; fixed-opponent input is not endogenous and is not real-money advice. |
| [Prepared two-street file](docs/prepared_two_street_file_workflow.md) | `python scripts/run_prepared_two_street_file.py inspect examples/prepared_two_street_file_v2.json` | Strict `prepared-two-street-file-v2` inspection of caller-supplied finite data. |
| [Stage-plan file](docs/stage_plan_diagnostic_file_workflow.md) | `python scripts/run_stage_plan_diagnostic_file.py examples/stage_plan_diagnostic_file_v1.json` | Two-phase, human-authored, canonical exact rational, and no-partial diagnostic; not real-money advice. |
| [Stage-plan worked example](docs/stage_plan_diagnostic_workflow.md) | `python examples/stage_plan_diagnostic_workflow.py` | A reported `FAIL` is successful diagnostic execution, not a process failure. |
| [Three-player CFR file](docs/three_player_cfr_file_workflow.md) | `python scripts/run_three_player_cfr_file.py examples/three_player_cfr_file_v1.json` | Two-phase, human-authored, no-partial diagnostic; not real-money advice. |
| [Three-player CFR example](docs/three_player_cfr_diagnostic_workflow.md) | `python examples/three_player_cfr_diagnostic_workflow.py` | Fixed Hero and separate O1/O2 finite-iteration diagnostic with an allocation-before-materialization cap; not a convergence claim. |
| [Three-player exact candidate workflow](docs/three_player_candidate_repeated_workflow.md) | `python examples/three_player_candidate_repeated_workflow.py` | Bounded finite exact-response/candidate workflow, not the guarded CFR-style diagnostic. |
| [Known-board HU certified global](docs/known_board_real_card_hu_certified_global.md) | `python examples/known_board_real_card_hu_certified_global.py` | M38 certificate for its identified bounded scalar objective only. |

Five standard-library browser prototypes edit, validate, save, and analyze the
abstract river modes:

| Scenario mode | Command | Port | Editor | Analyze |
|---|---|---:|---|---|
| Single hand | `python scripts/serve_single_hand_gui.py` | 8000 | Yes | Yes |
| Hero range | `python scripts/serve_hero_range_gui.py` | 8001 | Yes | Yes |
| Showdown matrix | `python scripts/serve_showdown_matrix_gui.py` | 8002 | Yes | Yes |
| Equity matrix | `python scripts/serve_equity_matrix_gui.py` | 8003 | Yes | Yes |
| River betting tree | `python scripts/serve_betting_tree_gui.py` | 8004 | Yes | Yes |

Keep the default 127.0.0.1 bind and use the GUI only on a trusted machine. The
shared handler checks Host, JSON content type, browser origin/fetch metadata,
and rejects CORS preflight before POST dispatch. This is a browser
request-provenance boundary, not authentication or a filesystem sandbox: an
accepted local client can still supply a local path and explicitly request an
overwrite.

## Use cases

- Hand-check the exact response to a completely fixed Hero policy in a tiny
  extensive-form game.
- Compare a declared finite set of commitment candidates before and after the
  opponent response.
- Explore how an assumed adaptation opportunity or idealized detection channel
  changes a repeated-value comparison.
- Validate a bounded real-card or three-player fixture with explicit card,
  response, identity, and workload contracts.
- Exercise a supported certified-global scalar objective and inspect the
  certificate or controlled no-certificate evidence.
- Use small reproducible examples when teaching or testing game-theoretic
  software.

None of these use cases turns the output into real-world poker advice.

## Validation and testing

At main commit 0accab0, this publication review ran:

| Validation | Result | Scope |
|---|---|---|
| pytest suite | 3,491 passed | Unit, property-style, schema, CLI, docs, security-boundary, numerical, cap, and integration tests |
| MVP validation | 9 of 9 checks passed | Full suite plus eight representative examples |
| CI configuration | Python 3.10 and 3.13 on Ubuntu | The v0.2.0 release records successful runs for both versions |
| Hand-calculated fixtures | Present | Tiny nuts/chop, ICM, AIoF, stage-plan, three-player, and optimizer examples |
| Public-content scan | Automated test plus review | Checks public files for private paths and sensitive strings |
| External solver comparison | Not performed | No independent production-solver certification |
| macOS execution | Not verified | No macOS CI job |

Test volume is evidence of exercised contracts, not proof of correctness for
arbitrary games. Some exact-rational and strict file workflows test byte
determinism. General run manifests include a UTC timestamp and best-effort Git
commit, so they provide provenance rather than guaranteeing that every complete
output file is byte-identical across runs.

## Limitations

- **Scale and complexity:** complete trees, response ties, exact rational
  arithmetic, card combinations, and branch-and-bound cells can grow
  combinatorially. The APIs use explicit caps and may return no result or no
  certificate.
- **Exactness boundary:** "exact" means exact within the supplied finite model
  and the documented numeric contract. It does not make the model a complete
  representation of poker.
- **Optimization boundary:** a certified-global success covers one identified
  bounded scalar objective and conforming bound oracle, not a Nash equilibrium
  or universal Hero optimum.
- **Model dependence:** results depend on the action tree, ranges, weights,
  rake, ICM payouts, fixed profiles, and timing assumptions supplied by the
  caller.
- **Detection:** T_detect is an idealized public-signal sensitivity diagnostic,
  not an opponent-learning or psychological forecast.
- **ICM:** the STT path uses Malmuth-Harville ICM prize-EV deltas and omits blind
  evolution, Future ICM, FGS, future hands, skill, and tournament dynamics.
- **Three players:** the exact workflow models separate non-cooperative O1 and
  O2 responses in the documented bounded tree. The CFR-style path is a finite
  diagnostic snapshot and does not establish convergence.
- **Real cards:** supported adapters cover explicit combos or known-board
  bounded models. There is no general range shorthand parser, raw solver import,
  arbitrary runout engine, or general multi-street solver.
- **Security and privacy:** no remote service is used, but CLIs and local GUIs
  can read or write user-selected local paths. There is no authentication,
  sandbox, or formal security-response policy.
- **API stability:** this is alpha software; Python APIs and experimental
  schemas may change between releases.
- **UI/UX:** the local GUIs are feature-frozen prototypes without graphing,
  authentication, deployment support, or a public-service threat model.
  The surface is frozen (bug fixes only).

See [Assumptions and Limitations](docs/assumptions_and_limitations.md) before
interpreting any result.

## Project structure

| Path | Purpose |
|---|---|
| src/repeated_poker | Library code and specialized workflow submodules |
| tests | Automated test suite |
| examples | Executable Python and JSON fixtures |
| scripts | CLI adapters, validation/export tools, MVP checker, and local GUIs |
| docs | Mathematical contracts, formats, workflows, and limitations |
| .github/workflows/ci.yml | Ubuntu CI for Python 3.10 and 3.13 |
| 02_research_and_implementation_plan.md | Original research model and design plan; not a current status tracker |

## Documentation

Start with:

- [MVP Walkthrough](docs/mvp_walkthrough.md) - original abstract two-player
  pipeline and output interpretation.
- [Examples Guide](docs/examples_guide.md) - recommended example order and
  commands.
- [Assumptions and Limitations](docs/assumptions_and_limitations.md) - model
  boundaries and non-claims.
- [Scenario Format Reference](docs/scenario_format_reference.md) - abstract
  river JSON schema.
- [STT Push/Fold Format Reference](docs/stt_pushfold_format_reference.md) -
  separate ICM scenario schema.
- [Baseline Solution Import Format](docs/baseline_solution_import_format.md) -
  scenario-native profile boundary, not raw solver import.
- [GUI/Form Input Design and Status](docs/gui_input_design.md) - implemented
  local editors, request boundary, and remaining UI limits.
- [Public Release Readiness Checklist](docs/public_readiness_checklist.md) -
  publication checks for releases and metadata changes.
- [Publication Policy](docs/publication_policy.md) - public positioning and
  disclosure boundaries.

Specialized workflows:

- [Prepared Two-Street File Workflow](docs/prepared_two_street_file_workflow.md)
- [Stage-Plan Diagnostic](docs/stage_plan_diagnostic_workflow.md)
- [Three-Player CFR-Style Diagnostic](docs/three_player_cfr_diagnostic_workflow.md)
- [Exact Three-Player Candidate/Repeated Workflow](docs/three_player_candidate_repeated_workflow.md)
- [Real-Card AIoF Workflow](docs/aiof_real_card_workflow.md)
- [Known-Board HU River/Rake Adapter](docs/known_board_real_card_hu_river_rake_adapter.md)
- [Known-Board Three-Player River/Rake Adapter](docs/known_board_real_card_three_player_river_rake_adapter.md)
- [Certified Global Optimizer Core](docs/certified_global_optimizer_core.md)
- [AIoF Preflop Certified Global](docs/aiof_preflop_certified_global.md)
- [Known-Board HU Certified Global](docs/known_board_real_card_hu_certified_global.md)
- [Three-Player Certified Global](docs/three_player_certified_global.md)
- [Unified Certified-Global Workflow](docs/unified_certified_global_workflow.md)

## Roadmap and status separation

Completed in the current release line:

- [x] Small-tree exact fixed-Hero response and finite candidate pipeline
- [x] JSON validation, reports, manifests, CLI adapters, and five local GUI modes
- [x] Abstract STT ICM, prepared-tree, real-card, and bounded three-player paths
- [x] M36-M40 certified-global core, consumer adapters, and unified workflow
- [x] v0.2.0 release validation on Python 3.10 and 3.13
- [x] v0.2.1 public documentation and repository metadata refresh

Experimental or only partially validated:

- [ ] Independent comparison with an external solver or published benchmark
- [ ] Stable public-API and deprecation policy
- [ ] macOS CI and documented platform verification
- [ ] Formal security-response and contribution processes

Not implemented and not promised by a dated roadmap:

- Full GTO or repeated-game equilibrium solving
- Large-scale commercial-solver range import or replacement
- Future ICM, FGS, real opponent learning, or public hosted deployment

The older [Research and Implementation Plan](02_research_and_implementation_plan.md)
contains target architecture and historical development phases. Current status is
defined by code, tests, this README, and the workflow documents above.

## Contributing

The repository does not yet have a formal external-contribution policy. Use
[GitHub Issues](https://github.com/guriguri215-lang/repeated-poker-analysis/issues)
for a reproducible bug report or documentation problem before proposing a large
change. Run python scripts/check_mvp.py before submitting code changes.

## License

Released under the [MIT License](LICENSE). The software is provided as-is,
without warranty.
