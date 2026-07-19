# Bounded stage-plan diagnostic file workflow

This strict two-phase adapter saves the existing M11 bounded diagnostic as a
versioned JSON document. It is a file boundary over the existing public
`diagnose_stage_plan_deviations` API, not a new analytic algorithm.

Run the bundled inspect fixture from the repository root:

```powershell
python scripts/run_stage_plan_diagnostic_file.py examples/stage_plan_diagnostic_file_v1.json
```

The command prints exactly one strict JSON line. A successful `inspect`, or a
complete analytic `PASS` or `FAIL` from `run`, exits 0. Every controlled outer
failure exits 2. stderr stays empty, and failures always have `output=null`.

## Two-phase boundary

The format is `stage-plan-diagnostic-file-v1`.

`inspect` strictly parses and preflights the finite tree, public monitoring,
complete C/P profiles, fixture version, canonical exact rational numeric
envelope, core plan cap, and caller-lowerable workflow caps. It returns:

- the existing public `tree_content_identity`;
- deterministic tree, monitoring, profile, plan, and predicted-row counts;
- a semantic inspection identity binding every analysis-relevant input and cap;
- an 11-field model-class template whose values are all null; and
- a fixture-bound 15-field manual perfect-recall template.

Inspect does not call `diagnose_stage_plan_deviations`. It does not establish an
analytic status, model sufficiency, perfect recall, or any human confirmation.
Derived member paths and legal actions are review material only. In particular,
the null `observations`, reviewer fields, evidence, confirmation, limitations,
invalidation conditions, and validity fields must be authored by a human.

`run` repeats all parsing and preflight, recomputes the inspection identity,
then requires the complete human-authored model and recall records. It rejects
stale, false, incomplete, mismatched, or invalidated evidence before analysis.
Only after every identity, evidence, and cap gate succeeds does it call the
existing public M11 diagnostic, exactly once.

## Document shape

Both operations have exactly these base keys:

```text
format_version, operation, request_id, fixture_version,
tree, monitoring, profiles, numeric, core_limits, workflow_limits
```

A run document adds exactly:

```text
inspection_identity, model_attestation, perfect_recall_attestation
```

Unknown, missing, or duplicate keys are rejected. The input must be UTF-8
without a BOM. NaN and Infinity tokens, invalid UTF-8, excessive JSON nesting,
and non-object top-level values are controlled failures.

### Recursive tree

The root is stored under `tree.root`. Node variants are exact-key objects:

- terminal: `type`, `node_id`, `hero_payoff`, `villain_payoff`,
  `house_residual`;
- chance: `type`, `node_id`, `children`, where each child has `probability` and
  `child`;
- hero or villain: `type`, `node_id`, `info_set`, `actions`, where each action
  has `action` and `child`.

Node IDs and action labels are unique where required. Chance probabilities sum
exactly to one. Player payoffs respect `stage_payoff_bound`, and every terminal
satisfies the exact Hero + Villain + residual accounting identity. The public
structural validator is also applied with zero tolerance and signed residuals
allowed.

### Public monitoring

`monitoring` contains:

- `public_action_node_ids`;
- one `terminal_observables` row for every terminal;
- the complete `signal_alphabet`, each signal holding an ordered
  `{actor, action}` trace and terminal observable; and
- a total `transitions` table keyed by C/P and `signal_index`.

The alphabet must equal the complete feasible public signals derived from the
tree. Actor, action, and order are preserved. The format has no field for
private information or deviator identity. Every P transition returns to P, so
P remains the same absorbing grim state used by M11.

### Complete profiles and exact numbers

`profiles` has exactly C and P, and each state has exactly Hero and Villain.
Each player value is a complete list of information-set rows and complete legal
action probabilities. Empty lists are required when a player has no
information sets.

Every analytic number is a canonical exact rational string: examples are
`"0"`, `"-1"`, and `"1/2"`. JSON numbers, floats, decimals such as `"0.5"`,
booleans, reducible fractions such as `"2/4"`, and spellings such as `"0/1"`
are rejected. Probabilities and error components must be non-negative, and
probability rows sum exactly to one. No value is normalized, coerced, rounded,
clamped, or silently replaced.

`numeric` has `delta`, `stage_payoff_bound`, `input_tolerance`,
`epsilon_claim`, and the existing nine-field `numeric_error_bound` object. The
exact reference fixture uses zero for all eight components and explicitly sets
`enclosure_established=true`.

## Human evidence

The model template has the exact 11 public `ModelClassAttestation` fields. A run
must explicitly set every field to true. Null, false, missing, or unknown fields
are rejected before the core call.

The perfect-recall template has the exact 15 public
`ManualPerfectRecallAttestation` fields. Inspect binds the tree identity,
fixture version, information-set members, derived own information-set/action
histories, and legal actions. A human must explicitly provide observations,
fixture ID, reviewer, date, method, evidence, confirmed result, non-empty known
limitations, non-empty invalidation conditions, valid-through version, and an
explicit false invalidated flag. A run rejects any mismatch, stale version,
false confirmation, invalidation, incomplete support, or tree-path history
drift before analysis. The record is fixture-specific human evidence, not a
general perfect-recall proof.

## Identity, caps, and no-partial failures

The inspection identity binds format, fixture version, tree content, public
monitoring, both C/P profiles, exact numeric envelope, core plan cap, and every
workflow cap. It deliberately excludes request ID, filesystem path, clock,
platform, Python or Git version, and other runtime metadata.

Caller-lowerable ceilings cover input bytes, JSON depth and values, tree depth,
nodes and branches, public-action declarations, terminal observables, signals,
signal actions, transitions, profile rows and actions, attestation records and
text, predicted deviation rows, output records, and output bytes. A requested
ceiling above the built-in maximum is rejected rather than clamped.

Counts and conservative output bounds are checked before plan materialization
or core execution. `max_plans_per_player=1` therefore refuses the bundled
two-plan Hero fixture before analysis. Rows are never truncated, sampled,
skipped, or partially returned to satisfy a cap.

Every parse, identity, attestation, cap, invalid-input, non-complete analytic,
or unexpected-exception path returns `output=null`. Error text is bounded and
single-line; no values, rows, counts, identities, paths, or traceback leak into
the failure payload.

## Complete run output

M11 analytic `PASS` and `FAIL` are both successful file executions. The run
output is a deterministic full projection containing:

- the qualified bounded claim, identities, fixture version, and analytic
  status/message;
- exact configuration, core/workflow caps, and plan/deviation counts;
- all four prescribed Hero/Villain C/P values;
- every bounded deviation row with its complete pure plan, value, gain, lower,
  and upper endpoint; and
- maximum lower/upper endpoints and the unnormalized value scale.

All rational outputs remain canonical strings. The bundled manual oracle has
Hero/Villain plans 2/1, six rows, four prescribed values equal to zero, maximum
interval 1/1, and analytic status `FAIL`. That FAIL is an intended complete
diagnostic result, not a process failure.

M11 `INDETERMINATE` or `UNSUPPORTED` is an outer controlled failure with
`output=null`, as are identity, evidence, cap, malformed-input, or exception
failures. There is no partial-success mode.

## Claim boundary and non-goals

The only claim is the existing bounded exhaustive one-period stage-plan
deviation diagnostic for the exact-rational iid stage game, C/P public
monitoring, absorbing grim state, and complete pure stage-plan deviations at
period boundaries.

Even a PASS is not an equilibrium, Nash equilibrium, subgame-perfect or
sequential result, certificate, proof, optimality result, strategy
recommendation, profitability claim, or real-money advice. The adapter adds no
new M11 math, auto-attestation, private monitoring, finite punishment, known
finite horizon, arbitrary history/FSA policy, top-level export, generic
pipeline, scenario, manifest, report, GUI, prepared-game, AIoF, three-player,
STT, or ICM integration.
