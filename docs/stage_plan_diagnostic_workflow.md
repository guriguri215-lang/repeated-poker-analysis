# Bounded stage-plan diagnostic public workflow

This guide runs one deliberately tiny exact-rational fixture through M11's
existing top-level `repeated_poker` API. It demonstrates the complete input
boundary: a finite stage tree, two public states, explicit public monitoring,
complete prescribed profiles, fixture-specific attestations, an exact numeric
enclosure, and an allocation-before-materialization plan cap.

Run it from the repository root:

```powershell
python examples/stage_plan_diagnostic_workflow.py
```

The script prints one strict JSON-safe line, writes no files, and uses no network
or external data. It omits paths, credentials, runtime identity, full deviation
rows, and dataclass dumps.

## Worked fixture and manual oracle

The stage tree has one public Hero information set `H1` at node `h`:

- `stay` ends with Hero / Villain / residual payoffs `0 / 0 / 0`.
- `deviate` ends with payoffs `1 / -1 / 0`.
- every numeric input is a `Fraction`.

The prescribed profile in both `C` and `P` is Hero `stay=1`, `deviate=0`.
Villain has no information set, so its complete profile is the empty mapping.
The prescribed value is therefore exactly zero in either state. A one-period
Hero plan choosing `deviate` gains exactly one. Both actions keep the same next
state, so this gain is independent of `delta=1/2`.

Hero has two pure plans and Villain has one empty pure plan. Both players are
checked in both states, giving the independently hand-counted row total:

```text
2 states * (2 Hero plans + 1 Villain plan) = 6 deviation rows
maximum lower = maximum upper = 1
```

With `epsilon_claim=0` and the exact zero error enclosure, the expected analytic
status is `FAIL`. That is the normal success path for this example: the
diagnostic found the intentionally positive deviation. The process exits zero.

## Public monitoring and two-state timing

The finite public signal alphabet is enumerated completely:

- `(Hero, stay)` followed by terminal observable `terminal`.
- `(Hero, deviate)` followed by terminal observable `terminal`.

Actor and action order are part of each signal; no private information or
deviator identity is added. The transition map is total on `{C,P}` times the two
signals. In `C`, either signal returns to `C`. In `P`, either signal returns to
`P`, so `P` is absorbing.

The current stage finishes and pays out before its signal selects the next
period's state. `P` is not a terminal state: the same iid stage game continues
forever under the `P` profile.

## Attestations are evidence, not automatic proofs

`ModelClassAttestation` contains eleven explicit assertions about this fixture:
iid stage kernel, no persistent private state or cross-period correlation,
public-state sufficiency, public-signal boundaries, no known finite horizon, and
absorbing grim only. Setting them to true records the fixture author's claim; it
does not make the library automatically prove those model facts.

The `ManualPerfectRecallAttestation` is equally explicit and fixture-specific
human evidence, not a general perfect-recall proof. It binds:

- fixture version `stage-plan-public-example-v1` and the full tree content hash;
- Hero information set `H1` to the single member node `h`;
- that member's empty prior observations, own actions, and information sets;
- the exact ordered legal actions `stay`, `deviate`;
- reviewer, date, method, evidence, limitations, and invalidation conditions.

Any tree, node, information partition, member, history, or legal-action change
invalidates this record. The example authors the record directly; it does not
call a production private helper or a test helper to manufacture attestation
evidence.

## Exact numbers, caps, identity, and no partial results

The worked call fixes `stage_payoff_bound=1`, `input_tolerance=0`,
`epsilon_claim=0`, `exact_zero_error_bound()`, and
`max_plans_per_player=2`. Exact `Fraction` inputs keep values and gain intervals
on the documented unnormalized discounted-value scale.

`max_plans_per_player` is an allocation-before-materialization refusal boundary.
If the complete count exceeds the cap, the result is `UNSUPPORTED` with
`deviations=()`. Callers must not clamp, truncate, skip, or xfail plans to obtain
a smaller answer.

The manual attestation's tree content identity prevents reusing the evidence
after a content change. An identity mismatch is `UNSUPPORTED` and carries no
deviation rows. Likewise, `UNSUPPORTED` and `INDETERMINATE` must not display
partial values or a success claim. The worked script emits stdout only after it
receives the expected complete `FAIL`; malformed input or an unexpected
exception returns nonzero, writes a short stderr error, and leaves stdout empty.

## Status and input-error meanings

- `PASS`: every completely enumerated one-period pure stage-plan interval has
  upper endpoint at or below `epsilon_claim` under the stated enclosure.
- `FAIL`: the maximum lower endpoint exceeds `epsilon_claim`; the worked fixture
  intentionally takes this normal analytic path.
- `INDETERMINATE`: valid inputs did not establish which side of the threshold
  contains the true gain, for example because an interval crosses it. It is not
  a pass and exposes no partial claim.
- `UNSUPPORTED`: the declared model evidence, identity, perfect recall record,
  or complete enumeration cap does not support running the bounded claim. It is
  not a pass and exposes no partial deviation result.
- `ValueError`: the input contract itself is malformed, such as an invalid cap,
  partial transition, illegal action, invalid exact number, or incomplete
  profile. This is distinct from all four analytic statuses.

## Claim boundary and non-goals

The only claim is a bounded exhaustive one-period stage-plan deviation
diagnostic for this exact-rational iid stage game, `{C,P}` public monitoring,
absorbing grim state, and complete pure stage-plan deviations at period
boundaries.

Even a `PASS` does not establish arbitrary multi-period deviation resistance,
stage-internal sequential rationality, beliefs, zero-reach action quality,
finite punishment, or a known finite-horizon result. It does not establish an
equilibrium, Nash equilibrium, subgame-perfect equilibrium, certificate, proof,
or optimality. This workflow adds no new JSON/schema/format, CLI operation,
pipeline, manifest, report, solver, private-monitoring model, strategy
recommendation, profitability claim, or real-money advice.
