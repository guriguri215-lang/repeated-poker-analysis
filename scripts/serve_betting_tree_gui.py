#!/usr/bin/env python3
"""Serve a minimal local GUI for editing a river betting-tree scenario (prototype v1).

Usage:

    python scripts/serve_betting_tree_gui.py --port 8004
    # then open http://127.0.0.1:8004/ in a browser

This is a pre-GUI-product prototype: a local-only web page that loads a river
betting-tree scenario JSON into a form (top-level fields, the three betting-tree
sizes, a matrix-type selector, a table of weighted Hero buckets with their two
decision-point distributions, a table of weighted Villain buckets, and a
Hero x Villain matrix), lets you add / remove / edit buckets, rebuild the matrix,
validate the form, and save it back to JSON. It is the betting-tree counterpart of
the matrix editors (``scripts/serve_showdown_matrix_gui.py`` /
``scripts/serve_equity_matrix_gui.py``) and reuses the shared local-GUI scaffolding
in ``scripts/gui_common.py``. Standard library only; no framework, no dependency,
and no new solver, model, or analysis logic.

The betting tree (river one-street) adds, on top of a matrix scenario, an IP stab
after an OOP check and a single raise line against an OOP bet; each Hero bucket
therefore has two decision distributions instead of one call/fold split. The
matchup outcomes still come from a matrix whose ``matrix_type`` is ``"showdown"``
(discrete hero / villain / chop) or ``"equity"`` (Hero pot share before rake in
[0, 1]); equity values are abstract inputs from the JSON, not computed from cards.

Scope (v1): betting-tree mode and editing only (load / edit buckets / edit sizes /
edit matrix cells / validate / save). Single-hand, Hero-range-only, and plain
showdown- / equity-matrix scenarios (without a ``betting_tree``) are rejected.
Running the analysis pipeline, charts, real-card parsing, and external solver
imports are all out of scope. It reuses the existing form model, parser, and game
builder as the source of truth.

Endpoints:

* ``GET  /``            -- the betting-tree editor page;
* ``POST /api/load``    -- ``{"path": "..."}`` -> the form fields of that
  betting-tree scenario (a non-betting-tree scenario is rejected);
* ``POST /api/validate``-- ``{"form": {...}}`` -> field-level validation messages;
* ``POST /api/save``    -- ``{"path", "form", "force", "strict_json"}`` -> writes
  the JSON only when the form validates and its ``to_dict`` re-parses and rebuilds.

Security / safety: it binds to ``127.0.0.1`` by default and makes no external
calls. It reads and writes only the local paths you type, refuses to overwrite an
existing file unless the overwrite box is checked, requires real booleans for the
save options, keeps the raw ``format_version`` (no coercion), and returns short
``error`` messages (never a traceback).
"""

from __future__ import annotations

import argparse
import sys
from http.server import ThreadingHTTPServer
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))  # for the sibling CLIs

from repeated_poker import (  # noqa: E402  (path is set up above)
    BettingTreeScenarioForm,
    BettingTreeSizingForm,
    HeroBettingTreeBucketForm,
    VillainMatrixBucketForm,
    betting_tree_form_from_dict,
    betting_tree_form_to_dict,
    build_river_steal_game_from_scenario,
    detect_scenario_form_mode,
    river_scenario_from_dict,
    validate_betting_tree_form,
)
from repeated_poker.report_export import _dump_json  # noqa: E402

# Reuse the loader, the safe file writer, and the value parsing from the sibling
# CLIs rather than duplicating them.
from inspect_scenario_form import _load_scenario_dict  # noqa: E402
from roundtrip_scenario_form import _write_output  # noqa: E402
from edit_scenario_form import _NO_CAP_VALUES, _to_float, _to_number_list  # noqa: E402

# Shared local-GUI scaffolding (HTTP handler / server builder) and small payload
# primitives, factored out of the sibling GUI scripts.
from gui_common import as_text as _as_text  # noqa: E402
from gui_common import build_server as _build_server  # noqa: E402
from gui_common import messages_payload as _messages_payload  # noqa: E402
from gui_common import require_bool as _require_bool  # noqa: E402

# Top-level (non-bucket) flat fields shown in the form.
_TOP_FIELDS = [
    "scenario_id",
    "description",
    "rake_rate",
    "rake_cap",
    "initial_commitment_hero",
    "initial_commitment_villain",
    "bet_size",
    "shift_amounts",
    "horizons",
    "discount",
]
# The three betting-tree sizes (nested under "betting_tree" in the payload).
_SIZING_FIELDS = [
    "oop_bet_size",
    "ip_bet_after_check_size",
    "ip_raise_size",
]
# Per-Hero-bucket fields: id / weight plus the two decision-point distributions.
_HERO_FIELDS = [
    "hand_id",
    "weight",
    "after_oop_check_check_probability",
    "after_oop_check_bet_probability",
    "vs_oop_bet_call_probability",
    "vs_oop_bet_fold_probability",
    "vs_oop_bet_raise_probability",
]
_VILLAIN_FIELDS = [
    "hand_id",
    "weight",
]
# The float per-bucket fields (hand_id stays a string).
_HERO_FLOAT_FIELDS = [f for f in _HERO_FIELDS if f != "hand_id"]


def _equity_cell_value(raw):
    """Convert one equity matrix cell to a float when possible, else keep it.

    Equity cells are numbers (the Hero pot share before rake), so a numeric string
    from the browser is parsed to ``float`` -- including ``"nan"`` / ``"inf"`` and
    out-of-range values, which stay as floats so the validator flags them. A value
    that does not parse as a number (an empty string, ``"abc"``, a bool, ``None``)
    is kept unchanged so :func:`validate_betting_tree_form` reports it as a bad cell
    rather than the conversion raising or the value being silently coerced (in
    particular it is never rounded to a default like ``0.5``).

    This mirrors the equity-matrix GUI's cell handling; see the repo-external memo
    for the note that this small helper is a candidate to move into
    ``gui_common`` once a third caller appears.
    """

    if isinstance(raw, bool):
        return raw  # keep so the validator rejects a boolean cell
    if isinstance(raw, (int, float)):
        return raw
    text = _as_text(raw).strip()
    try:
        return float(text)
    except ValueError:
        return raw


def _hero_bucket_from_payload(raw, index: int) -> HeroBettingTreeBucketForm:
    """Convert one Hero bucket dict from the browser into a bucket form."""

    bucket = HeroBettingTreeBucketForm()
    if "hand_id" in raw:
        bucket.hand_id = _as_text(raw["hand_id"])
    for field_name in _HERO_FLOAT_FIELDS:
        if field_name in raw:
            setattr(
                bucket,
                field_name,
                _to_float(f"hero_buckets[{index}].{field_name}", _as_text(raw[field_name])),
            )
    return bucket


def _villain_bucket_from_payload(raw, index: int) -> VillainMatrixBucketForm:
    """Convert one Villain bucket dict from the browser into a bucket form."""

    bucket = VillainMatrixBucketForm()
    if "hand_id" in raw:
        bucket.hand_id = _as_text(raw["hand_id"])
    if "weight" in raw:
        bucket.weight = _to_float(
            f"villain_buckets[{index}].weight", _as_text(raw["weight"])
        )
    return bucket


def _buckets_from_payload(raw_list, field_name, convert) -> list:
    """Convert a list of bucket dicts, keeping malformed entries for the validator.

    A non-list raises :class:`ValueError`; a non-dict entry is kept as-is so
    :func:`validate_betting_tree_form` reports it as a ``<field>[i]`` message
    instead of the conversion raising. A bad value inside a bucket dict (for
    example a non-numeric weight) raises :class:`ValueError`.
    """

    if not isinstance(raw_list, list):
        raise ValueError(f"{field_name} must be a list")
    buckets = []
    for index, entry in enumerate(raw_list):
        if not isinstance(entry, dict):
            buckets.append(entry)
        else:
            buckets.append(convert(entry, index))
    return buckets


def _sizing_from_payload(raw) -> BettingTreeSizingForm:
    """Convert the betting_tree sizes dict into a BettingTreeSizingForm."""

    sizing = BettingTreeSizingForm()
    if raw is None:
        return sizing
    if not isinstance(raw, dict):
        raise ValueError("betting_tree must be a JSON object")
    for field_name in _SIZING_FIELDS:
        if field_name in raw:
            setattr(
                sizing,
                field_name,
                _to_float(f"betting_tree.{field_name}", _as_text(raw[field_name])),
            )
    return sizing


def _matrix_from_payload(matrix_type, raw):
    """Convert the nested matrix, parsing equity cells to floats when applicable.

    For ``matrix_type == "equity"`` each numeric cell is parsed to ``float`` (a
    non-numeric cell is kept for the validator). For showdown (or an unknown
    ``matrix_type``, which the validator rejects) the cells are carried through
    unchanged. A non-object matrix or row is kept as-is so the validator reports it
    rather than the conversion raising.
    """

    if not isinstance(raw, dict):
        return raw
    if matrix_type != "equity":
        return {
            hero_id: dict(row) if isinstance(row, dict) else row
            for hero_id, row in raw.items()
        }
    matrix = {}
    for hero_id, row in raw.items():
        if not isinstance(row, dict):
            matrix[hero_id] = row
            continue
        matrix[hero_id] = {
            villain_id: _equity_cell_value(cell) for villain_id, cell in row.items()
        }
    return matrix


def _form_from_payload(payload) -> BettingTreeScenarioForm:
    """Build a BettingTreeScenarioForm from the browser's flat dict.

    Top-level numeric fields, the three sizes, and per-bucket weights /
    probabilities are converted with the edit-CLI rules (floats, optional
    ``rake_cap``, comma-separated ``shift_amounts`` / ``horizons``).
    ``format_version`` and ``matrix_type`` are kept raw (an unsupported value is
    reported by the validator, not coerced). Equity matrix cells are parsed to
    floats where possible (a non-numeric cell is kept for the validator); showdown
    cells and a non-object matrix are kept as-is. A non-list bucket section or a
    bad numeric value raises :class:`ValueError`.
    """

    if not isinstance(payload, dict):
        raise ValueError("form must be a JSON object")
    form = BettingTreeScenarioForm()
    if "format_version" in payload:
        form.format_version = payload["format_version"]
    if "scenario_id" in payload:
        form.scenario_id = _as_text(payload["scenario_id"])
    if "description" in payload:
        form.description = _as_text(payload["description"])
    if "rake_rate" in payload:
        form.rake_rate = _to_float("rake_rate", _as_text(payload["rake_rate"]))
    if "rake_cap" in payload:
        raw_cap = _as_text(payload["rake_cap"])
        form.rake_cap = (
            None if raw_cap.strip().lower() in _NO_CAP_VALUES else _to_float("rake_cap", raw_cap)
        )
    if "initial_commitment_hero" in payload:
        form.initial_commitment_hero = _to_float(
            "initial_commitment_hero", _as_text(payload["initial_commitment_hero"])
        )
    if "initial_commitment_villain" in payload:
        form.initial_commitment_villain = _to_float(
            "initial_commitment_villain", _as_text(payload["initial_commitment_villain"])
        )
    if "bet_size" in payload:
        form.bet_size = _to_float("bet_size", _as_text(payload["bet_size"]))
    if "shift_amounts" in payload:
        form.shift_amounts = _to_number_list(
            "shift_amounts", _as_text(payload["shift_amounts"]), float
        )
    if "horizons" in payload:
        form.horizons = _to_number_list("horizons", _as_text(payload["horizons"]), int)
    if "discount" in payload:
        form.discount = _to_float("discount", _as_text(payload["discount"]))
    if "betting_tree" in payload:
        form.betting_tree = _sizing_from_payload(payload["betting_tree"])
    if "matrix_type" in payload:
        form.matrix_type = _as_text(payload["matrix_type"])
    if "hero_buckets" in payload:
        form.hero_buckets = _buckets_from_payload(
            payload["hero_buckets"], "hero_buckets", _hero_bucket_from_payload
        )
    if "villain_buckets" in payload:
        form.villain_buckets = _buckets_from_payload(
            payload["villain_buckets"], "villain_buckets", _villain_bucket_from_payload
        )
    if "matrix" in payload:
        form.matrix = _matrix_from_payload(form.matrix_type, payload["matrix"])
    return form


def _form_to_payload(form: BettingTreeScenarioForm) -> dict:
    """Flatten a form into the string-friendly dict the browser displays."""

    sizing = form.betting_tree
    return {
        "format_version": form.format_version,
        "scenario_id": form.scenario_id,
        "description": form.description,
        "rake_rate": form.rake_rate,
        "rake_cap": "" if form.rake_cap is None else form.rake_cap,
        "initial_commitment_hero": form.initial_commitment_hero,
        "initial_commitment_villain": form.initial_commitment_villain,
        "bet_size": form.bet_size,
        "shift_amounts": ",".join(str(value) for value in form.shift_amounts),
        "horizons": ",".join(str(value) for value in form.horizons),
        "discount": form.discount,
        "betting_tree": {
            "oop_bet_size": sizing.oop_bet_size,
            "ip_bet_after_check_size": sizing.ip_bet_after_check_size,
            "ip_raise_size": sizing.ip_raise_size,
        },
        "matrix_type": form.matrix_type,
        "hero_buckets": [
            {
                "hand_id": bucket.hand_id,
                "weight": bucket.weight,
                "after_oop_check_check_probability": bucket.after_oop_check_check_probability,
                "after_oop_check_bet_probability": bucket.after_oop_check_bet_probability,
                "vs_oop_bet_call_probability": bucket.vs_oop_bet_call_probability,
                "vs_oop_bet_fold_probability": bucket.vs_oop_bet_fold_probability,
                "vs_oop_bet_raise_probability": bucket.vs_oop_bet_raise_probability,
            }
            for bucket in form.hero_buckets
        ],
        "villain_buckets": [
            {"hand_id": bucket.hand_id, "weight": bucket.weight}
            for bucket in form.villain_buckets
        ],
        "matrix": {hero_id: dict(row) for hero_id, row in form.matrix.items()},
    }


def api_load(payload) -> dict:
    """Load a betting-tree scenario file into form fields for the browser."""

    if not isinstance(payload, dict):
        raise ValueError("request must be a JSON object")
    path = payload.get("path")
    if not isinstance(path, str) or not path.strip():
        raise ValueError("path is required")
    data = _load_scenario_dict(path.strip())
    mode = detect_scenario_form_mode(data)
    if mode != "betting-tree":
        raise ValueError(
            "betting-tree GUI supports the betting_tree mode; "
            f"this scenario is {mode} mode"
        )
    form = betting_tree_form_from_dict(data)
    return {"ok": True, "mode": mode, "form": _form_to_payload(form)}


def api_validate(payload) -> dict:
    """Validate the form the browser sent, returning field-level messages."""

    if not isinstance(payload, dict):
        raise ValueError("request must be a JSON object")
    form = _form_from_payload(payload.get("form"))
    messages = validate_betting_tree_form(form)
    return {"ok": True, "valid": not messages, "messages": _messages_payload(messages)}


def api_save(payload) -> dict:
    """Validate, round-trip, and write the form the browser sent.

    Returns ``{"ok": False, ...}`` with messages when the form is invalid (nothing
    is written). Raises :class:`ValueError` for a missing path, a non-boolean
    option, a bad value, a failed round-trip, or a write problem.
    """

    if not isinstance(payload, dict):
        raise ValueError("request must be a JSON object")
    path = payload.get("path")
    if not isinstance(path, str) or not path.strip():
        raise ValueError("path is required")
    force = _require_bool(payload.get("force", False), "force")
    strict = _require_bool(payload.get("strict_json", False), "strict_json")

    form = _form_from_payload(payload.get("form"))
    messages = validate_betting_tree_form(form)
    if messages:
        return {
            "ok": False,
            "valid": False,
            "messages": _messages_payload(messages),
            "error": "form has validation messages; not saved",
        }

    try:
        out_dict = betting_tree_form_to_dict(form)
        scenario = river_scenario_from_dict(out_dict)
        build_river_steal_game_from_scenario(scenario)
    except Exception as exc:  # noqa: BLE001 - surface as a clean round-trip error
        raise ValueError(f"round-trip failed, not saved: {exc}")

    text = _dump_json(out_dict, strict)
    _write_output(text, path.strip(), force, print_func=lambda *_args, **_kwargs: None)
    return {"ok": True, "path": path.strip()}


_API = {
    "/api/load": api_load,
    "/api/validate": api_validate,
    "/api/save": api_save,
}

# Inline single-file page: HTML + CSS + vanilla JS, no external resources.
_PAGE = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>Betting-tree scenario editor (prototype)</title>
<style>
  body { font-family: sans-serif; max-width: 1000px; margin: 1.5rem auto; padding: 0 1rem; }
  h1 { font-size: 1.3rem; }
  fieldset { margin: 1rem 0; }
  label { display: block; margin: 0.4rem 0; }
  label span { display: inline-block; width: 18rem; }
  input[type=text] { width: 22rem; }
  table { border-collapse: collapse; margin: 0.5rem 0; }
  th, td { border: 1px solid #ccc; padding: 0.25rem 0.4rem; }
  td input[type=text] { width: 7rem; }
  td input.matrix_cell { width: 6rem; }
  .row { margin: 0.5rem 0; }
  button { margin-right: 0.5rem; }
  #status { margin: 0.75rem 0; font-weight: bold; }
  #messages { color: #a00; white-space: pre-wrap; }
  .hint { color: #555; font-size: 0.85rem; }
</style>
</head>
<body>
<h1>Betting-tree scenario editor (local prototype)</h1>
<p class="hint">Local-only river betting-tree editor. Load a scenario JSON, edit the
sizes, the weighted Hero and Villain buckets, and the Hero x Villain matrix, then
validate and save. Each Hero bucket has two decision points: after an OOP check
(check / bet) and versus an OOP bet (call / fold / raise). The matrix is read as
showdown (hero / villain / chop) or equity (Hero pot share before rake in [0, 1]),
chosen by the matrix type; equity values are abstract inputs from the JSON, not
computed from real cards. Only betting-tree scenarios are supported here, and the
analysis pipeline is not run.</p>

<fieldset>
  <legend>Load</legend>
  <div class="row">
    <label><span>scenario path</span>
      <input type="text" id="load_path" placeholder="examples/scenarios/range_equity_betting_tree_bet98.json"></label>
    <button id="load_btn">Load</button>
  </div>
</fieldset>

<fieldset>
  <legend>Top-level fields</legend>
  <label><span>scenario_id</span><input type="text" id="scenario_id"></label>
  <label><span>description</span><input type="text" id="description"></label>
  <label><span>rake_rate</span><input type="text" id="rake_rate"></label>
  <label><span>rake_cap (blank = no cap)</span><input type="text" id="rake_cap"></label>
  <label><span>initial_commitment_hero</span><input type="text" id="initial_commitment_hero"></label>
  <label><span>initial_commitment_villain</span><input type="text" id="initial_commitment_villain"></label>
  <label><span>bet_size (must equal oop_bet_size)</span><input type="text" id="bet_size"></label>
  <label><span>shift_amounts (comma list)</span><input type="text" id="shift_amounts"></label>
  <label><span>horizons (comma list)</span><input type="text" id="horizons"></label>
  <label><span>discount</span><input type="text" id="discount"></label>
</fieldset>

<fieldset>
  <legend>Betting-tree sizes</legend>
  <p class="hint">All positive. ip_raise_size is the total committed once IP's raise
  is called (not the increment) and must be greater than oop_bet_size.</p>
  <label><span>oop_bet_size</span><input type="text" id="oop_bet_size"></label>
  <label><span>ip_bet_after_check_size</span><input type="text" id="ip_bet_after_check_size"></label>
  <label><span>ip_raise_size</span><input type="text" id="ip_raise_size"></label>
</fieldset>

<fieldset>
  <legend>Matrix type</legend>
  <p class="hint">Choose how each matrix cell is read. Switching type rebuilds the
  matrix with default cells (chop for showdown, 0.5 for equity).</p>
  <label><span>matrix_type</span>
    <select id="matrix_type">
      <option value="showdown">showdown</option>
      <option value="equity">equity</option>
    </select>
  </label>
</fieldset>

<fieldset>
  <legend>Hero buckets</legend>
  <p class="hint">Weighted abstract Hero buckets (weights sum to 1; ids unique and
  disjoint from Villain ids). after_oop_check (check + bet) and vs_oop_bet
  (call + fold + raise) must each sum to 1. There is no per-bucket showdown --
  outcomes come from the matrix.</p>
  <table>
    <thead>
      <tr>
        <th>hand_id</th><th>weight</th>
        <th>after_oop_check_check</th><th>after_oop_check_bet</th>
        <th>vs_oop_bet_call</th><th>vs_oop_bet_fold</th><th>vs_oop_bet_raise</th><th></th>
      </tr>
    </thead>
    <tbody id="hero_body"></tbody>
  </table>
  <button id="add_hero_btn">Add hero bucket</button>
</fieldset>

<fieldset>
  <legend>Villain buckets</legend>
  <p class="hint">Weighted abstract Villain buckets (weights sum to 1; ids unique
  and disjoint from Hero ids). Villain buckets carry no baseline strategy.</p>
  <table>
    <thead>
      <tr><th>hand_id</th><th>weight</th><th></th></tr>
    </thead>
    <tbody id="villain_body"></tbody>
  </table>
  <button id="add_villain_btn">Add villain bucket</button>
</fieldset>

<fieldset>
  <legend>Matrix</legend>
  <p class="hint">One cell per Hero (row) vs Villain (column) pairing. After editing
  bucket hand_ids or the matrix type, click "Rebuild matrix" to regenerate the grid;
  cells for matching ids are kept when the type is unchanged.</p>
  <button id="rebuild_matrix_btn">Rebuild matrix</button>
  <table>
    <thead><tr id="matrix_head"></tr></thead>
    <tbody id="matrix_body"></tbody>
  </table>
</fieldset>

<fieldset>
  <legend>Save</legend>
  <div class="row">
    <label><span>output path</span><input type="text" id="save_path"></label>
  </div>
  <div class="row">
    <label><input type="checkbox" id="force"> overwrite existing file</label>
    <label><input type="checkbox" id="strict_json"> strict JSON</label>
  </div>
  <button id="validate_btn">Validate</button>
  <button id="save_btn">Save JSON</button>
</fieldset>

<div id="status"></div>
<div id="messages"></div>

<script>
var TOP_FIELDS = ["scenario_id","description","rake_rate","rake_cap",
  "initial_commitment_hero","initial_commitment_villain","bet_size",
  "shift_amounts","horizons","discount"];
var SIZING_FIELDS = ["oop_bet_size","ip_bet_after_check_size","ip_raise_size"];
var HERO_FIELDS = ["hand_id","weight",
  "after_oop_check_check_probability","after_oop_check_bet_probability",
  "vs_oop_bet_call_probability","vs_oop_bet_fold_probability","vs_oop_bet_raise_probability"];
var VILLAIN_FIELDS = ["hand_id","weight"];
var SHOWDOWN_RESULTS = ["chop","hero","villain"];
var formatVersion = "1";
// Track the matrix type the current grid was built for so a type switch can
// regenerate default cells instead of mixing showdown strings and equity numbers.
var renderedMatrixType = "showdown";

function currentMatrixType() {
  return document.getElementById("matrix_type").value;
}
function defaultCell(type) {
  return type === "equity" ? "0.5" : "chop";
}
function makeBucketRow(fields, bucket) {
  bucket = bucket || {};
  var tr = document.createElement("tr");
  fields.forEach(function (f) {
    var td = document.createElement("td");
    var input = document.createElement("input");
    input.type = "text";
    input.setAttribute("data-field", f);
    var v = bucket[f];
    input.value = (v === null || v === undefined) ? "" : v;
    td.appendChild(input);
    tr.appendChild(td);
  });
  var td = document.createElement("td");
  var remove = document.createElement("button");
  remove.type = "button";
  remove.className = "remove_bucket";
  remove.textContent = "Remove";
  remove.onclick = function () {
    tr.parentNode.removeChild(tr);
    rebuildMatrix();
  };
  td.appendChild(remove);
  tr.appendChild(td);
  return tr;
}
function addHero(bucket) {
  document.getElementById("hero_body").appendChild(makeBucketRow(HERO_FIELDS, bucket));
}
function addVillain(bucket) {
  document.getElementById("villain_body").appendChild(makeBucketRow(VILLAIN_FIELDS, bucket));
}
function clearChildren(node) {
  while (node.firstChild) { node.removeChild(node.firstChild); }
}
function collectBuckets(bodyId, fields) {
  var out = [];
  document.querySelectorAll("#" + bodyId + " tr").forEach(function (row) {
    var bucket = {};
    row.querySelectorAll("input[data-field]").forEach(function (input) {
      bucket[input.getAttribute("data-field")] = input.value;
    });
    out.push(bucket);
  });
  return out;
}
function bucketIds(bodyId) {
  var ids = [];
  document.querySelectorAll("#" + bodyId + " tr").forEach(function (row) {
    var first = row.querySelector("input[data-field=hand_id]");
    ids.push(first ? first.value : "");
  });
  return ids;
}
// Snapshot the current matrix cells into a nested object keyed by hero/villain id.
function snapshotMatrix() {
  var snap = {};
  document.querySelectorAll("#matrix_body [data-hero]").forEach(function (el) {
    var h = el.getAttribute("data-hero");
    var v = el.getAttribute("data-villain");
    if (!snap[h]) { snap[h] = {}; }
    snap[h][v] = el.value;
  });
  return snap;
}
function makeCell(type, heroId, villainId, value) {
  var td = document.createElement("td");
  var el;
  if (type === "showdown") {
    el = document.createElement("select");
    SHOWDOWN_RESULTS.forEach(function (r) {
      var opt = document.createElement("option");
      opt.value = r;
      opt.textContent = r;
      if (r === value) { opt.selected = true; }
      el.appendChild(opt);
    });
  } else {
    el = document.createElement("input");
    el.type = "text";
    el.className = "matrix_cell";
    el.value = (value === null || value === undefined) ? "" : value;
  }
  el.setAttribute("data-hero", heroId);
  el.setAttribute("data-villain", villainId);
  td.appendChild(el);
  return td;
}
function rebuildMatrix() {
  var type = currentMatrixType();
  // Only keep existing cell values when the matrix type has not changed; a switch
  // regenerates default cells so showdown strings and equity numbers never mix.
  var prior = (type === renderedMatrixType) ? snapshotMatrix() : {};
  var fallback = defaultCell(type);
  var heroIds = bucketIds("hero_body");
  var villainIds = bucketIds("villain_body");
  var head = document.getElementById("matrix_head");
  var body = document.getElementById("matrix_body");
  clearChildren(head);
  clearChildren(body);

  var corner = document.createElement("th");
  corner.textContent = "hero \\\\ villain";
  head.appendChild(corner);
  villainIds.forEach(function (vid) {
    var th = document.createElement("th");
    th.textContent = vid;
    head.appendChild(th);
  });

  heroIds.forEach(function (hid) {
    var tr = document.createElement("tr");
    var th = document.createElement("th");
    th.textContent = hid;
    tr.appendChild(th);
    villainIds.forEach(function (vid) {
      var value = (prior[hid] && prior[hid][vid] !== undefined) ? prior[hid][vid] : fallback;
      tr.appendChild(makeCell(type, hid, vid, value));
    });
    body.appendChild(tr);
  });
  renderedMatrixType = type;
}
// Seed the matrix cells from a loaded matrix object before rebuilding so loaded
// values appear in the regenerated grid.
function seedMatrix(type, matrix) {
  var body = document.getElementById("matrix_body");
  clearChildren(document.getElementById("matrix_head"));
  clearChildren(body);
  renderedMatrixType = type;
  Object.keys(matrix || {}).forEach(function (hid) {
    var row = matrix[hid] || {};
    Object.keys(row).forEach(function (vid) {
      var tr = document.createElement("tr");
      tr.appendChild(makeCell(type, hid, vid, row[vid]));
      body.appendChild(tr);
    });
  });
}
function collectMatrix() {
  var matrix = {};
  document.querySelectorAll("#matrix_body [data-hero]").forEach(function (el) {
    var h = el.getAttribute("data-hero");
    var v = el.getAttribute("data-villain");
    if (!matrix[h]) { matrix[h] = {}; }
    matrix[h][v] = el.value;
  });
  return matrix;
}
function collectForm() {
  var form = {format_version: formatVersion};
  TOP_FIELDS.forEach(function (f) { form[f] = document.getElementById(f).value; });
  var betting = {};
  SIZING_FIELDS.forEach(function (f) { betting[f] = document.getElementById(f).value; });
  form.betting_tree = betting;
  form.matrix_type = currentMatrixType();
  form.hero_buckets = collectBuckets("hero_body", HERO_FIELDS);
  form.villain_buckets = collectBuckets("villain_body", VILLAIN_FIELDS);
  form.matrix = collectMatrix();
  return form;
}
function fillForm(form) {
  formatVersion = form.format_version || "1";
  TOP_FIELDS.forEach(function (f) {
    var v = form[f];
    document.getElementById(f).value = (v === null || v === undefined) ? "" : v;
  });
  var betting = form.betting_tree || {};
  SIZING_FIELDS.forEach(function (f) {
    var v = betting[f];
    document.getElementById(f).value = (v === null || v === undefined) ? "" : v;
  });
  var type = (form.matrix_type === "equity") ? "equity" : "showdown";
  document.getElementById("matrix_type").value = type;
  clearChildren(document.getElementById("hero_body"));
  (form.hero_buckets || []).forEach(function (b) { addHero(b); });
  clearChildren(document.getElementById("villain_body"));
  (form.villain_buckets || []).forEach(function (b) { addVillain(b); });
  // Seed loaded cell values, then rebuild against the current bucket ids so the
  // grid stays consistent with the bucket tables.
  seedMatrix(type, form.matrix);
  rebuildMatrix();
}
function setStatus(text, isError) {
  var s = document.getElementById("status");
  s.textContent = text;
  s.style.color = isError ? "#a00" : "#070";
}
function showMessages(messages) {
  var box = document.getElementById("messages");
  if (!messages || !messages.length) { box.textContent = ""; return; }
  box.textContent = messages.map(function (m) {
    return "[" + m.severity + "] " + m.field + ": " + m.message;
  }).join("\\n");
}
function post(url, body) {
  return fetch(url, {
    method: "POST",
    headers: {"Content-Type": "application/json"},
    body: JSON.stringify(body)
  }).then(function (r) { return r.json(); });
}

document.getElementById("add_hero_btn").onclick = function () { addHero(); rebuildMatrix(); };
document.getElementById("add_villain_btn").onclick = function () { addVillain(); rebuildMatrix(); };
document.getElementById("rebuild_matrix_btn").onclick = function () { rebuildMatrix(); };
document.getElementById("matrix_type").onchange = function () { rebuildMatrix(); };

document.getElementById("load_btn").onclick = function () {
  showMessages([]);
  post("/api/load", {path: document.getElementById("load_path").value})
    .then(function (res) {
      if (!res.ok) { setStatus("error: " + res.error, true); return; }
      fillForm(res.form);
      if (!document.getElementById("save_path").value) {
        document.getElementById("save_path").value = document.getElementById("load_path").value;
      }
      setStatus("loaded (" + res.mode + ")", false);
    })
    .catch(function () { setStatus("request failed", true); });
};

document.getElementById("validate_btn").onclick = function () {
  post("/api/validate", {form: collectForm()})
    .then(function (res) {
      if (!res.ok) { setStatus("error: " + res.error, true); showMessages([]); return; }
      showMessages(res.messages);
      setStatus(res.valid ? "valid" : (res.messages.length + " validation message(s)"), !res.valid);
    })
    .catch(function () { setStatus("request failed", true); });
};

document.getElementById("save_btn").onclick = function () {
  post("/api/save", {
    path: document.getElementById("save_path").value,
    form: collectForm(),
    force: document.getElementById("force").checked,
    strict_json: document.getElementById("strict_json").checked
  })
    .then(function (res) {
      if (!res.ok) {
        setStatus("error: " + (res.error || "not saved"), true);
        showMessages(res.messages || []);
        return;
      }
      showMessages([]);
      setStatus("saved " + res.path, false);
    })
    .catch(function () { setStatus("request failed", true); });
};
</script>
</body>
</html>
"""


def build_server(host: str, port: int) -> ThreadingHTTPServer:
    """Create (but do not start) the local GUI server bound to ``host:port``."""

    return _build_server(host, port, _API, _PAGE)


def _parse_args(argv):
    parser = argparse.ArgumentParser(
        description="Serve a minimal local GUI for editing a river betting-tree scenario."
    )
    parser.add_argument(
        "--host",
        default="127.0.0.1",
        help="host to bind (default 127.0.0.1, local only)",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8004,
        help="port to bind (default 8004)",
    )
    return parser.parse_args(argv)


def main(argv) -> int:
    args = _parse_args(argv)
    try:
        server = build_server(args.host, args.port)
    except OSError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    host, port = server.server_address[0], server.server_address[1]
    print(f"serving betting-tree scenario GUI at http://{host}:{port}/ (Ctrl+C to stop)")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
