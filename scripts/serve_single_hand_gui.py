#!/usr/bin/env python3
"""Serve a minimal local GUI for editing a single-hand scenario (prototype v1).

Usage:

    python scripts/serve_single_hand_gui.py --port 8000
    # then open http://127.0.0.1:8000/ in a browser

This is a pre-GUI-product prototype: a tiny local-only web page that loads a
single-hand scenario JSON into a form, lets you edit the fields, validates the
form, and saves it back to JSON -- the browser equivalent of
``scripts/edit_scenario_form.py``. It is built on the Python standard library only
(``http.server`` plus inline HTML / CSS / vanilla JavaScript); it adds no
framework or dependency, and no new solver, model, or analysis logic.

Scope (v1): single-hand mode only. Range / matrix / betting-tree editing, the
analysis pipeline, candidate generation, charts, real-card parsing, and external
solver imports are all out of scope. It reuses the existing form model, parser,
and game builder as the source of truth.

Endpoints:

* ``GET  /``            -- the HTML form page;
* ``POST /api/load``    -- ``{"path": "..."}`` -> the form fields of that
  single-hand scenario (a non-single-hand scenario is rejected);
* ``POST /api/validate``-- ``{"form": {...}}`` -> field-level validation messages;
* ``POST /api/save``    -- ``{"path", "form", "force", "strict_json"}`` -> writes
  the JSON only when the form validates and its ``to_dict`` re-parses and rebuilds.

Security / safety: it binds to ``127.0.0.1`` by default and makes no external
calls. It reads and writes only the local paths you type into the form, refuses to
overwrite an existing file unless the overwrite box is checked, and returns short
``error`` messages (never a traceback) on failure.
"""

from __future__ import annotations

import argparse
import math
import sys
from http.server import ThreadingHTTPServer
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))  # for the sibling CLIs

from repeated_poker import (  # noqa: E402  (path is set up above)
    RiverScenarioAnalysisConfig,
    SingleHandScenarioForm,
    build_river_steal_game_from_scenario,
    detect_scenario_form_mode,
    river_scenario_from_dict,
    run_river_scenario_analysis,
    single_hand_form_from_dict,
    single_hand_form_to_dict,
    validate_single_hand_form,
)
from repeated_poker.report_export import _dump_json  # noqa: E402

# Reuse the loader, the safe file writer, and the field value parsing from the
# sibling form CLIs rather than duplicating them.
from inspect_scenario_form import _load_scenario_dict  # noqa: E402
from roundtrip_scenario_form import _write_output  # noqa: E402
from edit_scenario_form import _KNOWN_FIELDS, _convert_field  # noqa: E402

# Shared local-GUI scaffolding (HTTP handler / server builder) and small payload
# primitives, factored out of the sibling GUI scripts.
from gui_common import build_server as _build_server  # noqa: E402
from gui_common import messages_payload as _messages_payload  # noqa: E402

# The fields shown in the form, in display order (the flat SingleHandScenarioForm
# fields). shift_amounts / horizons are edited as comma-separated text.
_FORM_FIELDS = [
    "scenario_id",
    "description",
    "rake_rate",
    "rake_cap",
    "initial_commitment_hero",
    "initial_commitment_villain",
    "bet_size",
    "showdown",
    "baseline_call_probability",
    "baseline_fold_probability",
    "shift_amounts",
    "horizons",
    "discount",
]


def _form_to_payload(form: SingleHandScenarioForm) -> dict:
    """Flatten a form into the string-friendly dict the browser displays."""

    return {
        "format_version": form.format_version,
        "scenario_id": form.scenario_id,
        "description": form.description,
        "rake_rate": form.rake_rate,
        "rake_cap": "" if form.rake_cap is None else form.rake_cap,
        "initial_commitment_hero": form.initial_commitment_hero,
        "initial_commitment_villain": form.initial_commitment_villain,
        "bet_size": form.bet_size,
        "showdown": form.showdown,
        "baseline_call_probability": form.baseline_call_probability,
        "baseline_fold_probability": form.baseline_fold_probability,
        "shift_amounts": ",".join(str(value) for value in form.shift_amounts),
        "horizons": ",".join(str(value) for value in form.horizons),
        "discount": form.discount,
    }


def _form_from_payload(payload) -> SingleHandScenarioForm:
    """Build a form from the browser's flat dict, reusing the edit-CLI parsers.

    Values arrive as strings (HTML inputs); each known field is converted with the
    same per-field rules as ``edit_scenario_form`` (floats, optional ``rake_cap``,
    comma-separated ``shift_amounts`` / ``horizons``). ``format_version`` is carried
    through unedited and *not* coerced -- a non-string / unsupported value is kept
    as-is so ``validate_single_hand_form`` / the parser flag it, rather than being
    silently rounded to a valid ``"1"``. Raises :class:`ValueError` on a bad value.
    """

    if not isinstance(payload, dict):
        raise ValueError("form must be a JSON object")
    form = SingleHandScenarioForm()
    # Keep the raw value (no str() coercion); only fall back to the default when
    # the field is absent.
    if "format_version" in payload:
        form.format_version = payload["format_version"]
    for field in _KNOWN_FIELDS:
        if field not in payload:
            continue
        raw = payload[field]
        raw_str = "" if raw is None else str(raw)
        setattr(form, field, _convert_field(field, raw_str))
    return form


def api_load(payload) -> dict:
    """Load a single-hand scenario file into form fields for the browser."""

    if not isinstance(payload, dict):
        raise ValueError("request must be a JSON object")
    path = payload.get("path")
    if not isinstance(path, str) or not path.strip():
        raise ValueError("path is required")
    data = _load_scenario_dict(path.strip())
    mode = detect_scenario_form_mode(data)
    if mode != "single-hand":
        raise ValueError(
            f"GUI prototype supports single-hand mode only; this scenario is {mode} mode"
        )
    form = single_hand_form_from_dict(data)
    return {"ok": True, "mode": mode, "form": _form_to_payload(form)}


def api_validate(payload) -> dict:
    """Validate the form the browser sent, returning field-level messages."""

    if not isinstance(payload, dict):
        raise ValueError("request must be a JSON object")
    form = _form_from_payload(payload.get("form"))
    messages = validate_single_hand_form(form)
    return {"ok": True, "valid": not messages, "messages": _messages_payload(messages)}


def api_save(payload) -> dict:
    """Validate, round-trip, and write the form the browser sent.

    Returns ``{"ok": False, ...}`` with messages when the form is invalid (nothing
    is written). Raises :class:`ValueError` for a missing path, a bad value, a
    failed round-trip, or a write problem.
    """

    if not isinstance(payload, dict):
        raise ValueError("request must be a JSON object")
    path = payload.get("path")
    if not isinstance(path, str) or not path.strip():
        raise ValueError("path is required")
    # Require real booleans: a string like "false" must not silently enable an
    # overwrite or change the serialiser.
    force = payload.get("force", False)
    if not isinstance(force, bool):
        raise ValueError("force must be a boolean")
    strict = payload.get("strict_json", False)
    if not isinstance(strict, bool):
        raise ValueError("strict_json must be a boolean")

    form = _form_from_payload(payload.get("form"))
    messages = validate_single_hand_form(form)
    if messages:
        return {
            "ok": False,
            "valid": False,
            "messages": _messages_payload(messages),
            "error": "form has validation messages; not saved",
        }

    try:
        out_dict = single_hand_form_to_dict(form)
        scenario = river_scenario_from_dict(out_dict)
        build_river_steal_game_from_scenario(scenario)
    except Exception as exc:  # noqa: BLE001 - surface as a clean round-trip error
        raise ValueError(f"round-trip failed, not saved: {exc}")

    text = _dump_json(out_dict, strict)
    _write_output(text, path.strip(), force, print_func=lambda *_args, **_kwargs: None)
    return {"ok": True, "path": path.strip()}


def _optional_horizon(value):
    """Validate an optional horizon override (a positive int, ``None`` to skip)."""

    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError("horizon must be an integer")
    if value < 1:
        raise ValueError("horizon must be at least 1")
    return value


def _optional_discount(value):
    """Validate an optional discount override (a finite positive number)."""

    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError("discount must be a number")
    discount = float(value)
    if not math.isfinite(discount) or discount <= 0:
        raise ValueError("discount must be a finite positive number")
    return discount


def api_analyze(payload) -> dict:
    """Run the candidate analysis for the form the browser sent.

    Returns ``{"ok": False, ...}`` with messages when the form is invalid (the
    analysis is not run). Raises :class:`ValueError` for a bad value or a bad
    ``horizon`` / ``discount`` / ``render_markdown`` option; any other failure
    propagates to the handler, which returns a generic "internal error" without a
    traceback. The form supplies the default horizon (max of its horizons) and
    discount; ``horizon`` / ``discount`` in the payload override them.
    """

    if not isinstance(payload, dict):
        raise ValueError("request must be a JSON object")
    render_markdown = payload.get("render_markdown", True)
    if not isinstance(render_markdown, bool):
        raise ValueError("render_markdown must be a boolean")
    horizon = _optional_horizon(payload.get("horizon"))
    discount = _optional_discount(payload.get("discount"))

    form = _form_from_payload(payload.get("form"))
    messages = validate_single_hand_form(form)
    if messages:
        return {
            "ok": False,
            "valid": False,
            "messages": _messages_payload(messages),
            "error": "form has validation messages; not analyzed",
        }

    # A valid form re-parses and rebuilds; run_river_scenario_analysis does the
    # build itself, so any structural problem surfaces as a clean ValueError.
    scenario = river_scenario_from_dict(single_hand_form_to_dict(form))
    config = RiverScenarioAnalysisConfig(
        horizon=horizon, discount=discount, markdown=render_markdown
    )
    result = run_river_scenario_analysis(scenario, config)

    counts = result.pipeline_result.filter_result.summary_counts
    return {
        "ok": True,
        "valid": True,
        "scenario_id": result.scenario_id,
        "horizon": result.horizon,
        "discount": result.discount,
        "generated_count": len(result.pipeline_result.generated_candidates),
        "kept_count": counts.kept,
        "excluded_count": counts.excluded,
        "markdown_summary": result.markdown_summary,
    }


_API = {
    "/api/load": api_load,
    "/api/validate": api_validate,
    "/api/save": api_save,
    "/api/analyze": api_analyze,
}

# Inline single-file page: HTML + CSS + vanilla JS, no external resources.
_PAGE = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>Single-hand scenario form (prototype)</title>
<style>
  body { font-family: sans-serif; max-width: 760px; margin: 1.5rem auto; padding: 0 1rem; }
  h1 { font-size: 1.3rem; }
  fieldset { margin: 1rem 0; }
  label { display: block; margin: 0.4rem 0; }
  label span { display: inline-block; width: 16rem; }
  input[type=text] { width: 22rem; }
  .row { margin: 0.5rem 0; }
  button { margin-right: 0.5rem; }
  #status { margin: 0.75rem 0; font-weight: bold; }
  #messages { color: #a00; white-space: pre-wrap; }
  .hint { color: #555; font-size: 0.85rem; }
</style>
</head>
<body>
<h1>Single-hand scenario form (local prototype)</h1>
<p class="hint">Local-only single-hand editor. Load a scenario JSON, edit fields,
validate, and save. Range / matrix / betting-tree scenarios are not supported.</p>

<fieldset>
  <legend>Load</legend>
  <div class="row">
    <label><span>scenario path</span>
      <input type="text" id="load_path" placeholder="examples/scenarios/nuts_chop_steal_bet98.json"></label>
    <button id="load_btn">Load</button>
  </div>
</fieldset>

<fieldset>
  <legend>Fields</legend>
  <label><span>scenario_id</span><input type="text" id="scenario_id"></label>
  <label><span>description</span><input type="text" id="description"></label>
  <label><span>rake_rate</span><input type="text" id="rake_rate"></label>
  <label><span>rake_cap (blank = no cap)</span><input type="text" id="rake_cap"></label>
  <label><span>initial_commitment_hero</span><input type="text" id="initial_commitment_hero"></label>
  <label><span>initial_commitment_villain</span><input type="text" id="initial_commitment_villain"></label>
  <label><span>bet_size</span><input type="text" id="bet_size"></label>
  <label><span>showdown (hero/villain/chop)</span><input type="text" id="showdown"></label>
  <label><span>baseline_call_probability</span><input type="text" id="baseline_call_probability"></label>
  <label><span>baseline_fold_probability</span><input type="text" id="baseline_fold_probability"></label>
  <label><span>shift_amounts (comma list)</span><input type="text" id="shift_amounts"></label>
  <label><span>horizons (comma list)</span><input type="text" id="horizons"></label>
  <label><span>discount</span><input type="text" id="discount"></label>
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

<fieldset>
  <legend>Analyze</legend>
  <p class="hint">Runs the candidate analysis for the current form values (no file
  needed). Single-hand only; shows candidate counts and the Markdown summary.</p>
  <label><span>horizon override (blank = default)</span>
    <input type="text" id="opt_horizon" placeholder="e.g. 100"></label>
  <label><span>discount override (blank = default)</span>
    <input type="text" id="opt_discount" placeholder="e.g. 1.0"></label>
  <label><input type="checkbox" id="render_markdown" checked> render Markdown summary</label>
  <button id="analyze_btn">Analyze</button>
  <div id="analysis_result">
    <div id="analysis_counts"></div>
    <pre id="analysis_summary"></pre>
  </div>
</fieldset>

<div id="status"></div>
<div id="messages"></div>

<script>
var FIELDS = ["scenario_id","description","rake_rate","rake_cap",
  "initial_commitment_hero","initial_commitment_villain","bet_size","showdown",
  "baseline_call_probability","baseline_fold_probability","shift_amounts",
  "horizons","discount"];
var formatVersion = "1";

function collectForm() {
  var form = {format_version: formatVersion};
  FIELDS.forEach(function (f) { form[f] = document.getElementById(f).value; });
  return form;
}
function fillForm(form) {
  formatVersion = form.format_version || "1";
  FIELDS.forEach(function (f) {
    var v = form[f];
    document.getElementById(f).value = (v === null || v === undefined) ? "" : v;
  });
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
function clearAnalysisResult() {
  document.getElementById("analysis_counts").textContent = "";
  document.getElementById("analysis_summary").textContent = "";
}
function clearMessagesAndAnalysis() {
  showMessages([]);
  clearAnalysisResult();
}
function post(url, body) {
  return fetch(url, {
    method: "POST",
    headers: {"Content-Type": "application/json"},
    body: JSON.stringify(body)
  }).then(function (r) { return r.json(); });
}

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

// Parse an optional numeric override field: blank -> omit; otherwise a number.
// A non-numeric entry is reported client-side so it is not silently dropped.
function parseOption(elementId, label) {
  var text = document.getElementById(elementId).value.trim();
  if (text === "") { return {present: false}; }
  var n = Number(text);
  if (!isFinite(n)) { return {present: true, error: label + " must be a number"}; }
  return {present: true, value: n};
}

document.getElementById("analyze_btn").onclick = function () {
  // Clear stale validation messages and any previous analysis result up front, so
  // a client-side parse-error return below or a re-run never leaves old output.
  clearMessagesAndAnalysis();
  setStatus("analyzing...", false);

  var payload = {
    form: collectForm(),
    render_markdown: document.getElementById("render_markdown").checked
  };
  var horizon = parseOption("opt_horizon", "horizon");
  if (horizon.present) {
    if (horizon.error) { setStatus("error: " + horizon.error, true); return; }
    payload.horizon = horizon.value;
  }
  var discount = parseOption("opt_discount", "discount");
  if (discount.present) {
    if (discount.error) { setStatus("error: " + discount.error, true); return; }
    payload.discount = discount.value;
  }

  post("/api/analyze", payload)
    .then(function (res) {
      if (!res.ok) {
        setStatus("error: " + (res.error || "not analyzed"), true);
        showMessages(res.messages || []);
        return;
      }
      showMessages([]);
      // Analysis result is shown in its own area, separate from status/messages.
      document.getElementById("analysis_counts").textContent =
        "scenario_id=" + res.scenario_id +
        "  |  horizon=" + res.horizon + "  discount=" + res.discount +
        "  |  generated=" + res.generated_count +
        "  kept=" + res.kept_count + "  excluded=" + res.excluded_count;
      // Render the summary as text only (never as HTML) to avoid injection.
      if (res.markdown_summary === null || res.markdown_summary === undefined) {
        document.getElementById("analysis_summary").textContent =
          "(Markdown summary not rendered; enable \\"render Markdown summary\\")";
      } else {
        document.getElementById("analysis_summary").textContent = res.markdown_summary;
      }
      setStatus("analyzed", false);
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
        description="Serve a minimal local GUI for editing a single-hand scenario."
    )
    parser.add_argument(
        "--host",
        default="127.0.0.1",
        help="host to bind (default 127.0.0.1, local only)",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8000,
        help="port to bind (default 8000)",
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
    print(f"serving single-hand scenario GUI at http://{host}:{port}/ (Ctrl+C to stop)")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
