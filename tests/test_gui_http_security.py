"""HTTP request-security integration tests for every shared local GUI handler."""

import http.client
import json
import sys
import threading
from contextlib import contextmanager
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
_SCENARIO = _ROOT / "examples" / "scenarios" / "nuts_chop_steal_bet98.json"

sys.path.insert(0, str(_ROOT / "src"))
sys.path.insert(0, str(_ROOT / "scripts"))
import gui_common  # noqa: E402
import serve_single_hand_gui as single_hand_gui  # noqa: E402


@contextmanager
def _running_server(api, page="<html>safe GUI</html>"):
    server = gui_common.build_server("127.0.0.1", 0, api, page)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield server.server_address[1]
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)


def _request(port, method, path, *, body=b"", headers=None):
    connection = http.client.HTTPConnection("127.0.0.1", port, timeout=5)
    try:
        connection.request(method, path, body=body, headers=headers or {})
        response = connection.getresponse()
        response_body = response.read()
        return response.status, dict(response.getheaders()), response_body
    finally:
        connection.close()


def _json_body(payload):
    return json.dumps(payload).encode("utf-8")


def _same_origin_headers(port, content_type="application/json"):
    return {
        "Content-Type": content_type,
        "Origin": f"http://127.0.0.1:{port}",
    }


def _spy_api():
    calls = []

    def handler(payload):
        calls.append(payload)
        return {"ok": True, "payload": payload}

    return {"/api/test": handler}, calls


def test_same_origin_application_json_post_succeeds():
    api, calls = _spy_api()
    with _running_server(api) as port:
        status, _, body = _request(
            port,
            "POST",
            "/api/test",
            body=_json_body({"value": 7}),
            headers=_same_origin_headers(
                port,
                "application/json; charset=utf-8",
            ),
        )

    assert status == 200
    assert json.loads(body)["ok"] is True
    assert calls == [{"value": 7}]


def test_originless_application_json_non_browser_client_succeeds():
    api, calls = _spy_api()
    with _running_server(api) as port:
        status, _, body = _request(
            port,
            "POST",
            "/api/test",
            body=_json_body({"client": "urllib-compatible"}),
            headers={"Content-Type": "application/json"},
        )

    assert status == 200
    assert json.loads(body)["ok"] is True
    assert calls == [{"client": "urllib-compatible"}]


def test_text_plain_json_is_rejected_before_handler():
    api, calls = _spy_api()
    with _running_server(api) as port:
        status, _, body = _request(
            port,
            "POST",
            "/api/test",
            body=_json_body({"would": "dispatch"}),
            headers=_same_origin_headers(port, "text/plain"),
        )

    assert status == 415
    assert json.loads(body)["ok"] is False
    assert calls == []


def test_foreign_origin_is_rejected_before_handler():
    api, calls = _spy_api()
    with _running_server(api) as port:
        status, _, body = _request(
            port,
            "POST",
            "/api/test",
            body=_json_body({"would": "dispatch"}),
            headers={
                "Content-Type": "application/json",
                "Origin": "https://attacker.example",
            },
        )

    assert status == 403
    assert json.loads(body)["ok"] is False
    assert calls == []


def test_sec_fetch_site_cross_site_is_rejected_before_handler():
    api, calls = _spy_api()
    with _running_server(api) as port:
        status, _, body = _request(
            port,
            "POST",
            "/api/test",
            body=_json_body({"would": "dispatch"}),
            headers={
                "Content-Type": "application/json",
                "Sec-Fetch-Site": "cross-site",
            },
        )

    assert status == 403
    assert json.loads(body)["ok"] is False
    assert calls == []


def test_invalid_host_is_rejected_before_handler():
    api, calls = _spy_api()
    with _running_server(api) as port:
        status, _, body = _request(
            port,
            "POST",
            "/api/test",
            body=_json_body({"would": "dispatch"}),
            headers={
                "Content-Type": "application/json",
                "Host": f"127.0.0.1:{port + 1}",
                "Origin": f"http://127.0.0.1:{port}",
            },
        )

    assert status == 403
    assert json.loads(body)["ok"] is False
    assert calls == []


def test_cross_origin_preflight_is_denied_without_cors_headers():
    api, calls = _spy_api()
    with _running_server(api) as port:
        status, headers, body = _request(
            port,
            "OPTIONS",
            "/api/test",
            headers={
                "Origin": "https://attacker.example",
                "Access-Control-Request-Method": "POST",
                "Access-Control-Request-Headers": "content-type",
            },
        )

    assert status == 403
    assert json.loads(body)["ok"] is False
    assert not any(name.lower().startswith("access-control-allow") for name in headers)
    assert calls == []


def test_get_and_same_origin_validate_save_workflows_remain_available(tmp_path):
    assert not tmp_path.resolve().is_relative_to(_ROOT.resolve())
    form = single_hand_gui.api_load({"path": str(_SCENARIO)})["form"]
    destination = tmp_path / "accepted" / "scenario.json"

    with _running_server(single_hand_gui._API, single_hand_gui._PAGE) as port:
        get_status, _, page = _request(port, "GET", "/")
        validate_status, _, validate_body = _request(
            port,
            "POST",
            "/api/validate",
            body=_json_body({"form": form}),
            headers=_same_origin_headers(port),
        )
        save_status, _, save_body = _request(
            port,
            "POST",
            "/api/save",
            body=_json_body({"path": str(destination), "form": form}),
            headers=_same_origin_headers(port),
        )

    assert get_status == 200
    assert b"Single-hand scenario form" in page
    assert validate_status == 200
    assert json.loads(validate_body)["valid"] is True
    assert save_status == 200
    assert json.loads(save_body)["ok"] is True
    assert destination.is_file()


def test_rejected_save_does_not_create_file_or_call_handler(tmp_path):
    assert not tmp_path.resolve().is_relative_to(_ROOT.resolve())
    form = single_hand_gui.api_load({"path": str(_SCENARIO)})["form"]
    destination = tmp_path / "new-parent" / "scenario.json"
    original_handler = single_hand_gui._API["/api/save"]
    calls = []

    def guarded_handler(payload):
        calls.append(payload)
        return original_handler(payload)

    single_hand_gui._API["/api/save"] = guarded_handler
    try:
        with _running_server(single_hand_gui._API, single_hand_gui._PAGE) as port:
            status, _, _ = _request(
                port,
                "POST",
                "/api/save",
                body=_json_body(
                    {"path": str(destination), "form": form, "force": True}
                ),
                headers={
                    "Content-Type": "text/plain",
                    "Origin": "https://attacker.example",
                    "Sec-Fetch-Site": "cross-site",
                },
            )
    finally:
        single_hand_gui._API["/api/save"] = original_handler

    assert status == 403
    assert calls == []
    assert not destination.exists()
    assert not destination.parent.exists()


def test_rejected_force_save_does_not_overwrite_file_or_call_handler(tmp_path):
    assert not tmp_path.resolve().is_relative_to(_ROOT.resolve())
    form = single_hand_gui.api_load({"path": str(_SCENARIO)})["form"]
    destination = tmp_path / "existing.json"
    destination.write_text("ORIGINAL", encoding="utf-8")
    original_handler = single_hand_gui._API["/api/save"]
    calls = []

    def guarded_handler(payload):
        calls.append(payload)
        return original_handler(payload)

    single_hand_gui._API["/api/save"] = guarded_handler
    try:
        with _running_server(single_hand_gui._API, single_hand_gui._PAGE) as port:
            status, _, _ = _request(
                port,
                "POST",
                "/api/save",
                body=_json_body(
                    {"path": str(destination), "form": form, "force": True}
                ),
                headers={
                    "Content-Type": "application/json",
                    "Origin": "https://attacker.example",
                },
            )
    finally:
        single_hand_gui._API["/api/save"] = original_handler

    assert status == 403
    assert calls == []
    assert destination.read_text(encoding="utf-8") == "ORIGINAL"
