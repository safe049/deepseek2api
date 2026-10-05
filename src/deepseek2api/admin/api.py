from __future__ import annotations

import json
import time
import traceback
from http import HTTPStatus
from logging import Logger
from pathlib import Path
from typing import Any, Dict, Optional

from ..config import AppConfig
from ..core.model_profiles import get_all_models
from .store import (
    GLOBAL_STORE,
    RequestRecord,
    classify_protocol,
    get_store,
)

_STATIC_DIR = Path(__file__).parent / "static"

_MIME_TYPES = {
    ".html": "text/html; charset=utf-8",
    ".js": "application/javascript; charset=utf-8",
    ".css": "text/css; charset=utf-8",
    ".svg": "image/svg+xml",
    ".png": "image/png",
    ".ico": "image/x-icon",
    ".json": "application/json; charset=utf-8",
}


def _resolve_static(path: str) -> Optional[Path]:
    relpath = path
    if relpath.startswith("/admin/static/"):
        relpath = relpath[len("/admin/static/"):]
    elif relpath == "/admin/static":
        relpath = ""

    candidate = (_STATIC_DIR / relpath).resolve()
    try:
        candidate.relative_to(_STATIC_DIR.resolve())
    except ValueError:
        return None
    if not candidate.is_file():
        return None
    return candidate


def _authorize_admin(handler, config: AppConfig, logger: Logger) -> bool:
    token = handler.headers.get("X-Admin-Token", "").strip()
    if not token:
        from urllib.parse import urlparse, parse_qs
        qs = parse_qs(urlparse(handler.path).query)
        if "token" in qs and qs["token"]:
            token = qs["token"][0].strip()
    return get_store().validate_session(token)


def _send_json(handler, status: HTTPStatus, payload: Dict[str, Any], config: AppConfig) -> None:
    body = json.dumps(payload, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    handler.send_response(status)
    handler.send_header("Content-Type", "application/json; charset=utf-8")
    handler.send_header("Content-Length", str(len(body)))
    handler.send_header("Cache-Control", "no-store, no-cache, must-revalidate")
    handler.end_headers()
    handler.wfile.write(body)


def _send_static(handler, file_path: Path, config: AppConfig) -> None:
    try:
        body = file_path.read_bytes()
    except OSError:
        handler.send_error(HTTPStatus.NOT_FOUND, "Not Found")
        return

    suffix = file_path.suffix.lower()
    content_type = _MIME_TYPES.get(suffix, "application/octet-stream")

    handler.send_response(HTTPStatus.OK)
    handler.send_header("Content-Type", content_type)
    handler.send_header("Content-Length", str(len(body)))
    handler.send_header("Cache-Control", "no-cache")
    handler.end_headers()
    handler.wfile.write(body)


def _read_json_body(handler) -> Dict[str, Any]:
    content_length = int(handler.headers.get("Content-Length", "0") or "0")
    if content_length <= 0:
        return {}
    raw = handler.rfile.read(content_length)
    if not raw:
        return {}
    return json.loads(raw.decode("utf-8"))


def _handle_login(handler, config: AppConfig, logger: Logger) -> None:
    try:
        payload = _read_json_body(handler)
    except (json.JSONDecodeError, UnicodeDecodeError):
        _send_json(handler, HTTPStatus.BAD_REQUEST, {"error": "invalid_json"}, config)
        return

    password = str(payload.get("password", "")).strip()
    if password != config.admin_password:
        _send_json(handler, HTTPStatus.UNAUTHORIZED, {"error": "invalid_password"}, config)
        return

    ttl = 8 * 3600
    token = get_store().create_session(ttl_seconds=ttl)
    _send_json(handler, HTTPStatus.OK, {"token": token, "expires_in": ttl}, config)


def _handle_logout(handler, config: AppConfig, logger: Logger) -> None:
    token = handler.headers.get("X-Admin-Token", "").strip()
    if token:
        get_store().revoke_session(token)
    _send_json(handler, HTTPStatus.OK, {"ok": True}, config)


def _handle_dashboard(handler, config: AppConfig, logger: Logger) -> None:
    _send_json(handler, HTTPStatus.OK, get_store().dashboard(), config)


def _handle_logs(handler, config: AppConfig, logger: Logger) -> None:
    from urllib.parse import urlparse, parse_qs
    qs = parse_qs(urlparse(handler.path).query)
    limit = int(qs.get("limit", ["100"])[0])
    only_errors = qs.get("errors", ["0"])[0] in ("1", "true", "yes")
    limit = max(1, min(limit, 500))
    _send_json(handler, HTTPStatus.OK, {"logs": get_store().recent_logs(limit=limit, only_errors=only_errors)}, config)


def _handle_models(handler, config: AppConfig, logger: Logger) -> None:
    models = get_all_models()
    _send_json(handler, HTTPStatus.OK, {"models": models}, config)


def _handle_config(handler, config: AppConfig, logger: Logger) -> None:
    sanitized = {
        "host": config.host,
        "port": config.port,
        "api_prefix": config.api_prefix,
        "log_level": config.log_level,
        "debug_dump_all": config.debug_dump_all,
        "request_timeout": config.request_timeout,
        "default_model": config.default_model,
        "delete_conversation": config.delete_conversation,
        "max_concurrency": config.max_concurrency,
        "queue_wait_timeout": config.queue_wait_timeout,
        "base_url": config.base_url,
    }
    _send_json(handler, HTTPStatus.OK, sanitized, config)


_API_ROUTES = {
    "login": ("POST", _handle_login),
    "logout": ("POST", _handle_logout),
    "dashboard": ("GET", _handle_dashboard),
    "logs": ("GET", _handle_logs),
    "models": ("GET", _handle_models),
    "config": ("GET", _handle_config),
}


def handle_admin_request(handler, config: AppConfig, glm_client, logger: Logger) -> bool:
    from urllib.parse import urlparse
    path = urlparse(handler.path).path

    if handler.command == "OPTIONS":
        handler.send_response(HTTPStatus.NO_CONTENT)
        handler.end_headers()
        return True

    if not (path == "/admin" or path == "/admin/" or path.startswith("/admin/")):
        return False

    if path == "/admin" or path == "/admin/" or path == "/admin/index.html":
        index_file = _STATIC_DIR / "index.html"
        _send_static(handler, index_file, config)
        return True

    if path.startswith("/admin/static/"):
        file_path = _resolve_static(path)
        if file_path is None:
            handler.send_error(HTTPStatus.NOT_FOUND, "Not Found")
            return True
        _send_static(handler, file_path, config)
        return True

    if path.startswith("/admin/api/"):
        name = path[len("/admin/api/"):].rstrip("/")

        if name not in _API_ROUTES:
            _send_json(handler, HTTPStatus.NOT_FOUND, {"error": "unknown_endpoint"}, config)
            return True

        expected_method, fn = _API_ROUTES[name]
        if handler.command != expected_method:
            _send_json(handler, HTTPStatus.METHOD_NOT_ALLOWED, {"error": "method_not_allowed"}, config)
            return True

        if name != "login":
            if not _authorize_admin(handler, config, logger):
                _send_json(handler, HTTPStatus.UNAUTHORIZED, {"error": "unauthorized"}, config)
                return True

        try:
            fn(handler, config, logger)
        except Exception as exc:
            logger.error("admin endpoint failed path=%s error=%s\n%s", path, exc, traceback.format_exc())
            _send_json(handler, HTTPStatus.INTERNAL_SERVER_ERROR, {"error": str(exc)}, config)
        return True

    index_file = _STATIC_DIR / "index.html"
    _send_static(handler, index_file, config)
    return True


def record_request(
    *,
    method: str,
    path: str,
    protocol: str,
    model: str,
    status: int,
    duration_ms: int,
    client_ip: str,
    stream: bool,
    error: str,
    request_id: str,
) -> None:
    rec = RequestRecord(
        ts=time.time(),
        method=method,
        path=path,
        protocol=protocol,
        model=model,
        status=status,
        duration_ms=duration_ms,
        client_ip=client_ip,
        stream=stream,
        error=error,
        request_id=request_id,
    )
    get_store().record_request(rec)