from __future__ import annotations

import json
import time
import traceback
import uuid
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from logging import Logger
from urllib.parse import urlparse

from .admin.api import handle_admin_request, record_request as admin_record_request
from .admin.store import classify_protocol as admin_classify_protocol
from .config import AppConfig, EXPOSED_MODELS
from .logging_utils import debug_dump
from .services.deepseek_client import DeepSeekWebClient, QueueTimeoutError, UpstreamAPIError
from .core.openai_compat import (
    ERROR_AUTHENTICATION,
    ERROR_INVALID_REQUEST,
    ERROR_NOT_FOUND,
    ERROR_SERVER,
    ERROR_UPSTREAM,
    gen_chatcmpl_id,
    gen_request_id,
    make_error,
    system_fingerprint,
    now_timestamp,
)

_CLIENT_DISCONNECTED = (BrokenPipeError, ConnectionResetError, ConnectionAbortedError)


class DeepSeek2APIServer:
    def __init__(self, config: AppConfig, deepseek_client: DeepSeekWebClient, logger: Logger) -> None:
        self.config = config
        self.deepseek_client = deepseek_client
        self.logger = logger

        handler_cls = self._build_handler()
        self._server = ThreadingHTTPServer((config.host, config.port), handler_cls)
        self._server.daemon_threads = True
        self._server.allow_reuse_address = True

    def serve_forever(self) -> None:
        self._server.serve_forever()

    def shutdown(self) -> None:
        self._server.shutdown()
        self._server.server_close()

    def _build_handler(self):
        config = self.config
        deepseek_client = self.deepseek_client
        logger = self.logger

        class RequestHandler(BaseHTTPRequestHandler):
            server_version = "cloudflare"
            sys_version = ""
            protocol_version = "HTTP/1.1"

            def _send_cors_headers(self):
                self.send_header("Access-Control-Allow-Origin", "*")
                self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
                self.send_header(
                    "Access-Control-Allow-Headers",
                    "Content-Type, Authorization, x-api-key, X-Admin-Token",
                )
                self.send_header("Access-Control-Max-Age", "86400")

            def do_OPTIONS(self) -> None:
                if handle_admin_request(self, config, deepseek_client, logger):
                    return
                self.send_response(HTTPStatus.NO_CONTENT)
                self._send_cors_headers()
                self.end_headers()

            def do_HEAD(self) -> None:
                self._admin_begin_request("HEAD")
                try:
                    if handle_admin_request(self, config, deepseek_client, logger):
                        return
                    path = self._path_without_query()
                    if path == "/health":
                        self.send_response(HTTPStatus.OK)
                        self.send_header("Content-Type", "application/json; charset=utf-8")
                        self.send_header("Content-Length", "15")
                        self._send_cors_headers()
                        self.end_headers()
                        return
                    if path == f"{config.api_prefix}/models":
                        if not self._authorize():
                            return
                        self.send_response(HTTPStatus.OK)
                        self.send_header("Content-Type", "application/json; charset=utf-8")
                        self.send_header("Content-Length", "0")
                        self._send_cors_headers()
                        self.end_headers()
                        return
                    self.send_response(HTTPStatus.METHOD_NOT_ALLOWED)
                    self.end_headers()
                except _CLIENT_DISCONNECTED:
                    logger.warning("客户端在 HEAD 响应写回前断开 path=%s", self.path)
                finally:
                    self._admin_finalize_request()

            def do_GET(self) -> None:
                self._admin_begin_request("GET")
                try:
                    self._debug_log_request_start()

                    if handle_admin_request(self, config, deepseek_client, logger):
                        return

                    path = self._path_without_query()

                    if path == "/health":
                        try:
                            cache_stats = (
                                deepseek_client.session_cache.stats()
                                if deepseek_client.cache_enabled
                                else {"enabled": False, "total": 0, "active": 0}
                            )
                        except Exception:
                            cache_stats = {"enabled": False, "total": 0, "active": 0}

                        self._write_json(HTTPStatus.OK, {
                            "status": "ok",
                            "config": {
                                "default_model": config.default_model,
                                "api_prefix": config.api_prefix,
                                "delete_conversation": config.delete_conversation,
                                "max_concurrency": config.max_concurrency,
                                "auth_required": bool(config.server_api_keys),
                                "search_enabled": config.search_enabled, 
                            },
                            "cache": cache_stats,
                        })
                        return

                    if path.startswith(f"{config.api_prefix}/"):
                        if not self._authorize():
                            logger.warning("认证失败 path=%s ip=%s", self.path, self.client_address[0])
                            self._write_json(
                                HTTPStatus.UNAUTHORIZED,
                                make_error(
                                    "Incorrect API key provided.",
                                    error_type=ERROR_AUTHENTICATION,
                                    code="invalid_api_key",
                                    request_id=gen_request_id(),
                                ),
                            )
                            return

                        if path == f"{config.api_prefix}/models":
                            data = [
                                {
                                    "id": model,
                                    "object": "model",
                                    "created": 1700000000,
                                    "owned_by": "deepseek",
                                }
                                for model in EXPOSED_MODELS
                            ]
                            self._write_json(HTTPStatus.OK, {"object": "list", "data": data})
                            return

                        if path.startswith(f"{config.api_prefix}/models/"):
                            model_id = path[len(f"{config.api_prefix}/models/"):]
                            if model_id in EXPOSED_MODELS:
                                self._write_json(HTTPStatus.OK, {
                                    "id": model_id,
                                    "object": "model",
                                    "created": 1700000000,
                                    "owned_by": "deepseek",
                                })
                                return
                            self._write_json(
                                HTTPStatus.NOT_FOUND,
                                make_error(
                                    f"The model '{model_id}' does not exist",
                                    error_type=ERROR_INVALID_REQUEST,
                                    param="model",
                                    code="model_not_found",
                                    request_id=gen_request_id(),
                                ),
                            )
                            return

                    logger.debug("GET 未匹配 path=%s", self.path)
                    self._write_json(
                        HTTPStatus.NOT_FOUND,
                        make_error(
                            "Unknown endpoint",
                            error_type=ERROR_NOT_FOUND,
                            code="not_found",
                            request_id=gen_request_id(),
                        ),
                    )
                except _CLIENT_DISCONNECTED:
                    logger.warning("客户端在 GET 响应写回前断开 path=%s", self.path)
                except Exception as exc:
                    logger.error("处理 GET 请求失败 path=%s error=%s\n%s", self.path, exc, traceback.format_exc())
                    self._safe_write_json(
                        HTTPStatus.INTERNAL_SERVER_ERROR,
                        make_error(
                            f"Internal server error: {exc}",
                            error_type=ERROR_SERVER,
                            code="internal_error",
                            request_id=gen_request_id(),
                        ),
                    )
                finally:
                    self._admin_finalize_request()

            def do_POST(self) -> None:
                self._admin_begin_request("POST")
                try:
                    self._debug_log_request_start()

                    if handle_admin_request(self, config, deepseek_client, logger):
                        return

                    path = self._path_without_query()

                    if path != f"{config.api_prefix}/chat/completions":
                        logger.debug("POST 未匹配 path=%s", self.path)
                        self._write_json(
                            HTTPStatus.NOT_FOUND,
                            make_error(
                                "Unknown endpoint",
                                error_type=ERROR_NOT_FOUND,
                                code="not_found",
                                request_id=gen_request_id(),
                            ),
                        )
                        return

                    if not self._authorize():
                        logger.warning("认证失败 path=%s ip=%s", self.path, self.client_address[0])
                        self._write_json(
                            HTTPStatus.UNAUTHORIZED,
                            make_error(
                                "Incorrect API key provided.",
                                error_type=ERROR_AUTHENTICATION,
                                code="invalid_api_key",
                                request_id=gen_request_id(),
                            ),
                        )
                        return

                    content_length = self._parse_content_length()
                    if content_length < 0:
                        self._write_json(
                            HTTPStatus.BAD_REQUEST,
                            make_error(
                                "Content-Length cannot be negative",
                                error_type=ERROR_INVALID_REQUEST,
                                param="content_length",
                                code="invalid_content_length",
                                request_id=gen_request_id(),
                            ),
                        )
                        return

                    raw_body = self.rfile.read(content_length) if content_length else b"{}"
                    debug_dump(logger, config.debug_dump_all, f"HTTP 入站原始请求体 path={self.path}", raw_body)

                    try:
                        payload = json.loads(raw_body.decode("utf-8"))
                    except UnicodeDecodeError:
                        self._write_json(
                            HTTPStatus.BAD_REQUEST,
                            make_error(
                                "Request body must be UTF-8 encoded",
                                error_type=ERROR_INVALID_REQUEST,
                                code="invalid_encoding",
                                request_id=gen_request_id(),
                            ),
                        )
                        return
                    except json.JSONDecodeError as exc:
                        self._write_json(
                            HTTPStatus.BAD_REQUEST,
                            make_error(
                                f"Invalid JSON: {exc.msg}",
                                error_type=ERROR_INVALID_REQUEST,
                                code="invalid_json",
                                request_id=gen_request_id(),
                            ),
                        )
                        return

                    if not isinstance(payload, dict):
                        self._write_json(
                            HTTPStatus.BAD_REQUEST,
                            make_error(
                                "Request body must be a JSON object",
                                error_type=ERROR_INVALID_REQUEST,
                                code="invalid_payload",
                                request_id=gen_request_id(),
                            ),
                        )
                        return

                    debug_dump(logger, config.debug_dump_all, f"HTTP 入站解析后 JSON path={self.path}", payload)

                    model = str(payload.get("model", config.default_model))

                    messages = payload.get("messages")
                    if not isinstance(messages, list) or not messages:
                        self._write_json(
                            HTTPStatus.BAD_REQUEST,
                            make_error(
                                "you must provide a model and messages parameter",
                                error_type=ERROR_INVALID_REQUEST,
                                param="messages",
                                request_id=gen_request_id(),
                            ),
                        )
                        return

                    is_stream = bool(payload.get("stream"))

                    if is_stream:
                        self._stream_completion(payload)
                    else:
                        logger.info("收到 chat 请求 model=%s", model)
                        result = deepseek_client.chat_completion(payload)
                        self._write_json(HTTPStatus.OK, result)

                except QueueTimeoutError as exc:
                    logger.warning("DeepSeek 队列等待超时 error=%s", exc)
                    self._write_json(
                        HTTPStatus.SERVICE_UNAVAILABLE,
                        make_error(
                            f"Service temporarily unavailable: {exc}",
                            error_type=ERROR_SERVER,
                            code="queue_timeout",
                            request_id=gen_request_id(),
                        ),
                    )
                except UpstreamAPIError as exc:
                    logger.warning("上游 DeepSeek 返回错误 status=%s error=%s", exc.status_code, exc)
                    status = HTTPStatus.BAD_GATEWAY
                    if 400 <= exc.status_code < 600:
                        try:
                            status = HTTPStatus(exc.status_code)
                        except ValueError:
                            pass
                    self._write_json(
                        status,
                        make_error(
                            str(exc),
                            error_type=ERROR_UPSTREAM,
                            code="upstream_error",
                            request_id=gen_request_id(),
                        ),
                    )
                except ValueError as exc:
                    logger.warning("请求参数错误 path=%s error=%s", self.path, exc)
                    self._write_json(
                        HTTPStatus.BAD_REQUEST,
                        make_error(
                            str(exc),
                            error_type=ERROR_INVALID_REQUEST,
                            code="invalid_request",
                            request_id=gen_request_id(),
                        ),
                    )
                except _CLIENT_DISCONNECTED as exc:
                    logger.warning("客户端连接提前断开 path=%s error=%s", self.path, exc)
                    self._admin_error = f"client_disconnected: {exc}"
                except Exception as exc:
                    logger.error("处理请求失败 error=%s\n%s", exc, traceback.format_exc())
                    self._admin_error = str(exc)
                    self._safe_write_json(
                        HTTPStatus.BAD_GATEWAY,
                        make_error(
                            f"Upstream error: {exc}",
                            error_type=ERROR_UPSTREAM,
                            code=exc.__class__.__name__.lower(),
                            request_id=gen_request_id(),
                        ),
                    )
                finally:
                    self._admin_finalize_request()

            def _stream_completion(self, payload: dict[str, object]) -> None:
                model = str(payload.get("model", config.default_model))
                logger.info("开始流式响应 model=%s", model)

                stream_iter = deepseek_client.stream_chat_completion(payload)

                self.send_response(HTTPStatus.OK)
                self.send_header("Content-Type", "text/event-stream; charset=utf-8")
                self.send_header("Cache-Control", "no-cache")
                self.send_header("Connection", "close")
                self.send_header("X-Accel-Buffering", "no")
                self._send_cors_headers()
                self.end_headers()

                try:
                    for chunk in stream_iter:
                        self.wfile.write(chunk.encode("utf-8"))
                        self.wfile.flush()
                except _CLIENT_DISCONNECTED as exc:
                    logger.warning("客户端在流式响应过程中断开 model=%s error=%s", model, exc)
                    return
                except Exception as exc:
                    logger.error("流式请求失败 model=%s error=%s\n%s", model, exc, traceback.format_exc())
                    return

                logger.info("流式请求完成 model=%s", model)

            def _authorize(self) -> bool:
                if not config.server_api_keys:
                    return True

                authorization = self.headers.get("Authorization", "")
                if authorization.startswith("Bearer "):
                    token = authorization[7:].strip()
                    if token in config.server_api_keys:
                        return True

                x_api_key = self.headers.get("x-api-key", "")
                if x_api_key and x_api_key.strip() in config.server_api_keys:
                    return True

                return False

            def _write_json(self, status: HTTPStatus, payload: dict[str, object]) -> None:
                body = json.dumps(payload, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
                debug_dump(logger, config.debug_dump_all, f"HTTP 出站 JSON 响应 status={int(status)} path={self.path}", body)
                self.send_response(status)
                self.send_header("Content-Type", "application/json; charset=utf-8")
                self.send_header("Content-Length", str(len(body)))
                self._send_cors_headers()
                self.end_headers()
                self.wfile.write(body)

            def _safe_write_json(self, status: HTTPStatus, payload: dict[str, object]) -> None:
                try:
                    self._write_json(status, payload)
                except _CLIENT_DISCONNECTED:
                    logger.warning("客户端在 JSON 响应写回前断开 path=%s", self.path)

            def _parse_content_length(self) -> int:
                raw_value = self.headers.get("Content-Length", "0").strip()
                try:
                    return int(raw_value or "0")
                except ValueError as exc:
                    raise ValueError(f"无效的 Content-Length: {raw_value}") from exc

            def _debug_log_request_start(self) -> None:
                debug_dump(
                    logger,
                    config.debug_dump_all,
                    f"HTTP 入站请求 {self.command} {self.path} headers",
                    {key: value for key, value in self.headers.items()},
                )

            def _path_without_query(self) -> str:
                return urlparse(self.path).path

            def _admin_begin_request(self, method: str) -> None:
                self._admin_start = time.monotonic()
                self._admin_method = method
                self._admin_status = 200
                self._admin_model = ""
                self._admin_stream = False
                self._admin_error = ""
                self._admin_request_id = gen_request_id()

                if not getattr(self, "_admin_send_response_wrapped", False):
                    _orig_send_response = self.send_response
                    def _tracked_send_response(code, message=None):
                        try:
                            self._admin_status = int(code)
                        except (TypeError, ValueError):
                            pass
                        return _orig_send_response(code, message)
                    self.send_response = _tracked_send_response
                    self._admin_send_response_wrapped = True

            def _admin_finalize_request(self) -> None:
                try:
                    path = self._path_without_query()
                    if path.startswith("/admin"):
                        return

                    duration_ms = int((time.monotonic() - self._admin_start) * 1000)

                    admin_record_request(
                        method=getattr(self, "_admin_method", ""),
                        path=path,
                        protocol=admin_classify_protocol(path),
                        model=getattr(self, "_admin_model", ""),
                        status=getattr(self, "_admin_status", 200),
                        duration_ms=duration_ms,
                        client_ip=self.client_address[0] if self.client_address else "",
                        stream=getattr(self, "_admin_stream", False),
                        error=getattr(self, "_admin_error", ""),
                        request_id=getattr(self, "_admin_request_id", ""),
                    )
                except Exception:
                    pass

            def log_message(self, format: str, *args) -> None:
                logger.info("%s - %s", self.address_string(), format % args)

        return RequestHandler