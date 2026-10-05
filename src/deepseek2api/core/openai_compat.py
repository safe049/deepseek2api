from __future__ import annotations

import hashlib
import os
import platform
import time
import uuid
from typing import Any

_FINGERPRINT_SALT = f"{platform.python_version()}-{platform.system()}-{os.getpid()}"
_FINGERPRINT_HASH = hashlib.md5(_FINGERPRINT_SALT.encode("utf-8")).hexdigest()[:6]


def gen_chatcmpl_id() -> str:
    return f"chatcmpl-{uuid.uuid4().hex[:24]}"


def gen_response_id() -> str:
    return f"resp_{uuid.uuid4().hex[:24]}"


def gen_message_id() -> str:
    return f"msg_{uuid.uuid4().hex[:24]}"


def gen_request_id() -> str:
    return f"req_{uuid.uuid4().hex[:24]}"


def system_fingerprint(model: str = "") -> str:
    if model:
        date_str = time.strftime("%Y%m%d")
        base = hashlib.md5(f"{model}:{date_str}".encode("utf-8")).hexdigest()[:8]
        return f"fp_{base}"
    return f"fp_{_FINGERPRINT_HASH}"


def make_error(
    message: str,
    *,
    error_type: str = "invalid_request_error",
    param: str | None = None,
    code: str | None = None,
    request_id: str | None = None,
) -> dict[str, Any]:
    err: dict[str, Any] = {
        "message": message,
        "type": error_type,
        "param": param,
        "code": code,
    }
    if request_id:
        err["request_id"] = request_id
    return {"error": err}


ERROR_INVALID_REQUEST = "invalid_request_error"
ERROR_AUTHENTICATION = "authentication_error"
ERROR_NOT_FOUND = "not_found_error"
ERROR_SERVER = "server_error"
ERROR_UPSTREAM = "upstream_error"

CODE_MODEL_NOT_FOUND = "model_not_found"
CODE_INVALID_API_KEY = "invalid_api_key"
CODE_INTERNAL_ERROR = "internal_error"


def now_timestamp() -> int:
    return int(time.time())