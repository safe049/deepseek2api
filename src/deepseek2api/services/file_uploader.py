# src/deepseek2api/services/file_uploader.py
"""
DeepSeek 文件上传客户端。

把 OpenAI 消息中的 image_url（支持 data URL 与 HTTP URL）上传到 DeepSeek，
返回 file_id 供 chat/completion 请求的 ref_file_ids 使用。

上游流程（参考浏览器抓包）：
  1. POST /api/v0/chat/create_pow_challenge  target_path=/api/v0/file/upload_file
  2. 求解 PoW → x-ds-pow-response
  3. POST /api/v0/file/upload_file (multipart/form-data) → biz_data.id
  4. 该 id 直接进 chat/completion 的 ref_file_ids，服务端异步解析
"""

from __future__ import annotations

import base64
import hashlib
import json
import mimetypes
import re
import urllib.error
import urllib.parse
import urllib.request
import uuid
from logging import Logger
from typing import Optional

from ..config import AppConfig
from ..logging_utils import debug_dump
from .deepseek_auth import DeepSeekAuthManager
from .pow_solver import PoWSolver


_UPLOAD_TARGET_PATH = "/api/v0/file/upload_file"
_MAX_URL_CACHE = 256

# data:image/png;base64,xxxx  或  data:image/png,urlencoded
_DATA_URL_RE = re.compile(r"^data:([^;,]*?)(;base64)?,(.*)$", re.DOTALL)


class FileUploader:
    """封装 DeepSeek /api/v0/file/upload_file。

    内部维护 image_url hash → file_id 的 LRU 缓存，
    相同图片在一次进程生命周期内不会重复上传。
    """

    def __init__(
        self,
        config: AppConfig,
        auth: DeepSeekAuthManager,
        pow_solver: PoWSolver,
        logger: Logger,
    ) -> None:
        self.config = config
        self.auth = auth
        self.pow_solver = pow_solver
        self.logger = logger
        self._cache: dict[str, str] = {}   # url-hash → file_id

    # ── 公共入口 ─────────────────────────────────────────

    def upload_from_openai_image(
        self,
        image_url: str,
        *,
        name_hint: str = "",
    ) -> Optional[str]:
        """上传 OpenAI image_url，返回 file_id；失败返回 None。"""
        if not image_url:
            return None

        cache_key = hashlib.sha256(
            image_url.encode("utf-8", "ignore")
        ).hexdigest()
        cached = self._cache.get(cache_key)
        if cached:
            self.logger.info("FileUpload | cache HIT | file_id=%s", cached)
            return cached

        try:
            content, mime, filename = self._decode_image_url(image_url, name_hint)
        except Exception as exc:
            self.logger.warning("FileUpload | 解析 image_url 失败: %s", exc)
            return None

        file_id = self.upload_bytes(content, mime=mime, filename=filename)
        if file_id:
            self._cache_put(cache_key, file_id)
        return file_id

    def upload_bytes(
        self,
        content: bytes,
        *,
        mime: str = "application/octet-stream",
        filename: str = "",
    ) -> Optional[str]:
        if not content:
            return None
        if not filename:
            ext = mimetypes.guess_extension(mime) or ".bin"
            filename = f"upload_{uuid.uuid4().hex[:8]}{ext}"

        try:
            pow_header = self._solve_upload_pow()
        except Exception as exc:
            self.logger.warning("FileUpload | PoW 求解失败: %s", exc)
            return None

        boundary = f"----geckoformboundary{uuid.uuid4().hex}"
        body = self._build_multipart(boundary, filename, mime, content)

        headers = self.auth.build_full_headers(
            pow_response=pow_header,
            content_type=f"multipart/form-data; boundary={boundary}",
        )

        url = f"{self.config.base_url}{_UPLOAD_TARGET_PATH}"
        debug_dump(
            self.logger,
            self.config.debug_dump_all,
            "FileUpload | 请求",
            {
                "url": url,
                "filename": filename,
                "mime": mime,
                "size": len(content),
            },
        )

        try:
            request = urllib.request.Request(
                url, data=body, method="POST", headers=headers,
            )
            with urllib.request.urlopen(
                request, timeout=self.config.request_timeout,
            ) as resp:
                self.auth.absorb_response_cookies(resp)
                payload = self.auth._read_json_response(resp)
        except urllib.error.HTTPError as exc:
            self.logger.warning(
                "FileUpload | HTTP %s: %s", exc.code, self._safe_read(exc),
            )
            return None
        except urllib.error.URLError as exc:
            self.logger.warning("FileUpload | 网络错误: %s", exc)
            return None
        except Exception as exc:
            self.logger.warning("FileUpload | 异常: %s", exc)
            return None

        file_id = self._extract_file_id(payload)
        if not file_id:
            self.logger.warning(
                "FileUpload | 响应中无 file_id: %s",
                json.dumps(payload, ensure_ascii=False)[:300],
            )
            return None

        self.logger.info(
            "FileUpload | ok | id=%s | name=%s | size=%d",
            file_id, filename, len(content),
        )
        return file_id

    # ── 内部 ────────────────────────────────────────────

    def _solve_upload_pow(self) -> str:
        body = json.dumps({"target_path": _UPLOAD_TARGET_PATH}).encode("utf-8")
        headers = {
            **self.auth.get_auth_headers(),
            "Content-Type": "application/json",
        }
        request = urllib.request.Request(
            self.config.pow_challenge_url,
            data=body, method="POST", headers=headers,
        )
        with urllib.request.urlopen(request, timeout=15) as resp:
            self.auth.absorb_response_cookies(resp)
            payload = self.auth._read_json_response(resp)

        biz = (
            payload.get("data", {}).get("biz_data", {}).get("challenge", {})
            or payload.get("data", {})
        )
        if not biz.get("challenge"):
            raise RuntimeError(f"PoW challenge 响应格式异常: {payload}")

        return self.pow_solver.solve(
            challenge=biz.get("challenge", ""),
            salt=biz.get("salt", ""),
            difficulty=int(biz.get("difficulty", 144000)),
            algorithm=biz.get("algorithm", "DeepSeekHashV1"),
            expire_at=int(biz.get("expire_at", 0)),
            signature=biz.get("signature", ""),
            target_path=biz.get("target_path", _UPLOAD_TARGET_PATH),
        )

    def _decode_image_url(
        self, image_url: str, name_hint: str,
    ) -> tuple[bytes, str, str]:
        image_url = image_url.strip()

        # data URL
        m = _DATA_URL_RE.match(image_url)
        if m:
            mime = (m.group(1) or "image/png").strip() or "image/png"
            is_b64 = bool(m.group(2))
            data = m.group(3)
            if is_b64:
                data = re.sub(r"\s+", "", data)
                content = base64.b64decode(data)
            else:
                content = urllib.parse.unquote_to_bytes(data)
            ext = mimetypes.guess_extension(mime) or ".png"
            name = name_hint or f"image_{uuid.uuid4().hex[:8]}{ext}"
            return content, mime, name

        # HTTP / HTTPS
        if image_url.startswith(("http://", "https://")):
            request = urllib.request.Request(
                image_url, headers={"User-Agent": "Mozilla/5.0"},
            )
            with urllib.request.urlopen(request, timeout=30) as resp:
                content = resp.read()
                mime = (
                    (resp.headers.get("Content-Type") or "image/png")
                    .split(";")[0].strip()
                )
            ext = mimetypes.guess_extension(mime) or ".png"
            name = name_hint or f"image_{uuid.uuid4().hex[:8]}{ext}"
            return content, mime, name

        raise ValueError(f"不支持的 image_url: {image_url[:80]}")

    @staticmethod
    def _build_multipart(
        boundary: str, filename: str, mime: str, content: bytes,
    ) -> bytes:
        crlf = b"\r\n"
        return crlf.join([
            f"--{boundary}".encode("utf-8"),
            (
                f'Content-Disposition: form-data; name="file"; '
                f'filename="{filename}"'
            ).encode("utf-8"),
            f"Content-Type: {mime}".encode("utf-8"),
            b"",
            content,
            f"--{boundary}--".encode("utf-8"),
            b"",
        ])

    @staticmethod
    def _extract_file_id(payload: dict) -> str:
        if not isinstance(payload, dict):
            return ""
        data = payload.get("data") or {}
        biz_data = data.get("biz_data") or {}
        return str(biz_data.get("id") or "")

    @staticmethod
    def _safe_read(exc: urllib.error.HTTPError) -> str:
        try:
            return exc.read().decode("utf-8", errors="replace")[:200]
        except Exception:
            return str(exc)

    def _cache_put(self, key: str, value: str) -> None:
        if len(self._cache) >= _MAX_URL_CACHE:
            try:
                oldest = next(iter(self._cache))
                del self._cache[oldest]
            except StopIteration:
                pass
        self._cache[key] = value