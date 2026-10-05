from __future__ import annotations

import base64
import gzip
import json
import threading
import time
import urllib.error
import urllib.request
import uuid
from dataclasses import dataclass
from logging import Logger

from ..config import AppConfig
from ..logging_utils import debug_dump
from .cookie_jar import CookieJar


@dataclass(slots=True)
class HIFTokens:
    leim: str
    dliq: str
    fetched_at: float
    ttl_seconds: int = 600

    @property
    def is_expired(self) -> bool:
        return time.time() > self.fetched_at + self.ttl_seconds - 30


class DeepSeekAuthManager:
    def __init__(self, config: AppConfig, logger: Logger) -> None:
        self.config = config
        self.logger = logger
        self._lock = threading.RLock()
        self._hif_tokens: HIFTokens | None = None
        # 动态 cookie jar（承载 HWWAFSESID / ds_session_id 等的自动更新）
        self.cookie_jar = CookieJar(seed_cookie=config.cookie, logger=logger)

    def get_browser_headers(self) -> dict[str, str]:
        return {
            "Accept": "*/*",
            "Accept-Language": "zh-CN,zh;q=0.9,en;q=0.8,en-GB;q=0.7,en-US;q=0.6",
            "User-Agent": (
                "Mozilla/5.0 (X11; Linux x86_64; rv:156.0) "
                "Gecko/20100101 Firefox/156.0"
            ),
            "Accept-Encoding": "gzip, deflate, br, zstd",
            "Origin": "https://chat.deepseek.com",
            "Referer": "https://chat.deepseek.com/",
            "Sec-Fetch-Dest": "empty",
            "Sec-Fetch-Mode": "cors",
            "Sec-Fetch-Site": "same-origin",
        }

    def get_auth_headers(self) -> dict[str, str]:
        headers = {
            **self.get_browser_headers(),
            "Authorization": f"Bearer {self.config.auth_token}",
            "x-device-id": self.config.device_id,
            "x-device-model": "",
            "x-client-bundle-id": "com.deepseek.chat",
            "x-client-platform": "web",
            "x-client-version": "2.5.0",
            "x-client-locale": "zh_CN",
            "x-client-timezone-offset": "28800",
        }
        if self.config.settings_token:
            headers["x-settings-token"] = self.config.settings_token
        # 关键：动态构造 Cookie 头
        cookie_header = self.cookie_jar.build_cookie_header()
        if cookie_header:
            headers["Cookie"] = cookie_header
        return headers

    # ── 每次请求后调用，将 Set-Cookie 合并回 jar ──
    def absorb_response_cookies(self, response) -> None:
        try:
            set_cookies = response.headers.get_all("Set-Cookie") or []
            if set_cookies:
                self.cookie_jar.update_from_set_cookie(set_cookies)
        except Exception as exc:
            self.logger.debug("吸收 Set-Cookie 失败: %s", exc)

    # ── HIF 获取：正确解析 value 并吸收 Set-Cookie ──
    def _fetch_hif_tokens(self) -> None:
        self.logger.info("正在获取 HIF tokens...")

        leim_value = ""
        dliq_value = ""

        # x-hif-leim
        try:
            req = urllib.request.Request(
                self.config.hif_leim_url,
                method="GET",
                headers=self.get_auth_headers(),   # 带上完整认证头（含动态 Cookie）
            )
            with urllib.request.urlopen(req, timeout=10) as response:
                self.absorb_response_cookies(response)
                raw = response.read().decode("utf-8", errors="replace")
                payload = json.loads(raw)
                leim_value = payload.get("data", {}).get("biz_data", {}).get("value", "")
                if not leim_value:
                    raise RuntimeError(f"响应中未找到 value 字段: {raw[:200]}")
                self.logger.debug("获取 x-hif-leim 成功: %s", leim_value[:50])
        except Exception as exc:
            self.logger.warning("获取 x-hif-leim 失败: %s", exc)
            leim_value = ""

        # x-hif-dliq
        try:
            req = urllib.request.Request(
                self.config.hif_dliq_url,
                method="GET",
                headers=self.get_auth_headers(),
            )
            with urllib.request.urlopen(req, timeout=10) as response:
                self.absorb_response_cookies(response)
                raw = response.read().decode("utf-8", errors="replace")
                payload = json.loads(raw)
                dliq_value = payload.get("data", {}).get("biz_data", {}).get("value", "")
                if not dliq_value:
                    raise RuntimeError(f"响应中未找到 value 字段: {raw[:200]}")
                self.logger.debug("获取 x-hif-dliq 成功: %s", dliq_value[:50])
        except Exception as exc:
            self.logger.warning("获取 x-hif-dliq 失败: %s", exc)
            dliq_value = ""

        self._hif_tokens = HIFTokens(
            leim=leim_value,
            dliq=dliq_value,
            fetched_at=time.time(),
        )
        self.logger.info("HIF tokens 获取完成")

    def get_hif_tokens(self) -> HIFTokens:
        with self._lock:
            if self._hif_tokens and not self._hif_tokens.is_expired:
                return self._hif_tokens
            self._fetch_hif_tokens()
            return self._hif_tokens

    def build_full_headers(
        self,
        pow_response: str,              # 已经是 Base64 编码的完整 header 值
        content_type: str = "application/json",
    ) -> dict[str, str]:
        """构建包含所有必要信息的完整请求头。"""
        hif = self.get_hif_tokens()

        headers = {
            **self.get_auth_headers(),
            "Content-Type": content_type,
            "x-hif-leim": hif.leim,
            "x-hif-dliq": hif.dliq,
            "x-ds-pow-response": pow_response,   # 直接使用，不再处理
        }
        return headers

    def _read_json_response(self, response) -> dict[str, object]:
        try:
            raw_body = response.read()
            content_encoding = response.headers.get("Content-Encoding", "").lower()

            if content_encoding == "gzip":
                raw_body = gzip.decompress(raw_body)
            elif content_encoding == "deflate":
                import zlib
                try:
                    raw_body = zlib.decompress(raw_body)
                except zlib.error:
                    raw_body = zlib.decompress(raw_body, -zlib.MAX_WBITS)
            elif content_encoding == "br":
                try:
                    import brotli
                    raw_body = brotli.decompress(raw_body)
                except ImportError:
                    self.logger.warning("收到 Brotli 压缩响应但未安装 brotli 库")

            debug_dump(self.logger, self.config.debug_dump_all, "DeepSeek 原始 JSON 响应体", raw_body)

            try:
                text = raw_body.decode("utf-8")
            except UnicodeDecodeError:
                text = raw_body.decode("utf-8", errors="replace")
                self.logger.warning("DeepSeek 响应包含非 UTF-8 字节，已使用 replace 模式解码")

            payload = json.loads(text)

        except gzip.BadGzipFile as exc:
            raise RuntimeError("DeepSeek 响应 gzip 解压失败") from exc
        except json.JSONDecodeError as exc:
            raise RuntimeError(f"DeepSeek 响应不是合法 JSON: {exc}") from exc

        if not isinstance(payload, dict):
            raise RuntimeError(f"DeepSeek 响应格式异常，期望 JSON 对象，实际是: {type(payload).__name__}")
        return payload