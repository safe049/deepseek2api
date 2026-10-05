"""
线程安全的动态 CookieJar。

- 初始化时从 .env 的 DEEPSEEK_COOKIE 载入基线 cookie
- 每次 HTTP 响应后解析 Set-Cookie 并合并更新
- 保证 HWWAFSESID / HWWAFSESTIME / ds_session_id 等轮换 cookie 始终最新
"""

from __future__ import annotations

import threading
import time
from http.cookies import SimpleCookie
from logging import Logger


class CookieJar:
    def __init__(self, seed_cookie: str, logger: Logger) -> None:
        self._lock = threading.RLock()
        self.logger = logger
        self._jar: dict[str, str] = {}
        # 记录每个 cookie 的过期时间（None 表示会话 cookie）
        self._expires: dict[str, float] = {}
        if seed_cookie:
            self.update_from_header(seed_cookie, source="env")

    # ── 解析单个 Cookie 字符串（用于初始化） ──
    def update_from_header(self, cookie_header: str, source: str = "response") -> None:
        if not cookie_header:
            return
        with self._lock:
            for part in cookie_header.split(";"):
                part = part.strip()
                if not part or "=" not in part:
                    continue
                name, _, value = part.partition("=")
                name = name.strip()
                value = value.strip()
                if not name:
                    continue
                self._jar[name] = value
            self.logger.debug(
                "CookieJar 更新 (%s): %s",
                source,
                ", ".join(sorted(self._jar.keys())),
            )

    # ── 解析响应中的 Set-Cookie 头 ──
    def update_from_set_cookie(self, set_cookie_values: list[str]) -> None:
        if not set_cookie_values:
            return
        with self._lock:
            changed = []
            for raw in set_cookie_values:
                try:
                    sc = SimpleCookie()
                    sc.load(raw)
                except Exception:
                    continue
                for name, morsel in sc.items():
                    value = morsel.value
                    self._jar[name] = value
                    max_age = morsel["max-age"]
                    expires = morsel["expires"]
                    if max_age:
                        try:
                            self._expires[name] = time.time() + int(max_age)
                        except (TypeError, ValueError):
                            pass
                    elif expires:
                        # 略过 expires 解析（HTTP 日期格式），够用即可
                        pass
                    changed.append(name)
            if changed:
                self.logger.debug(
                    "CookieJar Set-Cookie 合并: %s",
                    ", ".join(sorted(set(changed))),
                )

    def build_cookie_header(self) -> str:
        with self._lock:
            now = time.time()
            # 剔除已过期 cookie
            for name, exp in list(self._expires.items()):
                if exp < now:
                    self._jar.pop(name, None)
                    self._expires.pop(name, None)
            return "; ".join(f"{k}={v}" for k, v in self._jar.items())

    def snapshot(self) -> dict[str, str]:
        with self._lock:
            return dict(self._jar)