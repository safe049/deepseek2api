"""
DeepSeek 会话缓存。

基于 OpenAI messages 前缀 hash 做 key，映射到 DeepSeek chat_session_id。
cache key 包含 system 消息与 model，避免不同 system / model 之间串扰。
"""

from __future__ import annotations

import hashlib
import json
import logging
import threading
import time
from collections import OrderedDict
from dataclasses import dataclass, field

logger = logging.getLogger("deepseek2api.cache")


@dataclass(slots=True)
class SessionState:
    session_id: str
    turn_count: int = 0
    last_message_id: str | None = None
    created_at: float = field(default_factory=time.time)
    last_used: float = field(default_factory=time.time)


def prefix_hash(messages: list[dict], model: str = "") -> str:
    canonical = json.dumps(
        {"model": model, "messages": messages},
        ensure_ascii=False,
        sort_keys=True,
    )
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


class SessionCache:
    """Thread-safe LRU cache: prefix_hash → SessionState."""

    def __init__(self, max_size: int = 500, ttl_seconds: int = 7200) -> None:
        self._lock = threading.RLock()
        self._store: OrderedDict[str, SessionState] = OrderedDict()
        self._max_size = max(1, max_size)
        self._ttl = ttl_seconds

    def get(self, key: str) -> SessionState | None:
        with self._lock:
            state = self._store.get(key)
            if state is None:
                return None
            if time.time() - state.last_used > self._ttl:
                del self._store[key]
                logger.debug("Session expired: %s...", key[:16])
                return None
            self._store.move_to_end(key)
            state.last_used = time.time()
            return state

    def put(self, key: str, state: SessionState) -> None:
        with self._lock:
            self._store[key] = state
            self._store.move_to_end(key)
            self._evict()

    def delete(self, key: str) -> SessionState | None:
        with self._lock:
            return self._store.pop(key, None)

    def delete_by_session_id(self, session_id: str) -> int:
        """删除所有指向给定 session_id 的缓存条目，返回删除数量。"""
        if not session_id:
            return 0
        removed = 0
        with self._lock:
            for k in [k for k, v in self._store.items() if v.session_id == session_id]:
                del self._store[k]
                removed += 1
        return removed

    def _evict(self) -> None:
        while len(self._store) > self._max_size:
            evicted_key, _ = self._store.popitem(last=False)
            logger.debug("LRU evicted: %s...", evicted_key[:16])

    @property
    def size(self) -> int:
        with self._lock:
            return len(self._store)

    def active_session_ids(self) -> set[str]:
        with self._lock:
            now = time.time()
            return {
                s.session_id
                for s in self._store.values()
                if now - s.last_used <= self._ttl
            }

    def stats(self) -> dict:
        with self._lock:
            now = time.time()
            active = sum(
                1 for s in self._store.values() if now - s.last_used <= self._ttl
            )
            return {
                "total": len(self._store),
                "active": active,
                "max_size": self._max_size,
                "ttl_seconds": self._ttl,
            }