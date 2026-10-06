"""
DeepSeek Web API 客户端。

修复要点：
- 上游错误 → 正确的 HTTP 状态码
- usage 使用 DeepSeek 返回的 accumulated_token_usage
- cache key 包含 system + model + tool_calls 完整结构
- 每个请求独立 StreamState
- reasoning_content 输出
- context_length_exceeded → 清理缓存+会话，使用完整历史重开新对话
- rate_limit_reached → 冷却 15 秒后自动重试（最多 3 次）
- 工具调用：哨兵协议 + 流式检测器 + OpenAI 格式输出
"""

from __future__ import annotations

import codecs
import gzip
import http.client
import itertools
import json
import socket
import threading
import time
import uuid
import urllib.error
import urllib.parse
import urllib.request
import random
from dataclasses import dataclass
from logging import Logger
from typing import Callable

from ..config import AppConfig
from ..logging_utils import debug_dump
from .deepseek_auth import DeepSeekAuthManager
from .pow_solver import PoWSolver
from .session_cache import SessionCache, SessionState, prefix_hash
from .tool_calling import (
    ParsedToolCall,
    StreamingToolDetector,
    parse_tool_calls,
)
from .file_uploader import FileUploader
from .translator import (
    StreamState,
    convert_openai_to_deepseek,
    convert_openai_to_deepseek_incremental,
    convert_deepseek_to_openai_chunk,
    extract_images_from_messages,
)
from .citation_utils import CitationStripper, strip_citation_marks

# ── 重试策略 ────────────────────────────────────────────────────

RATE_LIMIT_COOLDOWN_SECONDS = 60
MAX_RATE_LIMIT_RETRIES = 3
MAX_CONTEXT_LENGTH_RETRIES = 2


class UpstreamAPIError(RuntimeError):
    def __init__(self, status_code: int, message: str, payload: dict[str, object] | None = None) -> None:
        super().__init__(message)
        self.status_code = status_code
        self.payload = payload or {}


class QueueTimeoutError(RuntimeError):
    pass


class ContextLengthExceededError(RuntimeError):
    """DeepSeek 返回 context_length_exceeded（对话长度上限）。"""


class RateLimitError(RuntimeError):
    """DeepSeek 返回 rate_limit_reached（消息发送过于频繁）。"""


@dataclass(slots=True)
class QueueLease:
    ticket: int
    release_callback: Callable[[int], None]
    released: bool = False

    def release(self) -> None:
        if self.released:
            return
        self.released = True
        self.release_callback(self.ticket)


class ConcurrentRequestQueue:
    def __init__(self, logger: Logger, wait_timeout: int, max_concurrency: int) -> None:
        self.logger = logger
        self.wait_timeout = wait_timeout
        self.max_concurrency = max(1, max_concurrency)
        self._condition = threading.Condition()
        self._next_ticket = 0
        self._serving_ticket = 0
        self._released_tickets: set[int] = set()

    def acquire(self, request_name: str) -> QueueLease:
        with self._condition:
            ticket = self._next_ticket
            self._next_ticket += 1
            queue_ahead = max(0, ticket - (self._serving_ticket + self.max_concurrency) + 1)
            start = time.monotonic()

            if queue_ahead > 0:
                self.logger.info("请求进入队列 ticket=%s ahead=%s request=%s", ticket, queue_ahead, request_name)

            while ticket >= self._serving_ticket + self.max_concurrency:
                remaining = self.wait_timeout - (time.monotonic() - start)
                if remaining <= 0:
                    raise QueueTimeoutError(
                        f"队列等待超时，前方仍有 {ticket - (self._serving_ticket + self.max_concurrency) + 1} 个请求。"
                    )
                self._condition.wait(timeout=remaining)

            active_slots = ticket - self._serving_ticket + 1
            self.logger.info(
                "请求获得执行槽位 ticket=%s active=%s/%s request=%s",
                ticket, active_slots, self.max_concurrency, request_name,
            )
            return QueueLease(ticket=ticket, release_callback=self._release)

    def _release(self, ticket: int) -> None:
        with self._condition:
            self._released_tickets.add(ticket)
            while self._serving_ticket in self._released_tickets:
                self._released_tickets.remove(self._serving_ticket)
                self._serving_ticket += 1
            self.logger.info("请求离开执行槽位 ticket=%s", ticket)
            self._condition.notify_all()


# ═══════════════════════════════════════════════════════════
# Conversation helpers
# ═══════════════════════════════════════════════════════════

def _split_messages(messages: list[dict]) -> tuple[list[dict], dict]:
    """
    返回 (prefix_messages, last_message)。

    - prefix_messages：除最后一条外的所有消息（保留 system / tool_calls / tool_call_id
      等完整结构，用于 cache key 计算）
    - last_message：最后一条消息的完整 dict

    采用「除最后一条外全部为前缀」的语义，可保证：
      Turn 1 请求 messages=[user1] → prefix=[]，完成后存 hash([user1, assistant1])
      Turn 2 请求 messages=[user1, assistant1, user2] → prefix=[user1, assistant1]，命中
    """
    raw: list[dict] = []
    for m in messages:
        if not isinstance(m, dict):
            continue
        entry: dict = {"role": m.get("role", "user"), "content": ""}

        content = m.get("content")
        if isinstance(content, list):
            parts = []
            for p in content:
                if isinstance(p, dict) and p.get("type") == "text":
                    parts.append(str(p.get("text", "")))
                elif isinstance(p, str):
                    parts.append(p)
            entry["content"] = "\n".join(parts)
        elif content is not None:
            entry["content"] = str(content)

        if m.get("role") == "assistant" and isinstance(m.get("tool_calls"), list):
            entry["tool_calls"] = m["tool_calls"]
        if m.get("role") == "tool" and m.get("tool_call_id"):
            entry["tool_call_id"] = m["tool_call_id"]

        raw.append(entry)

    if not raw:
        return [], {"role": "user", "content": ""}
    if len(raw) == 1:
        return [], raw[0]
    return raw[:-1], raw[-1]


def _payload_has_tools(payload: dict) -> bool:
    tools = payload.get("tools")
    if not isinstance(tools, list) or not tools:
        return False
    return payload.get("tool_choice", "auto") != "none"


def _sse(payload: dict) -> str:
    return f"data: {json.dumps(payload, ensure_ascii=False, separators=(',', ':'))}\n\n"


def _make_content_chunk(text: str, model: str, role: str | None = None) -> str:
    delta: dict = {"content": text}
    if role:
        delta["role"] = role
    chunk = {
        "id": f"chatcmpl-{uuid.uuid4().hex[:24]}",
        "object": "chat.completion.chunk",
        "created": int(time.time()),
        "model": model,
        "system_fingerprint": "",
        "choices": [{
            "index": 0,
            "delta": delta,
            "finish_reason": None,
            "logprobs": None,
        }],
    }
    return _sse(chunk)


def _make_tool_call_chunks(calls: list[ParsedToolCall], model: str) -> list[str]:
    """按 OpenAI 流式规范：id+name 一个 chunk，arguments 一个 chunk。"""
    chunks: list[str] = []
    for i, call in enumerate(calls):
        cid = f"chatcmpl-{uuid.uuid4().hex[:24]}"
        created = int(time.time())

        head = {
            "id": cid, "object": "chat.completion.chunk",
            "created": created, "model": model, "system_fingerprint": "",
            "choices": [{
                "index": 0,
                "delta": {
                    "role": "assistant",
                    "tool_calls": [{
                        "index": i, "id": call.id, "type": "function",
                        "function": {"name": call.name, "arguments": ""},
                    }],
                },
                "finish_reason": None, "logprobs": None,
            }],
        }
        chunks.append(_sse(head))

        args = {
            "id": cid, "object": "chat.completion.chunk",
            "created": created, "model": model, "system_fingerprint": "",
            "choices": [{
                "index": 0,
                "delta": {
                    "tool_calls": [{
                        "index": i,
                        "function": {"arguments": call.arguments},
                    }],
                },
                "finish_reason": None, "logprobs": None,
            }],
        }
        chunks.append(_sse(args))
    return chunks


def _make_finish_chunk(finish_reason: str, model: str) -> str:
    chunk = {
        "id": f"chatcmpl-{uuid.uuid4().hex[:24]}",
        "object": "chat.completion.chunk",
        "created": int(time.time()),
        "model": model,
        "system_fingerprint": "",
        "choices": [{
            "index": 0,
            "delta": {},
            "finish_reason": finish_reason,
            "logprobs": None,
        }],
    }
    return _sse(chunk)


class DeepSeekWebClient:
    """DeepSeek Web API 客户端。"""

    def __init__(self, config: AppConfig, logger: Logger) -> None:
        self.config = config
        self.logger = logger
        self.auth = DeepSeekAuthManager(config=config, logger=logger)
        self.pow_solver = PoWSolver(wasm_path=config.wasm_path, logger=logger)
        self.request_queue = ConcurrentRequestQueue(
            logger=logger,
            wait_timeout=config.queue_wait_timeout,
            max_concurrency=config.max_concurrency,
        )
        self.session_cache = SessionCache(
            max_size=config.session_cache_max_size,
            ttl_seconds=config.session_cache_ttl,
        )
        self.cache_enabled = config.session_cache_enabled

        self._proxy_sessions: set[str] = set()
        self._proxy_sessions_lock = threading.Lock()

        self.file_uploader = FileUploader(
            config=config,
            auth=self.auth,
            pow_solver=self.pow_solver,
            logger=logger,
        )
        self.file_upload_enabled = getattr(config, "file_upload_enabled", True)

        if self.cache_enabled:
            self.logger.info(
                "会话缓存已启用 max_size=%d ttl=%ds",
                config.session_cache_max_size,
                config.session_cache_ttl,
            )
        else:
            self.logger.info("会话缓存已禁用（delete_conversation=true，每次请求独立）")

        if config.anti_rate_limit_enabled:
            self.logger.info(
                "AntiRateLimit | enabled | delay %.2f~%.2fs",
                config.anti_rate_limit_min_delay,
                config.anti_rate_limit_max_delay,
            )            

    # ═══════════════════════════════════════════════════════
    # Conversation resolution
    # ═══════════════════════════════════════════════════════

    def _resolve_conversation(self, payload: dict[str, object]) -> dict:
        messages = payload.get("messages", [])
        if not isinstance(messages, list):
            messages = []

        # ★ 先上传图片（缓存命中路径和 fresh 路径都需要它）
        ref_file_ids = self._upload_images_from_messages(messages)

        full_prefix, last_message = _split_messages(messages)
        last_content = (
            last_message.get("content", "")
            if isinstance(last_message, dict) else ""
        )
        model = str(payload.get("model", self.config.default_model))
        p_hash = prefix_hash(full_prefix, model=model)

        cached = self.session_cache.get(p_hash) if self.cache_enabled else None

        if cached:
            ds_payload = convert_openai_to_deepseek_incremental(payload, self.config)
            ds_payload["chat_session_id"] = cached.session_id
            if ref_file_ids:                        # ← 注入
                ds_payload["ref_file_ids"] = ref_file_ids

            self.logger.info(
                "Cache HIT  | prefix=%s... | session=%s... | turn=%d | "
                "prompt_len=%d | files=%d",
                p_hash[:12], cached.session_id[:12],
                cached.turn_count + 1, len(ds_payload.get("prompt", "")),
                len(ref_file_ids),
            )
            return {
                "mode": "cached",
                "ds_payload": ds_payload,
                "session_id": cached.session_id,
                "cache_key": p_hash,
                "full_prefix": full_prefix,
                "last_message": last_message,
                "last_content": last_content,
                "model": model,
                "turn_count": cached.turn_count,
            }

        ds_payload = convert_openai_to_deepseek(payload, self.config)
        if ref_file_ids:                            # ← 注入
            ds_payload["ref_file_ids"] = ref_file_ids

        self.logger.info(
            "Cache %s | prefix=%s... | prompt_len=%d (full history packed) | files=%d",
            "MISS" if self.cache_enabled else "OFF",
            p_hash[:12],
            len(ds_payload.get("prompt", "")),
            len(ref_file_ids),
        )
        return {
            "mode": "fresh",
            "ds_payload": ds_payload,
            "session_id": "",
            "cache_key": p_hash,
            "full_prefix": full_prefix,
            "last_message": last_message,
            "last_content": last_content,
            "model": model,
            "turn_count": 0,
        }

    def _store_after_response(
        self,
        resolved: dict,
        session_id: str,
        assistant_content: str,
        request_message_id: str | None = None,
        tool_calls: list[ParsedToolCall] | None = None,
    ) -> None:
        if not self.cache_enabled:
            if session_id:
                with self._proxy_sessions_lock:
                    self._proxy_sessions.add(session_id)
            return

        # 下一轮的 cache key = 本轮完整前缀 + 本轮"当前输入消息" + assistant 输出
        new_prefix = list(resolved["full_prefix"])
        last_msg = resolved.get("last_message")
        if isinstance(last_msg, dict):
            new_prefix.append(dict(last_msg))
        else:
            new_prefix.append({"role": "user", "content": resolved.get("last_content", "")})

        assistant_entry: dict = {"role": "assistant", "content": assistant_content or ""}
        if tool_calls:
            assistant_entry["tool_calls"] = [
                {
                    "id": c.id,
                    "type": "function",
                    "function": {"name": c.name, "arguments": c.arguments},
                }
                for c in tool_calls
            ]
        new_prefix.append(assistant_entry)

        new_hash = prefix_hash(new_prefix, model=resolved["model"])

        next_turn = (resolved.get("turn_count") or 0) + 1
        self.session_cache.put(
            new_hash,
            SessionState(
                session_id=session_id,
                turn_count=next_turn,
                last_message_id=request_message_id,
            ),
        )

        with self._proxy_sessions_lock:
            self._proxy_sessions.add(session_id)

        self.logger.info(
            "Cached     | %s → new_prefix=%s... | session=%s... | turn=%d | tools=%d | msg_id=%s",
            resolved["mode"], new_hash[:12], session_id[:12], next_turn,
            len(tool_calls) if tool_calls else 0,
            (request_message_id or "N/A")[:12] if request_message_id else "N/A",
        )

    # ═══════════════════════════════════════════════════════
    # Cache / session cleanup
    # ═══════════════════════════════════════════════════════

    def _invalidate_cache_for_resolved(self, resolved: dict) -> None:
        if not resolved:
            return
        key = resolved.get("cache_key")
        if key:
            self.session_cache.delete(key)

    def _invalidate_and_delete_session(self, resolved: dict) -> None:
        """context_length_exceeded 时：从缓存移除 + 删除上游会话。"""
        session_id = resolved.get("session_id") or ""

        # 缓存中所有指向该 session 的条目都清掉
        if session_id and self.cache_enabled:
            removed = self.session_cache.delete_by_session_id(session_id)
            if removed:
                self.logger.info(
                    "Cache purge | session=%s... | removed=%d entries",
                    session_id[:12], removed,
                )
        self._invalidate_cache_for_resolved(resolved)

        if session_id:
            self._delete_session(session_id)

        # 清空 resolved 里的 session，避免后续误用
        resolved["session_id"] = ""

    def cleanup_stale_sessions(self) -> None:
        active = self.session_cache.active_session_ids()
        with self._proxy_sessions_lock:
            stale = self._proxy_sessions - active
            if not stale:
                return
            stale_list = list(stale)
            self._proxy_sessions.difference_update(stale_list)

        for sid in stale_list:
            self._delete_session(sid)

        if stale_list:
            self.logger.info(
                "Cleanup    | %d stale session(s) deleted | %d tracked | %d active",
                len(stale_list), len(self._proxy_sessions), len(active),
            )

    def cleanup_all_sessions(self) -> None:
        if self.cache_enabled:
            with self.session_cache._lock:
                self.session_cache._store.clear()

        with self._proxy_sessions_lock:
            ids = list(self._proxy_sessions)
            self._proxy_sessions.clear()

        if not ids:
            self.logger.info("Shutdown   | 无残留会话需要清理")
            return

        self.logger.info("Shutdown   | deleting %d proxy session(s)...", len(ids))
        deleted = 0
        for sid in ids:
            try:
                self._delete_session(sid)
                deleted += 1
            except Exception:
                pass
        self.logger.info("Shutdown   | %d session(s) cleaned up", deleted)

    # ═══════════════════════════════════════════════════════
    # Public API - non-stream
    # ═══════════════════════════════════════════════════════

    def chat_completion(self, payload: dict[str, object]) -> dict[str, object]:
        resolved = self._resolve_conversation(payload)
        lease = self.request_queue.acquire(f"chat:{payload.get('model', 'unknown')}")

        try:
            return self._chat_completion_impl(payload, resolved)
        finally:
            lease.release()

    def _chat_completion_impl(self, payload: dict, resolved: dict) -> dict:
        context_retries = 0
        rate_limit_retries = 0
        has_tools = _payload_has_tools(payload)

        while True:
            response = None
            session_id = resolved.get("session_id") or ""
            try:
                response, session_id = self._open_chat_stream(resolved)

                full_text = ""
                full_reasoning = ""
                last_response_message_id: str | None = None
                accumulated_tokens = 0
                state = StreamState()

                for event in self._iter_sse_events(response):
                    if not event:
                        continue

                    # 直接捕获 hint error
                    if isinstance(event, dict) and event.get("_event_type") == "hint":
                        if event.get("type") == "error":
                            finish_reason = event.get("finish_reason")
                            if finish_reason == "context_length_exceeded":
                                raise ContextLengthExceededError("DeepSeek: 达到对话长度上限")
                            if finish_reason == "rate_limit_reached":
                                raise RateLimitError(
                                    event.get("content") or "DeepSeek: 消息发送过于频繁"
                                )
                        continue

                    if isinstance(event, dict) and event.get("_event_type") == "ready":
                        resp_id = event.get("response_message_id")
                        if resp_id is not None:
                            last_response_message_id = str(resp_id)

                    if isinstance(event, dict) and event.get("o") == "BATCH":
                        batch_v = event.get("v")
                        if isinstance(batch_v, list):
                            for item in batch_v:
                                if isinstance(item, dict) and item.get("p") == "accumulated_token_usage":
                                    try:
                                        accumulated_tokens = int(item.get("v", 0))
                                    except (TypeError, ValueError):
                                        pass

                    openai_chunk = convert_deepseek_to_openai_chunk(
                        event, resolved["model"], state
                    )
                    if openai_chunk:
                        choices = openai_chunk.get("choices", [])
                        if choices and isinstance(choices[0], dict):
                            delta = choices[0].get("delta", {})
                            c = delta.get("content", "")
                            if c:
                                full_text += c
                            r = delta.get("reasoning_content", "")
                            if r:
                                full_reasoning += r

                # ── 工具调用检测（无条件尝试；缺 tools 时只认 DSML / 哨兵标记） ──
                parsed = parse_tool_calls(
                    full_text,
                    payload.get("tools"),
                    require_marker=not has_tools,
                )
                if parsed.calls:
                    self.logger.info(
                        "ToolCall   | blocking | %s",
                        ", ".join(c.name for c in parsed.calls),
                    )
                    self._store_after_response(
                        resolved, session_id, parsed.clean_text,
                        last_response_message_id, tool_calls=parsed.calls,
                    )
                    if (
                        self.config.delete_conversation
                        and resolved["mode"] == "fresh"
                        and session_id
                    ):
                        self._delete_session(session_id)
                    return self._build_tool_call_result(
                        payload, resolved, parsed.calls, accumulated_tokens,
                    )

                # JSON 模式校验
                if self.config.strip_citations:
                    full_text = strip_citation_marks(full_text)
                self._validate_response_format(payload, full_text)

                self._store_after_response(resolved, session_id, full_text, last_response_message_id)

                if self.config.delete_conversation and resolved["mode"] == "fresh" and session_id:
                    self._delete_session(session_id)

                return self._build_non_stream_result(
                    payload, resolved, full_text, full_reasoning, accumulated_tokens,
                )

            except ContextLengthExceededError:
                context_retries += 1
                self.logger.warning(
                    "上下文超限 (attempt=%d/%d)，从缓存移除并删除会话，使用完整历史重开新对话",
                    context_retries, MAX_CONTEXT_LENGTH_RETRIES,
                )
                self._invalidate_and_delete_session(resolved)
                if context_retries >= MAX_CONTEXT_LENGTH_RETRIES:
                    raise
                # 重新解析 → 由于缓存被清空 + 会话已删除，将走 fresh 路径
                resolved = self._resolve_conversation(payload)

            except RateLimitError as exc:
                rate_limit_retries += 1
                if rate_limit_retries >= MAX_RATE_LIMIT_RETRIES:
                    self.logger.error(
                        "上游限流重试次数耗尽 (%d/%d)",
                        rate_limit_retries, MAX_RATE_LIMIT_RETRIES,
                    )
                    raise UpstreamAPIError(
                        status_code=429,
                        message=f"上游限流: {exc}",
                    ) from exc

                self.logger.warning(
                    "上游限流 (%d/%d)，%d 秒后重试... 原因: %s",
                    rate_limit_retries, MAX_RATE_LIMIT_RETRIES,
                    RATE_LIMIT_COOLDOWN_SECONDS, exc,
                )
                time.sleep(RATE_LIMIT_COOLDOWN_SECONDS)
                # 继续 while True，resolved 保持原样（session 已在 _open_chat_stream 里回写）

            finally:
                if response is not None:
                    try:
                        response.close()
                    except Exception:
                        pass

    def _upload_images_from_messages(
        self, messages: list[dict],
    ) -> list[str]:
        """把消息里所有 image_url 上传成 DeepSeek file_id。"""
        if not self.file_upload_enabled:
            return []

        images = extract_images_from_messages(messages)
        if not images:
            return []

        self.logger.info("Vision | 检测到 %d 张图片，开始上传", len(images))
        file_ids: list[str] = []
        for i, img in enumerate(images):
            name_hint = f"image_{i}.png"
            file_id = self.file_uploader.upload_from_openai_image(
                img["url"], name_hint=name_hint,
            )
            if file_id:
                file_ids.append(file_id)
            else:
                self.logger.warning("Vision | 第 %d 张图片上传失败，已跳过", i)

        return file_ids

    # ═══════════════════════════════════════════════════════
    # Public API - stream
    # ═══════════════════════════════════════════════════════

    def stream_chat_completion(self, payload: dict[str, object]):
        resolved = self._resolve_conversation(payload)
        lease = self.request_queue.acquire(f"stream:{payload.get('model', 'unknown')}")

        include_usage = bool(
            (payload.get("stream_options") or {}).get("include_usage", False)
            if isinstance(payload.get("stream_options"), dict) else False
        )
        has_tools = _payload_has_tools(payload)

        def generate():
            current_resolved = resolved
            current_session_id = current_resolved.get("session_id") or ""
            anything_yielded = False
            context_retries = 0
            rate_limit_retries = 0

            try:
                while True:
                    response = None
                    full_text = ""
                    full_reasoning = ""
                    try:
                        response, current_session_id = self._open_chat_stream(current_resolved)

                        state = StreamState()
                        stripper = CitationStripper() if self.config.strip_citations else None
                        last_response_message_id: str | None = None
                        accumulated_tokens = 0

                        # ★ 检测器永远启动。
                        #   has_tools=True  → require_marker=False：全套解析（DSML / 哨兵 / 代码块 / 裸 JSON）
                        #   has_tools=False → require_marker=True ：只认 DSML / 哨兵标记，
                        #                     防止模型无视 tool_choice=none 或缺失 tools 时
                        #                     把 DSML 块原样漏进 content
                        detector = StreamingToolDetector(
                            payload.get("tools"),
                            require_marker=not has_tools,
                        )
                        tool_calls_emitted = False
                        emitted_calls: list[ParsedToolCall] = []
                        upstream_finish_reason: str | None = None

                        for event in self._iter_sse_events(response):
                            if not event:
                                continue

                            if isinstance(event, dict) and event.get("_event_type") == "hint":
                                if event.get("type") == "error":
                                    fr = event.get("finish_reason")
                                    if fr == "context_length_exceeded":
                                        raise ContextLengthExceededError("DeepSeek: 达到对话长度上限")
                                    if fr == "rate_limit_reached":
                                        raise RateLimitError(
                                            event.get("content") or "DeepSeek: 消息发送过于频繁"
                                        )
                                continue

                            if isinstance(event, dict) and event.get("_event_type") == "ready":
                                resp_id = event.get("response_message_id")
                                if resp_id is not None:
                                    last_response_message_id = str(resp_id)

                            if isinstance(event, dict) and event.get("o") == "BATCH":
                                batch_v = event.get("v")
                                if isinstance(batch_v, list):
                                    for item in batch_v:
                                        if isinstance(item, dict) and item.get("p") == "accumulated_token_usage":
                                            try:
                                                accumulated_tokens = int(item.get("v", 0))
                                            except (TypeError, ValueError):
                                                pass

                            openai_chunk = convert_deepseek_to_openai_chunk(
                                event, current_resolved["model"], state
                            )

                            if not openai_chunk:
                                if self._is_finish_event(event):
                                    break
                                continue

                            choices = openai_chunk.get("choices") or []
                            if not choices or not isinstance(choices[0], dict):
                                yield _sse(openai_chunk)
                                anything_yielded = True
                                if self._is_finish_event(event):
                                    break
                                continue

                            delta = choices[0].get("delta") or {}
                            finish_reason = choices[0].get("finish_reason")
                            c = delta.get("content", "") or ""
                            r = delta.get("reasoning_content", "") or ""

                            if r:
                                # reasoning 原样透传，不参与检测
                                full_reasoning += r
                                yield _sse(openai_chunk)
                                anything_yielded = True
                            elif c:
                                # ★ 内容统一喂给检测器
                                full_text += c
                                for action in detector.feed(c):
                                    if action.kind == "text" and action.text:
                                        out = stripper.feed(action.text) if stripper else action.text
                                        if out:
                                            yield _make_content_chunk(out, current_resolved["model"])
                                            anything_yielded = True
                                    elif action.kind == "tool_calls" and action.tool_calls:
                                        for ch in _make_tool_call_chunks(
                                            action.tool_calls, current_resolved["model"]
                                        ):
                                            yield ch
                                        tool_calls_emitted = True
                                        emitted_calls.extend(action.tool_calls)
                                        anything_yielded = True
                            elif finish_reason is not None:
                                # 上游 finish chunk：记录但不透传，我们最后自己发
                                upstream_finish_reason = finish_reason
                            else:
                                # 纯 role chunk / 空 delta chunk：原样透传
                                yield _sse(openai_chunk)
                                anything_yielded = True

                            if self._is_finish_event(event):
                                break

                        # ── 流结束收尾 ──
                        # 1. 检测器残余
                        for action in detector.finalize():
                            if action.kind == "text" and action.text:
                                out = stripper.feed(action.text) if stripper else action.text
                                if out:
                                    yield _make_content_chunk(out, current_resolved["model"])
                                    anything_yielded = True
                            elif action.kind == "tool_calls" and action.tool_calls:
                                for ch in _make_tool_call_chunks(
                                    action.tool_calls, current_resolved["model"]
                                ):
                                    yield ch
                                tool_calls_emitted = True
                                emitted_calls.extend(action.tool_calls)
                                anything_yielded = True

                        # 2. stripper 尾部残留
                        if stripper is not None:
                            tail = stripper.finalize()
                            if tail:
                                yield _make_content_chunk(tail, current_resolved["model"])
                                anything_yielded = True

                        # 3. JSON 模式校验（仅在没有工具调用发出时才有意义）
                        if not tool_calls_emitted:
                            self._validate_response_format(payload, full_text)

                        # 4. usage chunk
                        if include_usage:
                            prompt_text = current_resolved["ds_payload"].get("prompt", "")
                            prompt_tokens = max(1, len(prompt_text) // 4) if prompt_text else 0
                            completion_tokens = accumulated_tokens or max(1, len(full_text) // 4)
                            usage_chunk = {
                                "id": f"chatcmpl-{uuid.uuid4().hex[:24]}",
                                "object": "chat.completion.chunk",
                                "created": int(time.time()),
                                "model": current_resolved["model"],
                                "system_fingerprint": "",
                                "choices": [],
                                "usage": {
                                    "prompt_tokens": prompt_tokens,
                                    "completion_tokens": completion_tokens,
                                    "total_tokens": prompt_tokens + completion_tokens,
                                },
                            }
                            yield _sse(usage_chunk)

                        # 5. ★ 统一在末尾发 finish chunk
                        #    发出过工具调用 → "tool_calls"
                        #    否则沿用上游 finish_reason（一般 "stop"），缺失时兜底 "stop"
                        if tool_calls_emitted:
                            final_finish = "tool_calls"
                        else:
                            final_finish = upstream_finish_reason or "stop"
                        yield _make_finish_chunk(final_finish, current_resolved["model"])

                        yield "data: [DONE]\n\n"

                        # 6. 缓存 / 会话记录
                        if full_text or full_reasoning or tool_calls_emitted:
                            stored_text = (
                                strip_citation_marks(full_text)
                                if self.config.strip_citations
                                else full_text
                            )
                            self._store_after_response(
                                current_resolved,
                                current_session_id,
                                stored_text,
                                last_response_message_id,
                                tool_calls=emitted_calls or None,
                            )

                        if (
                            self.config.delete_conversation
                            and current_resolved["mode"] == "fresh"
                            and current_session_id
                        ):
                            self._delete_session(current_session_id)

                        return  # success

                    except ContextLengthExceededError:
                        context_retries += 1
                        self.logger.warning(
                            "流式上下文超限 (attempt=%d/%d, yielded=%s)",
                            context_retries, MAX_CONTEXT_LENGTH_RETRIES, anything_yielded,
                        )
                        self._invalidate_and_delete_session(current_resolved)
                        if anything_yielded or context_retries >= MAX_CONTEXT_LENGTH_RETRIES:
                            raise
                        current_resolved = self._resolve_conversation(payload)
                        current_session_id = current_resolved.get("session_id") or ""

                    except RateLimitError as exc:
                        rate_limit_retries += 1
                        if anything_yielded:
                            self.logger.error(
                                "流式已 yield %d 字节内容，无法重试限流: %s",
                                len(full_text), exc,
                            )
                            raise UpstreamAPIError(
                                status_code=429,
                                message=f"上游限流（已在流式响应中）: {exc}",
                            ) from exc

                        if rate_limit_retries >= MAX_RATE_LIMIT_RETRIES:
                            self.logger.error(
                                "上游限流重试次数耗尽 (%d/%d)",
                                rate_limit_retries, MAX_RATE_LIMIT_RETRIES,
                            )
                            raise UpstreamAPIError(
                                status_code=429,
                                message=f"上游限流: {exc}",
                            ) from exc

                        self.logger.warning(
                            "上游限流 (%d/%d)，%d 秒后重试... 原因: %s",
                            rate_limit_retries, MAX_RATE_LIMIT_RETRIES,
                            RATE_LIMIT_COOLDOWN_SECONDS, exc,
                        )
                        time.sleep(RATE_LIMIT_COOLDOWN_SECONDS)
                        # 继续 while True；current_session_id 已由 _open_chat_stream 回写

                    finally:
                        if response is not None:
                            try:
                                response.close()
                            except Exception:
                                pass
            finally:
                lease.release()

        return generate()

    # ═══════════════════════════════════════════════════════
    # Anti rate-limit
    # ═══════════════════════════════════════════════════════

    def _anti_rate_limit_pause(self, context: str = "") -> None:
        """
        防风控：在每次真正打到上游的聊天请求前随机延迟。

        触发点：_open_chat_stream() 的入口。这样天然覆盖：
          - 非流式 / 流式
          - 缓存命中 (mode=cached) / 全量重放 (mode=fresh)
          - rate limit 重试、context length 重试
          - 工具链的每一轮（工具调用走的是同一条 chat 通道）
        """
        if not self.config.anti_rate_limit_enabled:
            return

        min_d = self.config.anti_rate_limit_min_delay
        max_d = self.config.anti_rate_limit_max_delay
        if max_d <= 0:
            return

        delay = max_d if min_d >= max_d else random.uniform(min_d, max_d)

        self.logger.info(
            "AntiRateLimit | sleep=%.2fs before upstream request%s",
            delay,
            f" ({context})" if context else "",
        )
        time.sleep(delay)

    # ═══════════════════════════════════════════════════════
    # Internal: request pipeline
    # ═══════════════════════════════════════════════════════

    def _open_chat_stream(self, resolved: dict):
        self._anti_rate_limit_pause(context=f"mode={resolved['mode']}")
        ds_payload = resolved["ds_payload"]
        mode = resolved["mode"]
        session_id = resolved["session_id"]

        if mode == "fresh":
            if session_id:
                # rate limit 重试路径：上次已经创建了 session，直接复用
                self.logger.info(
                    "Step 3/4: 复用已创建的 session=%s...（rate limit 重试）",
                    session_id[:12],
                )
                ds_payload["chat_session_id"] = session_id
            else:
                self.logger.info("Step 1/4: 获取 PoW challenge (session)...")
                pow_challenge = self._get_pow_challenge()

                self.logger.info("Step 2/4: 求解 PoW (difficulty=%s)...", pow_challenge.get("difficulty"))
                session_pow_header = self.pow_solver.solve(
                    challenge=pow_challenge["challenge"],
                    salt=pow_challenge["salt"],
                    difficulty=pow_challenge["difficulty"],
                    algorithm=pow_challenge.get("algorithm", "DeepSeekHashV1"),
                    expire_at=pow_challenge.get("expire_at", 0),
                    signature=pow_challenge.get("signature", ""),
                )
                self.logger.info("PoW 求解完成")

                self.logger.info("Step 3/4: 创建新 chat session...")
                session_id = self._create_session(session_pow_header)
                ds_payload["chat_session_id"] = session_id
                # 关键：回写 resolved，rate limit 重试时能复用
                resolved["session_id"] = session_id
                self.logger.info("Session 创建成功: %s", session_id)
        else:
            self.logger.info("Step 3/4: 复用缓存 session=%s...", session_id[:12])
            cached_state = self.session_cache.get(resolved["cache_key"])
            if cached_state and cached_state.last_message_id is not None:
                # 上游要求 parent_message_id 必须是 u32
                try:
                    parent_id = int(cached_state.last_message_id)
                except (TypeError, ValueError):
                    self.logger.warning(
                        "无效的 last_message_id，忽略 parent_message_id: %r",
                        cached_state.last_message_id,
                    )
                else:
                    ds_payload["parent_message_id"] = parent_id
                    self.logger.info("使用 parent_message_id=%d", parent_id)

        model = ds_payload.get("model_type", self.config.default_model)
        self.logger.info("Step 4/4: 发送 completion 请求 model=%s mode=%s...", model, mode)

        self.logger.info("获取 PoW challenge (completion)...")
        completion_pow_challenge = self._get_pow_challenge()

        self.logger.info("求解 PoW (completion, difficulty=%s)...", completion_pow_challenge.get("difficulty"))
        completion_pow_header = self.pow_solver.solve(
            challenge=completion_pow_challenge["challenge"],
            salt=completion_pow_challenge["salt"],
            difficulty=completion_pow_challenge["difficulty"],
            algorithm=completion_pow_challenge.get("algorithm", "DeepSeekHashV1"),
            expire_at=completion_pow_challenge.get("expire_at", 0),
            signature=completion_pow_challenge.get("signature", ""),
            target_path=completion_pow_challenge.get(
                "target_path", "/api/v0/chat/completion",
            ),                                       # ← 新增
        )
        self.logger.info("Completion PoW 求解完成")

        body = json.dumps(ds_payload, ensure_ascii=False).encode("utf-8")

        headers = self.auth.build_full_headers(
            pow_response=completion_pow_header,
            content_type="application/json",
        )

        debug_dump(self.logger, self.config.debug_dump_all, "DeepSeek completion 请求体", body)
        debug_dump(self.logger, self.config.debug_dump_all, "DeepSeek completion 请求头", headers)

        try:
            request = urllib.request.Request(
                self.config.chat_completion_url,
                data=body,
                method="POST",
                headers=headers,
            )
            response = urllib.request.urlopen(request, timeout=self.config.request_timeout)
            self.auth.absorb_response_cookies(response)
        except urllib.error.HTTPError as exc:
            error_payload = self._read_error_payload(exc)
            message = self._build_error_message(exc.code, error_payload)

            if mode == "cached":
                self.session_cache.delete(resolved["cache_key"])
                self.logger.warning("缓存 session 失效，已从缓存移除: %s", resolved["cache_key"][:12])

            raise UpstreamAPIError(status_code=exc.code, message=message, payload=error_payload) from exc
        except urllib.error.URLError as exc:
            raise UpstreamAPIError(status_code=502, message=f"网络连接失败: {exc}") from exc

        # 非 SSE 响应处理
        content_type = response.headers.get("Content-Type", "").lower()
        if "text/event-stream" not in content_type:
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

                debug_dump(self.logger, True, "⚠️ DeepSeek 返回非 SSE 响应", raw_body)
                text = raw_body.decode("utf-8", errors="replace")

                if mode == "cached":
                    self.session_cache.delete(resolved["cache_key"])

                try:
                    error_payload = json.loads(text)
                    if isinstance(error_payload, dict):
                        status_code = self._map_upstream_error(error_payload)
                        biz_err = (
                            error_payload.get("biz_data", {})
                            .get("biz_data", {})
                            .get("error", {})
                        )
                        error_msg = (
                            biz_err.get("msg")
                            or error_payload.get("msg")
                            or error_payload.get("message")
                            or text
                        )
                        raise UpstreamAPIError(
                            status_code=status_code,
                            message=f"DeepSeek 返回错误: {error_msg}",
                            payload=error_payload,
                        )
                except json.JSONDecodeError:
                    pass

                raise UpstreamAPIError(
                    status_code=502,
                    message=f"DeepSeek 返回非 SSE 响应: {text[:200]}",
                    payload={"raw": text},
                )
            except Exception:
                response.close()
                raise

        return response, session_id

    # ═══════════════════════════════════════════════════════
    # Upstream error mapping
    # ═══════════════════════════════════════════════════════

    @staticmethod
    def _map_upstream_error(payload: dict) -> int:
        code = payload.get("code")
        msg = str(payload.get("msg", "")).upper()

        if code == 40300 or "MISSING_HEADER" in msg:
            return 502
        if "INVALID_POW" in msg:
            return 502
        if code == 429 or "RATE" in msg or "LIMIT" in msg:
            return 429
        if code == 400 or "INVALID" in msg or "PARAM" in msg:
            return 400
        if "TIMEOUT" in msg:
            return 504
        return 502

    # ═══════════════════════════════════════════════════════
    # Response helpers
    # ═══════════════════════════════════════════════════════

    def _validate_response_format(self, payload: dict, full_text: str) -> None:
        response_format = payload.get("response_format") or {}
        if (
            isinstance(response_format, dict)
            and response_format.get("type") == "json_object"
        ):
            try:
                json.loads(full_text)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Model did not return valid JSON: {exc}") from exc

    def _build_non_stream_result(
        self,
        payload: dict,
        resolved: dict,
        full_text: str,
        full_reasoning: str,
        accumulated_tokens: int,
    ) -> dict:
        prompt_text = resolved["ds_payload"].get("prompt", "")
        prompt_tokens = max(1, len(prompt_text) // 4) if prompt_text else 0
        completion_tokens = accumulated_tokens or max(1, len(full_text) // 4)

        message: dict = {
            "role": "assistant",
            "content": full_text,
        }
        if full_reasoning:
            message["reasoning_content"] = full_reasoning

        return {
            "id": f"chatcmpl-{uuid.uuid4().hex[:24]}",
            "object": "chat.completion",
            "created": int(time.time()),
            "model": str(payload.get("model", self.config.default_model)),
            "choices": [
                {
                    "index": 0,
                    "message": message,
                    "finish_reason": "stop",
                    "logprobs": None,
                }
            ],
            "usage": {
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "total_tokens": prompt_tokens + completion_tokens,
            },
        }

    def _build_tool_call_result(
        self,
        payload: dict,
        resolved: dict,
        calls: list[ParsedToolCall],
        accumulated_tokens: int,
    ) -> dict:
        prompt_text = resolved["ds_payload"].get("prompt", "")
        prompt_tokens = max(1, len(prompt_text) // 4) if prompt_text else 0
        completion_tokens = accumulated_tokens or max(
            1, sum(len(c.arguments) for c in calls) // 4
        )

        return {
            "id": f"chatcmpl-{uuid.uuid4().hex[:24]}",
            "object": "chat.completion",
            "created": int(time.time()),
            "model": str(payload.get("model", self.config.default_model)),
            "choices": [
                {
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": None,
                        "tool_calls": [c.to_openai() for c in calls],
                    },
                    "finish_reason": "tool_calls",
                    "logprobs": None,
                }
            ],
            "usage": {
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "total_tokens": prompt_tokens + completion_tokens,
            },
        }

    # ═══════════════════════════════════════════════════════
    # DeepSeek API calls
    # ═══════════════════════════════════════════════════════

    def _get_pow_challenge(self) -> dict[str, object]:
        body = json.dumps({
            "target_path": "/api/v0/chat/completion",
        }).encode("utf-8")

        headers = {
            **self.auth.get_auth_headers(),
            "Content-Type": "application/json",
        }

        try:
            request = urllib.request.Request(
                self.config.pow_challenge_url,
                data=body,
                method="POST",
                headers=headers,
            )
            with urllib.request.urlopen(request, timeout=15) as response:
                self.auth.absorb_response_cookies(response)
                payload = self.auth._read_json_response(response)
        except urllib.error.HTTPError as exc:
            error_payload = self._read_error_payload(exc)
            raise UpstreamAPIError(
                status_code=exc.code,
                message=f"获取 PoW challenge 失败: {self._build_error_message(exc.code, error_payload)}",
                payload=error_payload,
            ) from exc
        except urllib.error.URLError as exc:
            raise UpstreamAPIError(status_code=502, message=f"获取 PoW challenge 网络失败: {exc}") from exc

        biz_data = payload.get("data", {}).get("biz_data", {}).get("challenge", {})
        if not biz_data:
            biz_data = payload.get("data", {})
            if not biz_data.get("challenge"):
                raise UpstreamAPIError(
                    status_code=500,
                    message=f"PoW challenge 响应格式异常: {json.dumps(payload, ensure_ascii=False)[:300]}",
                    payload=payload,
                )

        challenge = {
            "algorithm": biz_data.get("algorithm", "DeepSeekHashV1"),
            "challenge": biz_data.get("challenge", ""),
            "salt": biz_data.get("salt", ""),
            "signature": biz_data.get("signature", ""),
            "difficulty": int(biz_data.get("difficulty", 144000)),
            "expire_at": int(biz_data.get("expire_at", 0)),
            "target_path": biz_data.get("target_path", "/api/v0/chat/completion"),
        }

        debug_dump(self.logger, self.config.debug_dump_all, "PoW challenge 响应", challenge)
        return challenge

    def _create_session(self, pow_header: str) -> str:
        body = json.dumps({}).encode("utf-8")

        headers = self.auth.build_full_headers(
            pow_response=pow_header,
            content_type="application/json",
        )

        try:
            request = urllib.request.Request(
                self.config.session_create_url,
                data=body,
                method="POST",
                headers=headers,
            )
            with urllib.request.urlopen(request, timeout=15) as response:
                self.auth.absorb_response_cookies(response)
                payload = self.auth._read_json_response(response)
        except urllib.error.HTTPError as exc:
            error_payload = self._read_error_payload(exc)
            raise UpstreamAPIError(
                status_code=exc.code,
                message=f"创建 session 失败: {self._build_error_message(exc.code, error_payload)}",
                payload=error_payload,
            ) from exc
        except urllib.error.URLError as exc:
            raise UpstreamAPIError(status_code=502, message=f"创建 session 网络失败: {exc}") from exc

        data = payload.get("data", {}) if isinstance(payload, dict) else {}
        biz_data = data.get("biz_data", {}) if isinstance(data, dict) else {}

        session_id = ""
        chat_session = biz_data.get("chat_session")
        if isinstance(chat_session, dict):
            session_id = str(chat_session.get("id", "") or "")
        if not session_id:
            session_id = str(biz_data.get("id", "") or "")
        if not session_id:
            session_id = str(biz_data.get("chat_session_id", "") or "")
        if not session_id:
            session_id = str(data.get("id", "") or "")

        if not session_id:
            raise UpstreamAPIError(
                status_code=500,
                message=f"创建 session 响应中未找到 id: {json.dumps(payload, ensure_ascii=False)[:300]}",
                payload=payload,
            )

        debug_dump(self.logger, self.config.debug_dump_all, "Session 创建响应", payload)
        self.logger.info("Session ID 提取成功: %s", session_id)
        return session_id

    def _delete_session(self, session_id: str) -> None:
        if not session_id:
            return
        try:
            body = json.dumps({"chat_session_ids": [session_id]}).encode("utf-8")
            headers = {
                **self.auth.get_auth_headers(),
                "Content-Type": "application/json",
            }
            request = urllib.request.Request(
                self.config.session_delete_url,
                data=body,
                method="POST",
                headers=headers,
            )
            with urllib.request.urlopen(request, timeout=10) as response:
                self.auth.absorb_response_cookies(response)

            with self._proxy_sessions_lock:
                self._proxy_sessions.discard(session_id)

            self.logger.info("已删除 DeepSeek 会话 session_id=%s", session_id)
        except Exception as exc:
            self.logger.warning("删除 DeepSeek 会话失败 session_id=%s error=%s", session_id, exc)

    def _is_finish_event(self, event: dict) -> bool:
        data = event.get("data")
        if isinstance(data, dict):
            if data.get("p") == "response/status" and data.get("v") == "FINISHED":
                return True
            if data.get("p") == "response" and data.get("o") == "BATCH":
                batch_v = data.get("v")
                if isinstance(batch_v, list):
                    for item in batch_v:
                        if isinstance(item, dict) and item.get("p") == "quasi_status" and item.get("v") == "FINISHED":
                            return True
        if event.get("_event_type") == "close":
            return True
        return False

    def _iter_sse_events(self, response):
        pending = ""
        decoder = codecs.getincrementaldecoder("utf-8")("ignore")

        def emit_block(block: str):
            lines = block.split("\n")
            event_type = None
            data_lines = []

            for line in lines:
                if line.startswith("event:"):
                    event_type = line[6:].strip()
                elif line.startswith("data:"):
                    data_lines.append(line[5:].strip())
                elif line.startswith(":"):
                    continue

            if not data_lines:
                return None

            payload_str = "\n".join(data_lines)
            debug_dump(self.logger, self.config.debug_dump_all, "DeepSeek 原始 SSE block", block)

            if event_type in ("ready", "title", "close", "hint"):
                try:
                    parsed = json.loads(payload_str)
                except json.JSONDecodeError:
                    parsed = {"raw": payload_str}
                if isinstance(parsed, dict):
                    parsed["_event_type"] = event_type
                return parsed

            if event_type == "update_session":
                return None

            try:
                parsed = json.loads(payload_str)
                if event_type:
                    parsed["_event_type"] = event_type
                debug_dump(self.logger, self.config.debug_dump_all, "DeepSeek 解析后的 SSE payload", parsed)
                return parsed
            except json.JSONDecodeError:
                self.logger.debug("忽略无法解析的 SSE 片段: %s", payload_str[:200])
                return None

        while True:
            try:
                raw_chunk = response.read(4096)
            except http.client.IncompleteRead as exc:
                raw_chunk = exc.partial or b""
                self.logger.warning("上游 SSE 连接提前断开(IncompleteRead) bytes=%s", len(raw_chunk))
                if raw_chunk:
                    pending += decoder.decode(raw_chunk, False).replace("\r\n", "\n")
                    while "\n\n" in pending:
                        blk, pending = pending.split("\n\n", 1)
                        event = emit_block(blk.strip())
                        if event:
                            yield event
                break
            except (ConnectionResetError, ConnectionAbortedError, BrokenPipeError) as exc:
                self.logger.warning("上游 SSE 连接被重置(%s)", type(exc).__name__)
                break
            except (socket.timeout, TimeoutError) as exc:
                self.logger.warning("上游 SSE 读取超时(%s)", type(exc).__name__)
                break
            except OSError as exc:
                self.logger.warning("上游 SSE 网络 IO 异常(%s: %s)", type(exc).__name__, exc)
                break

            if not raw_chunk:
                break

            pending += decoder.decode(raw_chunk, False).replace("\r\n", "\n")

            while "\n\n" in pending:
                blk, pending = pending.split("\n\n", 1)
                event = emit_block(blk.strip())
                if event:
                    yield event

        remaining = decoder.decode(b"", True)
        if remaining:
            pending += remaining

        if pending.strip():
            for blk in pending.split("\n\n"):
                if blk.strip():
                    event = emit_block(blk.strip())
                    if event:
                        yield event

    def _read_error_payload(self, error: urllib.error.HTTPError) -> dict[str, object]:
        try:
            raw_body = error.read()
            content_encoding = error.headers.get("Content-Encoding", "").lower()
            if content_encoding == "gzip":
                raw_body = gzip.decompress(raw_body)
            text = raw_body.decode("utf-8", errors="ignore")
        except Exception as exc:
            return {"message": f"读取上游错误响应失败: {exc}"}

        try:
            payload = json.loads(text)
            if isinstance(payload, dict):
                return payload
        except json.JSONDecodeError:
            pass
        return {"message": text}

    def _build_error_message(self, status_code: int, payload: dict[str, object]) -> str:
        biz_data = payload.get("biz_data", {})
        if isinstance(biz_data, dict):
            inner = biz_data.get("biz_data", {})
            if isinstance(inner, dict):
                error = inner.get("error", {})
                if isinstance(error, dict) and error.get("msg"):
                    return error["msg"]
            if biz_data.get("msg"):
                return str(biz_data["msg"])

        message = str(payload.get("msg", "")).strip()
        if not message:
            message = str(payload.get("message", "")).strip()
        if not message:
            message = f"DeepSeek 请求失败 HTTP {status_code}"
        return message