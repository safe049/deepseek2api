"""
OpenAI ↔ DeepSeek 格式转换器。

- OpenAI 请求 → DeepSeek 请求（支持 response_format=json_object、reasoner→expert）
- DeepSeek SSE → OpenAI chunk（支持 reasoning_content / per-request state）
- 工具调用：把 tools 提示词注入 system，把历史 assistant tool_calls 还原成哨兵格式
"""

from __future__ import annotations

import json
import time
import uuid
from typing import Any

from ..config import AppConfig, EXPOSED_MODELS
from ..core.openai_compat import gen_chatcmpl_id, system_fingerprint
from .tool_calling import (
    TOOL_CLOSE_PLURAL,
    TOOL_OPEN_PLURAL,
    build_tool_prompt,
)


def _format_conversation_block(parts: list[str]) -> str:
    """
    把历史消息格式化为纯文本。
    """
    lines: list[str] = []
    for line in parts:
        if line.startswith("User: "):
            lines.append("【用户】")
            lines.append(line[len("User: "):])
        elif line.startswith("Assistant: "):
            lines.append("【助手】")
            lines.append(line[len("Assistant: "):])
        elif line.startswith("[Tool Result id="):
            idx = line.find("]:")
            if idx != -1:
                header = line[: idx + 1]
                body = line[idx + 1:].lstrip(": ").lstrip()
                lines.append(header)
                lines.append(body)
            else:
                lines.append(line)
        else:
            lines.append(line)
    return "\n".join(lines)


def _format_system_block(parts: list[str]) -> str:
    return "\n".join(parts)


def _extract_text_from_content(content: Any) -> str:
    """只提取文本部分——image_url 由 FileUploader 单独处理。"""
    if isinstance(content, list):
        text_parts: list[str] = []
        for part in content:
            if isinstance(part, dict):
                if part.get("type") == "text":
                    text_parts.append(str(part.get("text", "")))
                # image_url 忽略：通过 ref_file_ids 传递
            elif isinstance(part, str):
                text_parts.append(part)
        return "\n".join(text_parts)
    if content is None:
        return ""
    return str(content)


def extract_images_from_messages(messages: list[dict]) -> list[dict]:
    """
    从 OpenAI 消息中提取所有 image_url 项。

    返回 [{"url": "...", "detail": "..."}]，detail 可能为 None。

    兼容的两种写法：
      {"type": "image_url", "image_url": {"url": "...", "detail": "..."}}
      {"type": "image_url", "image_url": "..."}   # 简写
    """
    images: list[dict] = []
    for m in messages:
        if not isinstance(m, dict):
            continue
        content = m.get("content")
        if not isinstance(content, list):
            continue
        for part in content:
            if not isinstance(part, dict):
                continue
            if part.get("type") != "image_url":
                continue
            url_obj = part.get("image_url")
            if isinstance(url_obj, dict):
                url = url_obj.get("url", "")
                detail = url_obj.get("detail")
            else:
                url = str(url_obj or "")
                detail = None
            if url:
                images.append({"url": url, "detail": detail})
    return images


def _is_json_mode(openai_payload: dict) -> bool:
    response_format = openai_payload.get("response_format") or {}
    return (
        isinstance(response_format, dict)
        and response_format.get("type") == "json_object"
    )


def _has_active_tools(openai_payload: dict) -> bool:
    tools = openai_payload.get("tools")
    if not isinstance(tools, list) or not tools:
        return False
    return openai_payload.get("tool_choice", "auto") != "none"


def _render_assistant_with_tool_calls(msg: dict) -> str:
    """把带 tool_calls 的 assistant 消息渲染成哨兵格式的历史片段。"""
    tcs = msg.get("tool_calls")
    if not isinstance(tcs, list) or not tcs:
        return ""
    lines = [TOOL_OPEN_PLURAL]
    for tc in tcs:
        if not isinstance(tc, dict):
            continue
        fn = tc.get("function") if isinstance(tc.get("function"), dict) else {}
        name = fn.get("name", "")
        raw_args = fn.get("arguments", {})
        if isinstance(raw_args, str):
            try:
                raw_args = json.loads(raw_args)
            except json.JSONDecodeError:
                raw_args = {"value": raw_args}
        if not name:
            continue
        lines.append(json.dumps({"name": name, "arguments": raw_args}, ensure_ascii=False))
    lines.append(TOOL_CLOSE_PLURAL)
    return "\n".join(lines)


_THINKING_MODELS = frozenset({"deepseek-reasoner", "deepseek-r1", "deepseek-thinking"})


def _resolve_model_and_type(openai_payload: dict, config: AppConfig) -> tuple[str, str, bool]:
    """Return (model, model_type, thinking_enabled)."""
    model = str(openai_payload.get("model", config.default_model))
    if model in _THINKING_MODELS:
        return model, "expert", True
    return model, "default", False


def _finalize_mode(model_type: str, thinking_enabled: bool) -> tuple[str, bool]:
    """思考开启时强制 expert：上游只有 expert 通道产出 THINKING 片段。"""
    if thinking_enabled and model_type != "expert":
        return "expert", True
    return model_type, thinking_enabled


def _apply_extra_body(
    openai_payload: dict,
    thinking_enabled: bool,
    config: AppConfig,
) -> tuple[bool, bool]:
    """
    解析 search_enabled / thinking_enabled。

    优先级：
      search_enabled:   extra_body.search_enabled 显式值
                        > config.search_enabled 全局默认
                        > False
      thinking_enabled: extra_body.thinking_enabled 显式值
                        > 模型默认（deepseek-reasoner 强制 true）
    """
    search_enabled = bool(getattr(config, "search_enabled", False))

    extra = openai_payload.get("extra_body", {})
    if isinstance(extra, dict):
        if "search_enabled" in extra:
            search_enabled = bool(extra["search_enabled"])
        thinking_enabled = bool(extra.get("thinking_enabled", thinking_enabled))

    return search_enabled, thinking_enabled

# ═══════════════════════════════════════════════════════════════
# OpenAI → DeepSeek 请求转换
# ═══════════════════════════════════════════════════════════════

def convert_openai_to_deepseek(
    openai_payload: dict[str, object],
    config: AppConfig,
) -> dict[str, object]:
    messages = openai_payload.get("messages", [])
    if not isinstance(messages, list):
        messages = []

    system_parts: list[str] = []
    history_lines: list[str] = []
    current_lines: list[str] = []

    # 找最后一条 user 消息；它及其之后的内容归入"当前输入"
    last_user_idx = -1
    for i in range(len(messages) - 1, -1, -1):
        m = messages[i]
        if isinstance(m, dict) and m.get("role") == "user":
            last_user_idx = i
            break

    for i, m in enumerate(messages):
        if not isinstance(m, dict):
            continue
        role = m.get("role", "user")
        content = _extract_text_from_content(m.get("content", ""))

        if role == "system":
            system_parts.append(content)
            continue

        target = history_lines if (last_user_idx >= 0 and i < last_user_idx) else current_lines

        if role == "assistant":
            if m.get("tool_calls"):
                rendered = _render_assistant_with_tool_calls(m)
                if rendered:
                    target.append(rendered)
            if content:
                target.append(f"Assistant: {content}")
        elif role == "user":
            target.append(f"User: {content}")
        elif role == "tool":
            tcid = m.get("tool_call_id", "")
            target.append(f"[Tool Result id={tcid}]: {content}")

    # 注入工具提示词
    if _has_active_tools(openai_payload):
        tool_prompt = build_tool_prompt(
            openai_payload.get("tools") or [],
            openai_payload.get("tool_choice", "auto"),
        )
        if tool_prompt:
            system_parts.append(tool_prompt)

    prompt_parts: list[str] = []

    if _is_json_mode(openai_payload):
        prompt_parts.append(
            "[系统指令]\n请仅以合法 JSON 格式返回结果，不要包含任何额外文字。"
        )

    if system_parts:
        prompt_parts.append("[系统指令]\n" + _format_system_block(system_parts))

    if history_lines:
        prompt_parts.append("【历史对话】\n" + _format_conversation_block(history_lines))

    if current_lines:
        prompt_parts.append("[当前用户消息]\n" + "\n".join(current_lines))

    prompt = "\n\n".join(prompt_parts) if prompt_parts else ""

    _, model_type, thinking_enabled = _resolve_model_and_type(openai_payload, config)
    search_enabled, thinking_enabled = _apply_extra_body(openai_payload, thinking_enabled, config)
    model_type, thinking_enabled = _finalize_mode(model_type, thinking_enabled)

    return {
        "chat_session_id": "",
        "parent_message_id": None,
        "model_type": model_type,
        "prompt": prompt,
        "ref_file_ids": [],
        "thinking_enabled": thinking_enabled,
        "search_enabled": search_enabled,
        "action": None,
        "preempt": False,
    }


def convert_openai_to_deepseek_incremental(
    openai_payload: dict[str, object],
    config: AppConfig,
) -> dict[str, object]:
    """
    缓存命中时的增量 payload。

    与 fresh 路径的关键区别：
      DeepSeek 会话里已经存在「上一条 assistant 消息及其之前的所有内容」。
      所以我们只发送「最后一条 assistant 消息之后的新消息」——
      通常是 [tool_result] 或 [new_user_msg] 或 [tool_result_1, ..., tool_result_n]。

    边界选择：last_assistant_idx + 1
      而不是 last_user_idx，因为工具链的多轮里 last_user 往往就是最早的
      那条 user 消息，用它做边界会把整个对话重发一遍。

    边界例子：
      [user_1]                                 → (首轮，走 fresh)
      [user_1, assistant_tool, tool_res]        → [tool_res]
      [user_1, assistant_text, user_2]          → [user_2]
      [user_1, assistant_tool, tool_res, assistant_text, user_2]
                                                 → [user_2]
      [user_1, a_tool_1, res_1, a_tool_2, res_2] → [res_2]
    """
    messages = openai_payload.get("messages", [])
    if not isinstance(messages, list):
        messages = []

    system_parts: list[str] = []
    for m in messages:
        if isinstance(m, dict) and m.get("role") == "system":
            system_parts.append(_extract_text_from_content(m.get("content", "")))

    # ── 找最后一条 assistant 消息的位置 ──
    last_assistant_idx = -1
    for i in range(len(messages) - 1, -1, -1):
        m = messages[i]
        if isinstance(m, dict) and m.get("role") == "assistant":
            last_assistant_idx = i
            break

    start = last_assistant_idx + 1

    # ── 收集边界之后的新消息 ──
    current_parts: list[str] = []
    for i in range(start, len(messages)):
        m = messages[i]
        if not isinstance(m, dict):
            continue
        role = m.get("role", "user")
        if role == "system":
            continue
        content = _extract_text_from_content(m.get("content", ""))

        if role == "user":
            current_parts.append(content)
        elif role == "assistant":
            if m.get("tool_calls"):
                rendered = _render_assistant_with_tool_calls(m)
                if rendered:
                    current_parts.append(rendered)
            if content:
                current_parts.append(content)
        elif role == "tool":
            tcid = m.get("tool_call_id", "")
            current_parts.append(f"[Tool Result id={tcid}]: {content}")

    # ── 工具提示词注入 system ──
    if _has_active_tools(openai_payload):
        tool_prompt = build_tool_prompt(
            openai_payload.get("tools") or [],
            openai_payload.get("tool_choice", "auto"),
        )
        if tool_prompt:
            system_parts.append(tool_prompt)

    prompt_parts: list[str] = []

    if _is_json_mode(openai_payload):
        prompt_parts.append(
            "[系统指令]\n请仅以合法 JSON 格式返回结果，不要包含任何额外文字。"
        )

    if system_parts:
        prompt_parts.append("[系统指令]\n" + _format_system_block(system_parts))

    if current_parts:
        prompt_parts.append("[当前用户消息]\n" + "\n".join(current_parts))

    prompt = "\n\n".join(prompt_parts) if prompt_parts else ""

    _, model_type, thinking_enabled = _resolve_model_and_type(openai_payload, config)
    search_enabled, thinking_enabled = _apply_extra_body(openai_payload, thinking_enabled, config)
    model_type, thinking_enabled = _finalize_mode(model_type, thinking_enabled)

    return {
        "chat_session_id": "",
        "parent_message_id": None,
        "model_type": model_type,
        "prompt": prompt,
        "ref_file_ids": [],
        "thinking_enabled": thinking_enabled,
        "search_enabled": search_enabled,
        "action": None,
        "preempt": False,
    }
# ═══════════════════════════════════════════════════════════════
# DeepSeek SSE → OpenAI Chunk 转换
# ═══════════════════════════════════════════════════════════════

# ── 片段类型归一化 ─────────────────────────────────────────────
# 上游（Web 端）实际下发的思考片段 type 是 "THINK"，
# 而旧版文档/测试里写的是 "THINKING"。两者（以及未来的变体）
# 统一映射成内部的 "THINKING" / "RESPONSE"，避免漏判。
_THINKING_FRAGMENT_TYPES = frozenset({
    "THINK", "THINKING", "THINKER", "REASONING", "REASON",
})


def _normalize_fragment_type(frag_type: Any) -> str:
    """把上游 fragment type 归一化为 'THINKING' 或 'RESPONSE'。"""
    if isinstance(frag_type, str):
        upper = frag_type.strip().upper()
        if upper in _THINKING_FRAGMENT_TYPES:
            return "THINKING"
        # 兜底：包含 THINK / REASON 的未知类型也按思考处理
        if "THINK" in upper or "REASON" in upper:
            return "THINKING"
    return "RESPONSE"


class StreamState:
    """每个请求独立维护，避免并发/跨请求串扰。"""

    __slots__ = (
        "request_message_id",
        "response_message_id",
        "is_first_content",
        "is_finished",
        "finished_sent",
        "context_exceeded",
        "last_fragment_type",
        "emitted_fragment_ids",
        "pending_chunks",
    )

    def __init__(self) -> None:
        self.request_message_id: int | None = None
        self.response_message_id: int | None = None
        self.is_first_content: bool = True
        self.is_finished: bool = False
        self.finished_sent: bool = False
        self.context_exceeded: bool = False
        self.last_fragment_type: str = "RESPONSE"
        self.emitted_fragment_ids: set = set()
        self.pending_chunks: list[dict] = []

    def pop_pending_chunks(self) -> list[dict]:
        """取走因一个 SSE 事件含多个片段而暂存的额外 chunk。"""
        if not self.pending_chunks:
            return []
        chunks, self.pending_chunks = self.pending_chunks, []
        return chunks


def _build_kind_chunk(model: str, state: StreamState, kind: str, value: str) -> dict:
    """按显式 kind（'THINKING' / 'RESPONSE'）包装一个增量 chunk。"""
    state.last_fragment_type = kind
    delta: dict = {}
    if state.is_first_content:
        delta["role"] = "assistant"
        state.is_first_content = False
    if kind == "THINKING":
        delta["reasoning_content"] = value
    else:
        delta["content"] = value
    return _build_chunk(model, delta)


def _build_delta_chunk(model: str, state: StreamState, value: str) -> dict:
    """按当前片段类型把一个增量字符串包成 OpenAI chunk。

    思考片段 → delta.reasoning_content；否则 → delta.content。
    """
    return _build_kind_chunk(model, state, state.last_fragment_type, value)


def _build_chunk(model: str, delta: dict, finish_reason: str | None = None) -> dict:
    return {
        "id": gen_chatcmpl_id(),
        "object": "chat.completion.chunk",
        "created": int(time.time()),
        "model": model,
        "system_fingerprint": system_fingerprint(model),
        "choices": [{
            "index": 0,
            "delta": delta,
            "finish_reason": finish_reason,
            "logprobs": None,
        }],
    }


def convert_deepseek_to_openai_chunk(
    ds_event: dict[str, object],
    model: str,
    state: StreamState,
) -> dict[str, object] | None:
    if not isinstance(ds_event, dict):
        return None

    event_type = ds_event.get("_event_type", "")

    # ── ready ──
    if event_type == "ready":
        state.request_message_id = ds_event.get("request_message_id")
        state.response_message_id = ds_event.get("response_message_id")
        state.is_first_content = True
        state.is_finished = False
        state.finished_sent = False
        state.context_exceeded = False
        state.emitted_fragment_ids = set()
        state.last_fragment_type = "RESPONSE"
        # OpenAI 标准首块：content 显式给 null（而非空串），
        # 让严格校验的客户端也认这是 assistant 回合开始。
        return _build_chunk(model, {"role": "assistant", "content": None})

    # ── hint（错误/提示，主要由 client 侧提前捕获） ──
    if event_type == "hint":
        if ds_event.get("type") == "error":
            fr = ds_event.get("finish_reason", "")
            if fr == "context_length_exceeded":
                state.context_exceeded = True
        return None

    # ── close ──
    if event_type == "close":
        if state.finished_sent:
            return None
        state.is_finished = True
        state.finished_sent = True
        return _build_chunk(model, {}, "stop")

    if event_type in ("title", "update_session"):
        return None

    # ── data 行 ──
    data = ds_event if "v" in ds_event and "p" not in ds_event else ds_event.get("data", ds_event)
    if not isinstance(data, dict):
        return None

    path = data.get("p", "")
    operation = data.get("o", "")
    value = data.get("v")

    # 裸 {"v": "..."} 追加
    if not path and not operation and isinstance(value, str) and value:
        return _build_delta_chunk(model, state, value)

    # APPEND 到 response/fragments（新的片段开始）：
    # 上游形如 {"p":"response/fragments","o":"APPEND","v":[{"id":3,"type":"RESPONSE",...}]}
    # 这是 THINK → RESPONSE 的切换信号；必须更新 last_fragment_type，
    # 否则后续裸 {"v": "..."} 增量会一直被当成 reasoning_content。
    if (
        operation == "APPEND"
        and path == "response/fragments"
        and isinstance(value, list)
    ):
        new_type: str | None = None
        for frag in value:
            if not isinstance(frag, dict):
                continue
            frag_id = frag.get("id")
            if frag_id is not None:
                state.emitted_fragment_ids.add(frag_id)
            kind = _normalize_fragment_type(frag.get("type", "RESPONSE"))
            new_type = kind
            content = frag.get("content", "")
            if isinstance(content, str) and content:
                return _build_kind_chunk(model, state, kind, content)
        if new_type is not None:
            state.last_fragment_type = new_type
        return None

    # APPEND 到当前片段的 content
    if operation == "APPEND" and "content" in path:
        if not isinstance(value, str) or not value:
            return None
        return _build_delta_chunk(model, state, value)

    # SET status FINISHED
    if operation == "SET" and path == "response/status" and value == "FINISHED":
        if state.finished_sent:
            return None
        state.is_finished = True
        state.finished_sent = True
        return _build_chunk(model, {}, "stop")

    # BATCH
    if operation == "BATCH" and isinstance(value, list):
        has_finished = False
        for item in value:
            if isinstance(item, dict):
                if item.get("p") == "quasi_status" and item.get("v") == "FINISHED":
                    has_finished = True
                elif item.get("p") == "response/status" and item.get("v") == "FINISHED":
                    has_finished = True
        if has_finished and not state.finished_sent:
            state.is_finished = True
            state.finished_sent = True
            return _build_chunk(model, {}, "stop")

    # 嵌套完整 response（首次携带 fragments 内容）
    if isinstance(value, dict):
        response = value.get("response", {})
        if isinstance(response, dict):
            fragments = response.get("fragments", [])
            if isinstance(fragments, list):
                # 一个事件可能同时带多个片段；合并同一类型的连续文本，
                # 避免旧实现「循环内 return，只发出第一个片段」丢内容。
                merged: list[tuple[str, str]] = []
                for frag in fragments:
                    if not isinstance(frag, dict):
                        continue
                    frag_id = frag.get("id")
                    if frag_id is not None and frag_id in state.emitted_fragment_ids:
                        continue
                    content = frag.get("content", "")
                    if not isinstance(content, str) or not content:
                        continue

                    if frag_id is not None:
                        state.emitted_fragment_ids.add(frag_id)
                    kind = _normalize_fragment_type(frag.get("type", "RESPONSE"))
                    if merged and merged[-1][0] == kind:
                        merged[-1] = (kind, merged[-1][1] + content)
                    else:
                        merged.append((kind, content))

                if not merged:
                    return None

                # 一个 chunk 只能承载一种增量字段，所以按类型分组发出。
                # 只把最后一个 chunk 标上 finish_reason。
                chunks: list[dict] = []
                status = response.get("status", "")
                for i, (kind, text) in enumerate(merged):
                    state.last_fragment_type = kind
                    delta: dict = {}
                    if state.is_first_content:
                        delta["role"] = "assistant"
                        state.is_first_content = False
                    if kind == "THINKING":
                        delta["reasoning_content"] = text
                    else:
                        delta["content"] = text

                    finish = None
                    if (
                        i == len(merged) - 1
                        and status == "FINISHED"
                        and not state.finished_sent
                    ):
                        finish = "stop"
                        state.finished_sent = True
                        state.is_finished = True
                    chunks.append(_build_chunk(model, delta, finish))

                # 调用方约定：返回单个 chunk。多片段时把其余片段
                # 挂到 pending 上，由 pop_pending_chunks() 取走。
                if len(chunks) == 1:
                    return chunks[0]
                state.pending_chunks.extend(chunks[1:])
                return chunks[0]

    return None