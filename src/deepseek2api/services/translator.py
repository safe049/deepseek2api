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
    if isinstance(content, list):
        text_parts: list[str] = []
        for part in content:
            if isinstance(part, dict):
                if part.get("type") == "text":
                    text_parts.append(str(part.get("text", "")))
                elif part.get("type") == "image_url":
                    url = part.get("image_url", {})
                    if isinstance(url, dict):
                        url = url.get("url", "")
                    if url:
                        text_parts.append(f"[image:{url}]")
            elif isinstance(part, str):
                text_parts.append(part)
        return "\n".join(text_parts)
    if content is None:
        return ""
    return str(content)


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


def _resolve_model_and_type(openai_payload: dict, config: AppConfig) -> tuple[str, str, bool]:
    """Return (model, model_type, thinking_enabled)."""
    model = str(openai_payload.get("model", config.default_model))
    if model in ("deepseek-reasoner", "deepseek-r1"):
        return model, "expert", True
    return model, "default", False


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
        return _build_chunk(model, {"role": "assistant", "content": ""})

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
        delta = {}
        if state.is_first_content:
            delta["role"] = "assistant"
            state.is_first_content = False
        if state.last_fragment_type == "THINKING":
            delta["reasoning_content"] = value
        else:
            delta["content"] = value
        return _build_chunk(model, delta)

    # APPEND
    if operation == "APPEND" and "content" in path:
        if not isinstance(value, str) or not value:
            return None
        delta = {}
        if state.is_first_content:
            delta["role"] = "assistant"
            state.is_first_content = False
        if "thinking" in path.lower() or state.last_fragment_type == "THINKING":
            delta["reasoning_content"] = value
        else:
            delta["content"] = value
        return _build_chunk(model, delta)

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
                for frag in fragments:
                    if not isinstance(frag, dict):
                        continue
                    frag_id = frag.get("id")
                    frag_type = frag.get("type", "RESPONSE")
                    content = frag.get("content", "")
                    if frag_id in state.emitted_fragment_ids:
                        continue
                    if not content or not isinstance(content, str):
                        continue

                    state.emitted_fragment_ids.add(frag_id)
                    state.last_fragment_type = frag_type

                    delta = {}
                    if state.is_first_content:
                        delta["role"] = "assistant"
                        state.is_first_content = False

                    if frag_type == "THINKING":
                        delta["reasoning_content"] = content
                    else:
                        delta["content"] = content

                    status = response.get("status", "")
                    finish = None
                    if status == "FINISHED" and not state.finished_sent:
                        finish = "stop"
                        state.finished_sent = True
                        state.is_finished = True

                    return _build_chunk(model, delta, finish)

    return None