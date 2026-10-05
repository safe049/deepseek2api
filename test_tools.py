#!/usr/bin/env python3
"""
deepseek2api 工具调用高级测试。

覆盖:
   1. 单工具非流式
   2. 单工具流式
   3. 多工具并行（非流式）
   4. 多工具并行（流式）
   5. 多轮工具链 + 缓存命中（非流式）
   6. 多轮工具链 + 缓存命中（流式）
   7. tool_choice="none" 时禁用工具
   8. 无工具文本响应回归（非流式）
   9. 无工具文本响应回归（流式）
  10. 并行工具调用的缓存命中（已知限制检查）

依赖: 仅标准库
用法:
   python test_tools_advanced.py
   python test_tools_advanced.py --base-url http://localhost:8000/v1 --api-key sk-xxx
   python test_tools_advanced.py --test 5      # 只跑第 5 项
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import urllib.error
import urllib.request
from typing import Any

# ── ANSI ────────────────────────────────────────────────────────

_GREEN = "\033[92m"
_RED = "\033[91m"
_YELLOW = "\033[93m"
_CYAN = "\033[96m"
_DIM = "\033[2m"
_BOLD = "\033[1m"
_RESET = "\033[0m"


def _ok(msg: str) -> None:
    print(f"  {_GREEN}✔{_RESET} {msg}")


def _fail(msg: str) -> None:
    print(f"  {_RED}✘{_RESET} {msg}")


def _warn(msg: str) -> None:
    print(f"  {_YELLOW}⚠{_RESET} {msg}")


def _info(msg: str) -> None:
    print(f"  {_CYAN}ℹ{_RESET} {msg}")


def _header(title: str) -> None:
    print(f"\n{_BOLD}{'─' * 72}{_RESET}")
    print(f"{_BOLD}  {title}{_RESET}")
    print(f"{_BOLD}{'─' * 72}{_RESET}")


# ═══════════════════════════════════════════════════════════════
# HTTP helpers
# ═══════════════════════════════════════════════════════════════

def _http(url, method="GET", payload=None, api_key="", timeout=180):
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"

    body = None
    if payload is not None:
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")

    req = urllib.request.Request(url, data=body, headers=headers, method=method)
    try:
        resp = urllib.request.urlopen(req, timeout=timeout)
        return resp, None
    except urllib.error.HTTPError as exc:
        raw = exc.read().decode("utf-8", errors="replace")
        return None, (exc.code, raw)
    except urllib.error.URLError as exc:
        return None, (-1, f"network error: {exc}")


def get_health(base_url: str, api_key: str = "") -> dict:
    root = base_url.rstrip("/")
    if root.endswith("/v1"):
        root = root[:-3]
    resp, err = _http(f"{root}/health", api_key=api_key, timeout=10)
    if resp is None:
        return {}
    try:
        return json.loads(resp.read().decode("utf-8"))
    except Exception:
        return {}


def chat_non_stream(base_url, messages, model="deepseek-chat", api_key="",
                    tools=None, tool_choice=None):
    payload: dict[str, Any] = {
        "model": model, "messages": messages, "stream": False,
    }
    if tools is not None:
        payload["tools"] = tools
    if tool_choice is not None:
        payload["tool_choice"] = tool_choice

    resp, err = _http(f"{base_url.rstrip('/')}/chat/completions", "POST", payload, api_key)
    if resp is None:
        code, raw = err
        raise RuntimeError(f"HTTP {code}: {raw[:500]}")
    return json.loads(resp.read().decode("utf-8"))


def chat_stream(base_url, messages, model="deepseek-chat", api_key="",
                tools=None, tool_choice=None):
    """
    返回 (chunks, full_text, finish_reason)。

    chunks: SSE 解析后的 dict 列表（不含 [DONE]）
    full_text: 拼接的所有 content delta
    finish_reason: 最后一个非 None 的 finish_reason
    """
    payload: dict[str, Any] = {
        "model": model, "messages": messages, "stream": True,
        "stream_options": {"include_usage": True},
    }
    if tools is not None:
        payload["tools"] = tools
    if tool_choice is not None:
        payload["tool_choice"] = tool_choice

    resp, err = _http(f"{base_url.rstrip('/')}/chat/completions", "POST", payload, api_key)
    if resp is None:
        code, raw = err
        raise RuntimeError(f"HTTP {code}: {raw[:500]}")

    chunks: list[dict] = []
    full_text = ""
    finish_reason: str | None = None
    usage: dict | None = None

    for raw_line in resp:
        line = raw_line.decode("utf-8", errors="replace").strip()
        if not line.startswith("data:"):
            continue
        data_str = line[5:].strip()
        if data_str == "[DONE]":
            break
        try:
            chunk = json.loads(data_str)
        except json.JSONDecodeError:
            continue
        chunks.append(chunk)

        choices = chunk.get("choices") or []
        if choices and isinstance(choices[0], dict):
            delta = choices[0].get("delta") or {}
            c = delta.get("content")
            if c:
                full_text += c
            fr = choices[0].get("finish_reason")
            if fr:
                finish_reason = fr

        if chunk.get("usage"):
            usage = chunk["usage"]

    # 挂到函数对象上（简化返回）
    chat_stream.last_usage = usage
    return chunks, full_text, finish_reason


def extract_message(resp: dict) -> dict:
    return (resp.get("choices") or [{}])[0].get("message") or {}


def extract_finish_reason(resp: dict) -> str | None:
    return (resp.get("choices") or [{}])[0].get("finish_reason")


def extract_usage(resp: dict) -> dict:
    return resp.get("usage") or {}


# ═══════════════════════════════════════════════════════════════
# Tool call helpers
# ═══════════════════════════════════════════════════════════════

def merge_tool_calls_from_stream(chunks: list[dict]) -> list[dict]:
    """把 SSE chunks 里的 delta.tool_calls 按 index 合并成完整调用列表。"""
    by_index: dict[int, dict] = {}
    for ch in chunks:
        choices = ch.get("choices") or []
        if not choices or not isinstance(choices[0], dict):
            continue
        delta = choices[0].get("delta") or {}
        for tc in (delta.get("tool_calls") or []):
            idx = tc.get("index", 0)
            entry = by_index.setdefault(idx, {
                "id": None,
                "type": "function",
                "function": {"name": "", "arguments": ""},
            })
            if tc.get("id"):
                entry["id"] = tc["id"]
            if tc.get("type"):
                entry["type"] = tc["type"]
            fn = tc.get("function") or {}
            if fn.get("name"):
                entry["function"]["name"] = fn["name"]
            if fn.get("arguments"):
                entry["function"]["arguments"] += fn["arguments"]
    return [by_index[i] for i in sorted(by_index)]


def validate_tool_calls(tc_list: list[dict], known_names: set[str]) -> tuple[bool, list[str]]:
    """校验工具调用列表形状。返回 (ok, errors)。"""
    errors: list[str] = []
    if not tc_list:
        errors.append("tool_calls 列表为空")
        return False, errors

    for i, tc in enumerate(tc_list):
        if not isinstance(tc, dict):
            errors.append(f"[{i}] 不是 dict")
            continue
        if not tc.get("id"):
            errors.append(f"[{i}] 缺少 id")
        if tc.get("type") != "function":
            errors.append(f"[{i}] type != 'function'（实际 {tc.get('type')!r}）")
        fn = tc.get("function")
        if not isinstance(fn, dict):
            errors.append(f"[{i}] 缺少 function 对象")
            continue
        name = fn.get("name")
        if not name:
            errors.append(f"[{i}] function.name 为空")
        elif known_names and name not in known_names:
            errors.append(f"[{i}] 未知工具名 {name!r}（可能幻觉）")
        args_str = fn.get("arguments")
        if not isinstance(args_str, str):
            errors.append(f"[{i}] function.arguments 不是字符串（实际 {type(args_str).__name__}）")
        else:
            try:
                parsed = json.loads(args_str)
                if not isinstance(parsed, dict):
                    errors.append(f"[{i}] arguments 解析后不是 dict")
            except json.JSONDecodeError as e:
                errors.append(f"[{i}] arguments 不是合法 JSON: {e}")

    return (len(errors) == 0), errors


# ═══════════════════════════════════════════════════════════════
# Tools under test
# ═══════════════════════════════════════════════════════════════

WEATHER_TOOL = {
    "type": "function",
    "function": {
        "name": "get_weather",
        "description": "Get the current weather for a city.",
        "parameters": {
            "type": "object",
            "properties": {
                "city": {"type": "string", "description": "City name, e.g. Beijing"},
                "unit": {"type": "string", "enum": ["celsius", "fahrenheit"],
                         "description": "Temperature unit, defaults to celsius"},
            },
            "required": ["city"],
        },
    },
}

TIME_TOOL = {
    "type": "function",
    "function": {
        "name": "get_time",
        "description": "Get the current time in a given IANA timezone.",
        "parameters": {
            "type": "object",
            "properties": {
                "timezone": {"type": "string",
                             "description": "IANA timezone name, e.g. Asia/Shanghai"},
            },
            "required": ["timezone"],
        },
    },
}

ALL_TOOLS = [WEATHER_TOOL, TIME_TOOL]
KNOWN_NAMES = {"get_weather", "get_time"}

# 长 user 消息：用于缓存命中测试（HIT/MISS 时 prompt_tokens 差异明显）
_LONG_FILLER = "这是一段用于测试长上下文缓存命中的填充文本。" * 30
LONG_USER = (
    "以下是一段背景信息，请忽略它，只需关注最后的指令。\n\n"
    + _LONG_FILLER
    + "\n\n现在请调用 get_weather 工具查询北京的天气。"
)


# ═══════════════════════════════════════════════════════════════
# Test context
# ═══════════════════════════════════════════════════════════════

class TestContext:
    def __init__(self, base_url: str, model: str, api_key: str):
        self.base_url = base_url
        self.model = model
        self.api_key = api_key
        self.health: dict = {}

    def load_health(self) -> None:
        self.health = get_health(self.base_url, self.api_key)

    @property
    def delete_mode(self) -> bool:
        return bool((self.health.get("config") or {}).get("delete_conversation", True))

    @property
    def cache_enabled(self) -> bool:
        if self.delete_mode:
            return False
        return bool((self.health.get("cache") or {}).get("enabled", True))


# ═══════════════════════════════════════════════════════════════
# Tests
# ═══════════════════════════════════════════════════════════════

def test_single_tool_non_stream(ctx: TestContext) -> bool:
    _header("Test 1: 单工具非流式")

    resp = chat_non_stream(
        ctx.base_url,
        [{"role": "user", "content": "What is the weather in Beijing? Use the tool."}],
        model=ctx.model, api_key=ctx.api_key, tools=[WEATHER_TOOL],
    )

    fr = extract_finish_reason(resp)
    msg = extract_message(resp)
    tcs = msg.get("tool_calls") or []
    _info(f"finish_reason={fr}")
    _info(f"tool_calls={json.dumps(tcs, ensure_ascii=False)[:240]}")

    passed = True
    if fr != "tool_calls":
        _fail(f"finish_reason 应为 'tool_calls'，实际 {fr!r}")
        passed = False
    else:
        _ok("finish_reason = tool_calls")

    if not tcs:
        _fail("未返回 tool_calls")
        return False
    _ok(f"返回 {len(tcs)} 个工具调用")

    ok, errors = validate_tool_calls(tcs, KNOWN_NAMES)
    if not ok:
        for e in errors:
            _fail(e)
        passed = False
    else:
        _ok("工具调用结构合法")

    if len(tcs) == 1 and tcs[0].get("function", {}).get("name") == "get_weather":
        _ok("正确调用了 get_weather")
    else:
        _warn(f"预期 1 个 get_weather，实际 {len(tcs)} 个")

    return passed


def test_single_tool_stream(ctx: TestContext) -> bool:
    _header("Test 2: 单工具流式")

    chunks, text, fr = chat_stream(
        ctx.base_url,
        [{"role": "user", "content": "What is the weather in Beijing? Use the tool."}],
        model=ctx.model, api_key=ctx.api_key, tools=[WEATHER_TOOL],
    )
    tcs = merge_tool_calls_from_stream(chunks)
    _info(f"总 chunk 数: {len(chunks)} | finish_reason={fr}")
    _info(f"合并后 tool_calls={json.dumps(tcs, ensure_ascii=False)[:240]}")

    passed = True
    if fr != "tool_calls":
        _fail(f"finish_reason 应为 'tool_calls'，实际 {fr!r}")
        passed = False
    else:
        _ok("finish_reason = tool_calls")

    if not tcs:
        _fail("流式未产出 tool_calls")
        return False
    _ok(f"合并后 {len(tcs)} 个工具调用")

    # 校验至少一个 tool_call 有 id 和 name
    if not any(tc.get("id") for tc in tcs):
        _fail("tool_calls 缺少 id")
        passed = False
    else:
        _ok("tool_call 带 id")

    if not any(tc.get("function", {}).get("name") for tc in tcs):
        _fail("tool_calls 缺少 name")
        passed = False
    else:
        _ok("tool_call 带 name")

    # 参数完整
    ok, errors = validate_tool_calls(tcs, KNOWN_NAMES)
    if not ok:
        for e in errors:
            _fail(e)
        passed = False
    else:
        _ok("arguments 是合法 JSON")

    return passed


def test_multi_tool_non_stream(ctx: TestContext) -> bool:
    _header("Test 3: 多工具并行（非流式）")

    resp = chat_non_stream(
        ctx.base_url,
        [{"role": "user",
          "content": "Please tell me both: the weather in Beijing AND the current time in Shanghai. "
                     "Call the appropriate tools."}],
        model=ctx.model, api_key=ctx.api_key, tools=ALL_TOOLS,
    )

    msg = extract_message(resp)
    tcs = msg.get("tool_calls") or []
    fr = extract_finish_reason(resp)
    _info(f"finish_reason={fr}")
    _info(f"tool_calls={json.dumps(tcs, ensure_ascii=False)[:300]}")

    passed = True
    if fr != "tool_calls":
        _fail(f"finish_reason 应为 'tool_calls'，实际 {fr!r}")
        passed = False

    if not tcs:
        _fail("未返回 tool_calls")
        return False

    names = [tc.get("function", {}).get("name") for tc in tcs]
    _info(f"调用的工具: {names}")

    if len(tcs) >= 2:
        _ok(f"返回 {len(tcs)} 个工具调用（并行）")
    else:
        _warn(f"只返回 {len(tcs)} 个工具调用（模型可能合并了请求，非协议问题）")

    ok, errors = validate_tool_calls(tcs, KNOWN_NAMES)
    if not ok:
        for e in errors:
            _fail(e)
        passed = False
    else:
        _ok("所有工具调用结构合法")

    if "get_weather" in names:
        _ok("包含 get_weather")
    else:
        _warn("未调用 get_weather（模型行为，非协议问题）")

    if "get_time" in names:
        _ok("包含 get_time")
    else:
        _warn("未调用 get_time（模型行为，非协议问题）")

    return passed


def test_multi_tool_stream(ctx: TestContext) -> bool:
    _header("Test 4: 多工具并行（流式）")

    chunks, text, fr = chat_stream(
        ctx.base_url,
        [{"role": "user",
          "content": "Please tell me both: the weather in Beijing AND the current time in Shanghai. "
                     "Call the appropriate tools."}],
        model=ctx.model, api_key=ctx.api_key, tools=ALL_TOOLS,
    )
    tcs = merge_tool_calls_from_stream(chunks)
    _info(f"总 chunk 数: {len(chunks)} | finish_reason={fr}")
    _info(f"合并后 tool_calls={json.dumps(tcs, ensure_ascii=False)[:300]}")

    passed = True
    if fr != "tool_calls":
        _fail(f"finish_reason 应为 'tool_calls'，实际 {fr!r}")
        passed = False

    if not tcs:
        _fail("流式未产出 tool_calls")
        return False

    names = [tc.get("function", {}).get("name") for tc in tcs]
    _info(f"调用的工具: {names}")

    if len(tcs) >= 2:
        _ok(f"返回 {len(tcs)} 个工具调用（并行）")
    else:
        _warn(f"只返回 {len(tcs)} 个（模型合并了请求）")

    # 检查 index 递增是否正确（模拟 SDK 分组）
    indices = []
    for ch in chunks:
        for tc in ((ch.get("choices") or [{}])[0].get("delta") or {}).get("tool_calls") or []:
            indices.append(tc.get("index"))
    unique_indices = sorted(set(i for i in indices if i is not None))
    _info(f"出现过的 index: {unique_indices}")
    if len(unique_indices) == len(tcs):
        _ok(f"index 数量与工具数量一致（{len(tcs)}）")
    else:
        _warn(f"index 数量 {len(unique_indices)} != 工具数量 {len(tcs)}")

    ok, errors = validate_tool_calls(tcs, KNOWN_NAMES)
    if not ok:
        for e in errors:
            _fail(e)
        passed = False
    else:
        _ok("所有工具调用结构合法")

    return passed


def test_tool_chain_cache(ctx: TestContext) -> bool:
    _header("Test 5: 多轮工具链 + 缓存命中（非流式）")

    if ctx.delete_mode:
        _info("当前 delete_conversation=true，跳过缓存命中检查，仅验证多轮语义")

    # ── Turn 1：长 user 消息，触发工具调用 ──
    msgs_t1 = [{"role": "user", "content": LONG_USER}]
    t0 = time.time()
    r1 = chat_non_stream(ctx.base_url, msgs_t1, ctx.model, ctx.api_key,
                         tools=[WEATHER_TOOL])
    t1_elapsed = time.time() - t0

    u1 = extract_usage(r1)
    msg1 = extract_message(r1)
    tcs1 = msg1.get("tool_calls") or []

    _info(f"Turn 1 ({t1_elapsed:.2f}s) | prompt_tokens={u1.get('prompt_tokens')}")

    if not tcs1:
        _fail("Turn 1 未产生工具调用（模型可能未遵守协议）")
        _info(f"内容: {(msg1.get('content') or '')[:200]!r}")
        return False

    tc = tcs1[0]
    _ok(f"Turn 1 得到 {tc['function']['name']}")

    # ── Turn 2：回传 tool result ──
    msgs_t2 = [
        {"role": "user", "content": LONG_USER},
        {"role": "assistant", "content": None, "tool_calls": [
            {"id": tc["id"], "type": "function",
             "function": {"name": tc["function"]["name"],
                          "arguments": tc["function"]["arguments"]}}
        ]},
        {"role": "tool", "tool_call_id": tc["id"],
         "content": '{"temperature": 18, "condition": "cloudy", "unit": "celsius"}'},
    ]
    t0 = time.time()
    r2 = chat_non_stream(ctx.base_url, msgs_t2, ctx.model, ctx.api_key,
                         tools=[WEATHER_TOOL])
    t2_elapsed = time.time() - t0

    u2 = extract_usage(r2)
    msg2 = extract_message(r2)
    fr2 = extract_finish_reason(r2)
    content2 = msg2.get("content") or ""

    _info(f"Turn 2 ({t2_elapsed:.2f}s) | prompt_tokens={u2.get('prompt_tokens')} | finish_reason={fr2}")
    _info(f"Turn 2 内容: {content2[:200]!r}")

    passed = True
    if fr2 != "stop":
        _fail(f"Turn 2 finish_reason 应为 'stop'，实际 {fr2!r}")
        passed = False
    else:
        _ok("Turn 2 finish_reason = stop")

    if "18" in content2 or "cloudy" in content2 or "多云" in content2 or "阴" in content2:
        _ok("最终回答引用了 tool result（18/cloudy）")
    else:
        _warn("最终回答似乎未引用 tool result（可能是模型重新组织语言）")

    # ── 缓存命中检测：Turn 2 的 prompt 应该比 Turn 1 短得多 ──
    if not ctx.delete_mode and u1.get("prompt_tokens") and u2.get("prompt_tokens"):
        ratio = u2["prompt_tokens"] / u1["prompt_tokens"]
        _info(f"prompt_tokens 比 (T2/T1) = {ratio:.2f}")
        if ratio < 0.8:
            _ok(f"Turn 2 prompt_tokens 明显减小（{u1['prompt_tokens']} → {u2['prompt_tokens']}），缓存命中")
        else:
            _warn(f"Turn 2 prompt_tokens 与 Turn 1 相当（{u1['prompt_tokens']} → {u2['prompt_tokens']}），"
                  f"缓存可能未命中——请查看服务端日志确认")
            # 不视为 fail，因为有多种原因（lazy 转换等）

    return passed


def test_tool_chain_cache_stream(ctx: TestContext) -> bool:
    _header("Test 6: 多轮工具链 + 缓存命中（流式）")

    if ctx.delete_mode:
        _info("delete_conversation=true，仅验证流式多轮语义")

    # ── Turn 1 ──
    t0 = time.time()
    chunks1, text1, fr1 = chat_stream(
        ctx.base_url, [{"role": "user", "content": LONG_USER}],
        ctx.model, ctx.api_key, tools=[WEATHER_TOOL],
    )
    t1_elapsed = time.time() - t0
    tcs1 = merge_tool_calls_from_stream(chunks1)
    u1 = chat_stream.last_usage or {}

    _info(f"Turn 1 ({t1_elapsed:.2f}s) | chunks={len(chunks1)} | "
          f"prompt_tokens={u1.get('prompt_tokens')} | finish_reason={fr1}")

    if not tcs1:
        _fail("Turn 1 流式未产生工具调用")
        return False
    tc = tcs1[0]
    _ok(f"Turn 1 得到 {tc['function']['name']}")

    # ── Turn 2 ──
    msgs_t2 = [
        {"role": "user", "content": LONG_USER},
        {"role": "assistant", "content": None, "tool_calls": [
            {"id": tc["id"], "type": "function",
             "function": {"name": tc["function"]["name"],
                          "arguments": tc["function"]["arguments"]}}
        ]},
        {"role": "tool", "tool_call_id": tc["id"],
         "content": '{"temperature": 18, "condition": "cloudy"}'},
    ]
    t0 = time.time()
    chunks2, text2, fr2 = chat_stream(
        ctx.base_url, msgs_t2, ctx.model, ctx.api_key, tools=[WEATHER_TOOL],
    )
    t2_elapsed = time.time() - t0
    u2 = chat_stream.last_usage or {}

    _info(f"Turn 2 ({t2_elapsed:.2f}s) | chunks={len(chunks2)} | "
          f"prompt_tokens={u2.get('prompt_tokens')} | finish_reason={fr2}")
    _info(f"Turn 2 内容: {text2[:200]!r}")

    passed = True
    if fr2 != "stop":
        _fail(f"Turn 2 finish_reason 应为 'stop'，实际 {fr2!r}")
        passed = False
    else:
        _ok("Turn 2 finish_reason = stop")

    if text2:
        _ok(f"Turn 2 返回 {len(text2)} 字符内容")
    else:
        _fail("Turn 2 内容为空")
        passed = False

    if "18" in text2 or "cloudy" in text2 or "多云" in text2 or "阴" in text2:
        _ok("最终回答引用了 tool result")
    else:
        _warn("最终回答似乎未引用 tool result")

    # 缓存命中提示
    if not ctx.delete_mode and u1.get("prompt_tokens") and u2.get("prompt_tokens"):
        ratio = u2["prompt_tokens"] / u1["prompt_tokens"]
        _info(f"prompt_tokens 比 (T2/T1) = {ratio:.2f}")
        if ratio < 0.8:
            _ok("缓存命中（T2 的 prompt_tokens 明显减小）")
        else:
            _warn("缓存可能未命中（详见服务端日志）")

    return passed


def test_tool_choice_none(ctx: TestContext) -> bool:
    _header("Test 7: tool_choice='none' 时禁用工具")

    resp = chat_non_stream(
        ctx.base_url,
        [{"role": "user",
          "content": "What is the weather in Beijing? Just answer in one sentence."}],
        model=ctx.model, api_key=ctx.api_key,
        tools=[WEATHER_TOOL], tool_choice="none",
    )

    msg = extract_message(resp)
    tcs = msg.get("tool_calls") or []
    content = msg.get("content") or ""
    fr = extract_finish_reason(resp)

    _info(f"finish_reason={fr}")
    _info(f"内容: {content[:200]!r}")

    passed = True
    if tcs:
        _fail(f"tool_choice='none' 仍返回 tool_calls: {len(tcs)} 个")
        passed = False
    else:
        _ok("未返回 tool_calls（符合预期）")

    if fr == "tool_calls":
        _fail("finish_reason 不应为 'tool_calls'")
        passed = False
    else:
        _ok(f"finish_reason = {fr!r}")

    if content:
        _ok(f"返回文本内容 ({len(content)} 字符)")
    else:
        _warn("返回内容为空")

    return passed


def test_no_tools_regression(ctx: TestContext) -> bool:
    _header("Test 8: 无工具文本响应回归（非流式）")

    resp = chat_non_stream(
        ctx.base_url,
        [{"role": "user", "content": "用一句话介绍 Python。"}],
        model=ctx.model, api_key=ctx.api_key,
    )

    msg = extract_message(resp)
    content = msg.get("content") or ""
    tcs = msg.get("tool_calls")
    fr = extract_finish_reason(resp)

    _info(f"finish_reason={fr}")
    _info(f"内容: {content[:200]!r}")

    passed = True
    if fr != "stop":
        _fail(f"finish_reason 应为 'stop'，实际 {fr!r}")
        passed = False
    else:
        _ok("finish_reason = stop")

    if not content:
        _fail("内容为空")
        passed = False
    else:
        _ok(f"返回 {len(content)} 字符内容")

    if tcs:
        _fail("无工具请求返回了 tool_calls")
        passed = False
    else:
        _ok("未返回 tool_calls")

    # 不应该出现哨兵
    if "<|tool_call" in content:
        _fail("内容里泄漏了哨兵标记")
        passed = False
    else:
        _ok("未泄漏哨兵标记")

    return passed


def test_stream_no_tools_regression(ctx: TestContext) -> bool:
    _header("Test 9: 无工具文本响应回归（流式）")

    # 用一个有明确语义、模型不会返回空的 prompt
    prompt = "请用 3 句话介绍法国巴黎这座城市。"
    chunks, text, fr = chat_stream(
        ctx.base_url,
        [{"role": "user", "content": prompt}],
        model=ctx.model, api_key=ctx.api_key,
    )

    _info(f"chunks={len(chunks)} | finish_reason={fr}")
    _info(f"内容: {text[:200]!r}")

    passed = True
    if fr != "stop":
        _fail(f"finish_reason 应为 'stop'，实际 {fr!r}")
        passed = False
    else:
        _ok("finish_reason = stop")

    if not text:
        _fail("拼接内容为空")
        passed = False
    else:
        _ok(f"拼接内容 {len(text)} 字符")

    if "<|tool_call" in text:
        _fail("内容里泄漏了哨兵标记")
        passed = False
    else:
        _ok("未泄漏哨兵标记")

    tcs = merge_tool_calls_from_stream(chunks)
    if tcs:
        _fail(f"无工具请求流式返回了 tool_calls: {len(tcs)}")
        passed = False
    else:
        _ok("未返回 tool_calls delta")

    return passed

def test_parallel_tool_cache(ctx: TestContext) -> bool:
    _header("Test 10: 并行工具调用的缓存命中（已知限制检查）")

    if ctx.delete_mode:
        _info("delete_conversation=true，跳过")
        return True

    # Turn 1：触发多个 tool calls
    t0 = time.time()
    r1 = chat_non_stream(
        ctx.base_url,
        [{"role": "user",
          "content": LONG_USER + " 同时也调用 get_time 查询上海当前时间。"}],
        model=ctx.model, api_key=ctx.api_key, tools=ALL_TOOLS,
    )
    t1_elapsed = time.time() - t0
    u1 = extract_usage(r1)
    tcs1 = extract_message(r1).get("tool_calls") or []

    _info(f"Turn 1 ({t1_elapsed:.2f}s) | prompt_tokens={u1.get('prompt_tokens')} | "
          f"tool_calls={len(tcs1)}")

    if len(tcs1) < 2:
        _warn(f"Turn 1 只返回 {len(tcs1)} 个工具调用，无法测试并行缓存")
        _info("（模型行为，非协议问题——重跑可能触发）")
        return True

    # Turn 2：并行回传多个 tool result
    msgs_t2 = [{"role": "user", "content": LONG_USER + " 同时也调用 get_time 查询上海当前时间。"}]
    msgs_t2.append({
        "role": "assistant", "content": None,
        "tool_calls": [
            {"id": tc["id"], "type": "function",
             "function": {"name": tc["function"]["name"],
                          "arguments": tc["function"]["arguments"]}}
            for tc in tcs1
        ],
    })
    for tc in tcs1:
        name = tc["function"]["name"]
        if name == "get_weather":
            result = '{"temperature": 18, "condition": "cloudy"}'
        else:
            result = '{"time": "14:30", "timezone": "Asia/Shanghai"}'
        msgs_t2.append({
            "role": "tool", "tool_call_id": tc["id"], "content": result,
        })

    t0 = time.time()
    r2 = chat_non_stream(ctx.base_url, msgs_t2, ctx.model, ctx.api_key,
                         tools=ALL_TOOLS)
    t2_elapsed = time.time() - t0
    u2 = extract_usage(r2)
    content2 = extract_message(r2).get("content") or ""

    _info(f"Turn 2 ({t2_elapsed:.2f}s) | prompt_tokens={u2.get('prompt_tokens')}")
    _info(f"Turn 2 内容: {content2[:200]!r}")

    passed = True
    if u1.get("prompt_tokens") and u2.get("prompt_tokens"):
        ratio = u2["prompt_tokens"] / u1["prompt_tokens"]
        _info(f"prompt_tokens 比 (T2/T1) = {ratio:.2f}")
        if ratio < 0.8:
            _ok("并行工具调用后缓存命中")
        else:
            _warn("缓存未命中——这是 _split_messages 的已知限制："
                  "并行 tool result 之间无法切分前缀 hash，"
                  "每轮都会重新走 fresh 路径。功能正确但会浪费上游算力。")

    return passed


# ═══════════════════════════════════════════════════════════════
# Runner
# ═══════════════════════════════════════════════════════════════

def main() -> int:
    parser = argparse.ArgumentParser(description="deepseek2api 工具调用高级测试")
    parser.add_argument("--base-url", default="http://localhost:8000/v1")
    parser.add_argument("--api-key", default="")
    parser.add_argument("--model", default="deepseek-chat")
    parser.add_argument("--test", type=int, default=0,
                        help="只跑指定测试 (1-10)，0=全部")
    args = parser.parse_args()

    print(f"\n{_BOLD}deepseek2api 工具调用高级测试{_RESET}")
    print(f"  Base URL: {_CYAN}{args.base_url}{_RESET}")
    print(f"  Model:    {_CYAN}{args.model}{_RESET}")
    print(f"  API Key:  {_CYAN}{'(none)' if not args.api_key else '***'}{_RESET}")

    ctx = TestContext(args.base_url, args.model, args.api_key)
    ctx.load_health()

    if ctx.health:
        cfg = ctx.health.get("config") or {}
        cache = ctx.health.get("cache") or {}
        _info(f"服务端模式: delete_conversation={cfg.get('delete_conversation')} "
              f"| cache_enabled={cache.get('enabled')}")

    tests = [
        ("单工具非流式",            test_single_tool_non_stream),
        ("单工具流式",              test_single_tool_stream),
        ("多工具并行（非流式）",    test_multi_tool_non_stream),
        ("多工具并行（流式）",      test_multi_tool_stream),
        ("多轮工具链 + 缓存命中",   test_tool_chain_cache),
        ("多轮工具链 + 流式缓存",   test_tool_chain_cache_stream),
        ("tool_choice=none",        test_tool_choice_none),
        ("无工具回归（非流式）",    test_no_tools_regression),
        ("无工具回归（流式）",      test_stream_no_tools_regression),
        ("并行工具调用缓存",        test_parallel_tool_cache),
    ]

    results: list[tuple[str, bool]] = []
    for idx, (name, fn) in enumerate(tests, 1):
        if args.test != 0 and args.test != idx:
            continue
        try:
            passed = fn(ctx)
            results.append((name, passed))
        except Exception as exc:
            _fail(f"{name} 异常: {exc}")
            import traceback
            traceback.print_exc()
            results.append((name, False))

    _header("测试结果汇总")
    all_passed = True
    for name, passed in results:
        flag = f"{_GREEN}PASS{_RESET}" if passed else f"{_RED}FAIL{_RESET}"
        print(f"  [{flag}] {name}")
        if not passed:
            all_passed = False

    total = len(results)
    ok_count = sum(1 for _, p in results if p)
    color = _GREEN if all_passed else _RED
    print(f"\n  {color}{ok_count}/{total} 通过{_RESET}\n")

    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())