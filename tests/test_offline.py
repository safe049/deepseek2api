#!/usr/bin/env python3
"""离线回归测试：不碰上游、不需要凭证。

覆盖：深度思考决策矩阵、SSE 事件 -> OpenAI chunk 转换、工具调用解析、cache key。
用法：python test_offline.py  （或 python -m pytest test_offline.py -q）
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "src"))

from deepseek2api.config import AppConfig
from deepseek2api.services.session_cache import prefix_hash
from deepseek2api.services.tool_calling import parse_tool_calls
from deepseek2api.services.translator import (
    StreamState,
    _apply_extra_body,
    _finalize_mode,
    _resolve_model_and_type,
    convert_deepseek_to_openai_chunk,
)

_TOOLS = [{
    "type": "function",
    "function": {
        "name": "get_weather",
        "description": "weather",
        "parameters": {"type": "object", "properties": {"city": {"type": "string"}}},
    },
}]


def _cfg(**overrides) -> AppConfig:
    base = dict(
        env_file_path=Path(".env"), env_file_created=False,
        host="127.0.0.1", port=8000, api_prefix="/v1", log_level="INFO",
        debug_dump_all=False, request_timeout=180, server_api_keys=[],
        auth_token="t", device_id="d", settings_token="s", cookie="c",
        base_url="https://chat.deepseek.com", wasm_path="w",
        default_model="deepseek-chat", delete_conversation=False,
        max_concurrency=1, queue_wait_timeout=60, admin_password="a",
    )
    base.update(overrides)
    return AppConfig(**base)


def _decide(model, cfg=None, extra=None):
    cfg = cfg or _cfg()
    payload = {"model": model}
    if extra is not None:
        payload["extra_body"] = extra
    _, mt, th = _resolve_model_and_type(payload, cfg)
    _, th = _apply_extra_body(payload, th, cfg)
    return _finalize_mode(mt, th)


# -- 1. thinking decision matrix ---------------------------------

def test_thinking_matrix():
    assert _decide("deepseek-chat") == ("default", False)
    assert _decide("deepseek-reasoner") == ("expert", True)
    assert _decide("deepseek-r1") == ("expert", True)
    assert _decide("deepseek-thinking") == ("expert", True)
    assert _decide("deepseek-chat", extra={"thinking_enabled": True}) == ("expert", True)
    assert _decide("deepseek-chat", extra={"thinking_enabled": False}) == ("default", False)
    assert _decide("deepseek-chat", _cfg(search_enabled=True)) == ("default", False)


# -- 2. SSE -> OpenAI chunk --------------------------------------

def test_sse_ready_and_content():
    st = StreamState()
    chunk = convert_deepseek_to_openai_chunk(
        {"_event_type": "ready", "request_message_id": 1, "response_message_id": 2},
        "deepseek-chat", st,
    )
    assert chunk["choices"][0]["delta"]["role"] == "assistant"
    assert st.response_message_id == 2
    chunk = convert_deepseek_to_openai_chunk(
        {"p": "response/content", "o": "APPEND", "v": "hello"}, "deepseek-chat", st,
    )
    assert chunk["choices"][0]["delta"]["content"] == "hello"


def test_sse_thinking_fragment():
    st = StreamState()
    event = {"v": {"response": {
        "fragments": [{"id": 1, "type": "THINKING", "content": "let me think"}],
        "status": "WIP",
    }}}
    chunk = convert_deepseek_to_openai_chunk(event, "deepseek-thinking", st)
    assert chunk["choices"][0]["delta"]["reasoning_content"] == "let me think"
    assert "content" not in chunk["choices"][0]["delta"]
    assert convert_deepseek_to_openai_chunk(event, "deepseek-thinking", st) is None
    chunk = convert_deepseek_to_openai_chunk(
        {"p": "response/content", "o": "APPEND", "v": "more"}, "deepseek-thinking", st,
    )
    assert chunk["choices"][0]["delta"]["reasoning_content"] == "more"


def test_sse_real_upstream_think_to_response_switch():
    """复刻真实上游事件序列：THINK 片段 → response/fragments APPEND RESPONSE → 裸 v 增量。

    关键点：上游 fragment type 是 "THINK"（不是 "THINKING"），
    且 THINK→RESPONSE 的切换信号来自 {"p":"response/fragments","o":"APPEND"}，
    漏掉任一处都会让整段回答混进 reasoning_content。
    """
    st = StreamState()
    chunks = []

    def feed(ev):
        c = convert_deepseek_to_openai_chunk(ev, "deepseek-reasoner", st)
        if c:
            chunks.append(c)
        chunks.extend(st.pop_pending_chunks())

    feed({"_event_type": "ready", "request_message_id": 1, "response_message_id": 2})
    feed({"v": {"response": {"status": "WIP", "fragments": [
        {"id": 2, "type": "THINK", "content": "嗯"},
    ]}}})
    feed({"p": "response/fragments/-1/content", "o": "APPEND", "v": "，想"})
    feed({"p": "response/fragments", "o": "APPEND", "v": [
        {"id": 3, "type": "RESPONSE", "content": "你好"},
    ]})
    feed({"v": "！"})

    reasoning = "".join(
        c["choices"][0]["delta"].get("reasoning_content") or "" for c in chunks
    )
    content = "".join(
        c["choices"][0]["delta"].get("content") or "" for c in chunks
    )
    assert reasoning == "嗯，想", reasoning
    assert content == "你好！", content


def test_sse_multi_fragment_event_not_dropped():
    """一个事件里带多个片段时，不能只发出第一个。"""
    st = StreamState()
    chunk = convert_deepseek_to_openai_chunk({"v": {"response": {
        "status": "WIP",
        "fragments": [
            {"id": 1, "type": "THINK", "content": "a"},
            {"id": 2, "type": "THINK", "content": "b"},
            {"id": 3, "type": "RESPONSE", "content": "c"},
        ],
    }}}, "m", st)
    chunks = [chunk] + st.pop_pending_chunks()
    joined = "".join(
        c["choices"][0]["delta"].get("reasoning_content") or
        c["choices"][0]["delta"].get("content") or ""
        for c in chunks
    )
    assert joined == "abc", joined
    # 只有最后一个片段承载 finish_reason（此处 WIP，所以都没有）
    assert all(c["choices"][0]["finish_reason"] is None for c in chunks)


def test_sse_finish_once():
    st = StreamState()
    convert_deepseek_to_openai_chunk({"_event_type": "ready"}, "m", st)
    first = convert_deepseek_to_openai_chunk(
        {"p": "response/status", "o": "SET", "v": "FINISHED"}, "m", st,
    )
    assert first["choices"][0]["finish_reason"] == "stop"
    assert convert_deepseek_to_openai_chunk(
        {"p": "response/status", "o": "SET", "v": "FINISHED"}, "m", st,
    ) is None


# -- 3. tool call parsing ---------------------------------------

def test_dsml_parse():
    D = "\uff5c\uff5cDSML\uff5c\uff5c"
    text = (
        "<{D} calls>\n"
        "<{D} invoke name=\"get_weather\">\n"
        "<{D} parameter name=\"city\" string=\"true\">Beijing</{D} parameter>\n"
        "</{D} invoke>\n"
        "</{D} calls>"
    ).format(D=D)
    res = parse_tool_calls(text, tools=_TOOLS)
    assert len(res.calls) == 1
    assert res.calls[0].name == "get_weather"
    assert "Beijing" in res.calls[0].arguments
    assert res.clean_text == ""


def test_sentinel_parse():
    text = '<|tool_calls|>\n{"name": "bash", "arguments": {"command": "ls"}}\n<|/tool_calls|>'
    res = parse_tool_calls(text)
    assert len(res.calls) == 1
    assert res.calls[0].name == "bash"


def test_require_marker_ignores_bare_json():
    text = '{"name": "bash", "arguments": {"command": "ls"}}'
    assert parse_tool_calls(text, require_marker=True).calls == []
    assert len(parse_tool_calls(text).calls) == 1


# -- 3b. DSML prompt + schema coercion --------------------------

_COERCE_TOOLS = [{
    "type": "function",
    "function": {
        "name": "bash",
        "parameters": {
            "type": "object",
            "properties": {
                "command": {"type": "string"},
                "timeout": {"type": "number"},
                "nested": {"type": "object"},
            },
        },
    },
}]


def test_prompt_has_string_semantics_and_no_escaped_noise():
    from deepseek2api.services.tool_calling import build_tool_prompt
    p = build_tool_prompt(_COERCE_TOOLS)
    assert "string attribute semantics" in p
    assert "Multiline values" in p
    assert '\\\\"' not in p, "escaped quote leaked into prompt"
    # 多行示例里参数值必须是真实换行，不是字面量反斜杠n
    assert "cd /tmp\nls -la" in p


def test_schema_coerces_mislabeled_string_true():
    import json
    D = "\uff5c\uff5cDSML\uff5c\uff5c"
    text = (
        f"<{D} calls>"
        f'<{D} invoke name="bash">'
        f'<{D} parameter name="command" string="true">ls</{D} parameter>'
        f'<{D} parameter name="timeout" string="true">30</{D} parameter>'
        f'<{D} parameter name="nested" string="true">{{"a": 1}}</{D} parameter>'
        f"</{D} invoke></{D} calls>"
    )
    res = parse_tool_calls(text, tools=_COERCE_TOOLS)
    args = json.loads(res.calls[0].arguments)
    assert args == {"command": "ls", "timeout": 30, "nested": {"a": 1}}


def test_schema_keeps_declared_strings_as_strings():
    import json
    D = "\uff5c\uff5cDSML\uff5c\uff5c"
    text = (
        f"<{D} calls>"
        f'<{D} invoke name="bash">'
        f'<{D} parameter name="command" string="true">30</{D} parameter>'
        f"</{D} invoke></{D} calls>"
    )
    res = parse_tool_calls(text, tools=_COERCE_TOOLS)
    assert json.loads(res.calls[0].arguments)["command"] == "30"


def test_streaming_detector_coerces_same_as_non_stream():
    import json
    from deepseek2api.services.tool_calling import StreamingToolDetector
    D = "\uff5c\uff5cDSML\uff5c\uff5c"
    text = (
        f"<{D} calls>"
        f'<{D} invoke name="bash">'
        f'<{D} parameter name="command" string="true">ls</{D} parameter>'
        f'<{D} parameter name="timeout" string="true">30</{D} parameter>'
        f"</{D} invoke></{D} calls>"
    )
    det = StreamingToolDetector(tools=_COERCE_TOOLS, require_marker=False)
    acts = []
    for ch in text:
        acts.extend(det.feed(ch))
    acts.extend(det.finalize())
    calls = [a for a in acts if a.kind == "tool_calls"]
    assert calls, "streaming detector produced no tool calls"
    args = json.loads(calls[0].tool_calls[0].arguments)
    assert args["timeout"] == 30 and args["command"] == "ls"


# -- 4. cache key -----------------------------------------------

def test_prefix_hash_isolation():
    a = [{"role": "user", "content": "hi"}]
    b = [{"role": "user", "content": "hi "}]
    assert prefix_hash(a, "deepseek-chat") != prefix_hash(b, "deepseek-chat")
    assert prefix_hash(a, "deepseek-chat") != prefix_hash(a, "deepseek-thinking")
    assert prefix_hash(a, "deepseek-chat") == prefix_hash(a, "deepseek-chat")


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    failed = 0
    for fn in tests:
        try:
            fn()
            print(f"  PASS  {fn.__name__}")
        except AssertionError as exc:
            failed += 1
            print(f"  FAIL  {fn.__name__}: {exc}")
    print(f"\n{len(tests) - failed}/{len(tests)} passed")
    raise SystemExit(1 if failed else 0)
