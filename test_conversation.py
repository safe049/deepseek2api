#!/usr/bin/env python3
"""
deepseek2api 完整功能测试器

覆盖:
  1. /health 端点 + 模式识别
  2. /v1/models 端点
  3. 单轮非流式 (内容完整性 + usage tokens)
  4. 流式 SSE 响应
  5. 多轮对话上下文保持
  6. 分支对话隔离
  7. 错误处理 (空 messages → 400)

用法:
  python test_conversation.py
  python test_conversation.py --base-url http://localhost:8000/v1 --api-key sk-xxx
  python test_conversation.py --test 3         # 只跑第 3 个测试
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import urllib.error
import urllib.request
from typing import Any


# ── ANSI colors ────────────────────────────────────────────────

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


def _info(msg: str) -> None:
    print(f"  {_CYAN}ℹ{_RESET} {msg}")


def _warn(msg: str) -> None:
    print(f"  {_YELLOW}⚠{_RESET} {msg}")


def _header(title: str) -> None:
    print(f"\n{_BOLD}{'─' * 66}{_RESET}")
    print(f"{_BOLD}  {title}{_RESET}")
    print(f"{_BOLD}{'─' * 66}{_RESET}")


# ── HTTP helpers ───────────────────────────────────────────────

def _http(
    url: str,
    method: str = "GET",
    payload: dict | None = None,
    api_key: str = "",
    timeout: int = 120,
):
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
    """GET /health 并解析。base_url 通常是 http://host:port/v1，需去掉 /v1。"""
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


def chat_non_stream(
    base_url: str,
    messages: list[dict],
    model: str = "deepseek-chat",
    api_key: str = "",
) -> dict:
    payload = {"model": model, "messages": messages, "stream": False}
    resp, err = _http(f"{base_url.rstrip('/')}/chat/completions", "POST", payload, api_key)
    if resp is None:
        code, raw = err
        raise RuntimeError(f"HTTP {code}: {raw[:500]}")
    return json.loads(resp.read().decode("utf-8"))


def chat_stream(
    base_url: str,
    messages: list[dict],
    model: str = "deepseek-chat",
    api_key: str = "",
) -> tuple[str, int]:
    """返回 (拼接后的文本, chunk 数量)"""
    payload = {"model": model, "messages": messages, "stream": True}
    resp, err = _http(f"{base_url.rstrip('/')}/chat/completions", "POST", payload, api_key)
    if resp is None:
        code, raw = err
        raise RuntimeError(f"HTTP {code}: {raw[:500]}")

    text = ""
    chunk_count = 0
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
        chunk_count += 1
        choices = chunk.get("choices") or []
        if choices:
            delta = choices[0].get("delta") or {}
            content = delta.get("content", "")
            if content:
                text += content
    return text, chunk_count


def extract_content(resp: dict) -> str:
    choices = resp.get("choices") or []
    if not choices:
        return ""
    return (choices[0].get("message") or {}).get("content", "")


def extract_usage(resp: dict) -> dict:
    return resp.get("usage") or {}


# ── Tests ──────────────────────────────────────────────────────

class TestContext:
    """保存运行期间的模式信息，供后续测试判断。"""
    def __init__(self, base_url: str, model: str, api_key: str):
        self.base_url = base_url
        self.model = model
        self.api_key = api_key
        self.health: dict = {}
        self.delete_mode: bool = True   # 默认假设
        self.cache_enabled: bool = False

    def load_health(self) -> None:
        self.health = get_health(self.base_url, self.api_key)
        cfg = self.health.get("config") or {}
        self.delete_mode = bool(cfg.get("delete_conversation", True))
        self.cache_enabled = bool((self.health.get("cache") or {}).get("enabled", False))


def test_health(ctx: TestContext) -> bool:
    """Test 1: /health 端点 + 模式识别。"""
    _header("Test 1: /health 端点 + 模式识别")

    if not ctx.health:
        _fail("无法访问 /health 端点")
        return False

    _info(f"响应: {json.dumps(ctx.health, ensure_ascii=False)}")

    if ctx.health.get("status") != "ok":
        _fail(f"status 不是 ok: {ctx.health.get('status')}")
        return False
    _ok("status = ok")

    cfg = ctx.health.get("config") or {}
    _ok(f"delete_conversation = {cfg.get('delete_conversation')}")
    _ok(f"default_model      = {cfg.get('default_model')}")
    _ok(f"auth_required      = {cfg.get('auth_required')}")

    cache = ctx.health.get("cache") or {}
    if ctx.delete_mode:
        if cache.get("enabled") is False:
            _ok("缓存已禁用（与 delete_conversation=true 互斥，符合预期）")
        else:
            _warn(f"delete_conversation=true 但缓存显示为启用: {cache}")
    else:
        if cache.get("enabled") is True:
            _ok(f"缓存已启用: {cache}")
        else:
            _warn(f"delete_conversation=false 但缓存未启用: {cache}")

    return True


def test_models(ctx: TestContext) -> bool:
    """Test 2: /v1/models 端点。"""
    _header("Test 2: /v1/models 端点")

    resp, err = _http(f"{ctx.base_url.rstrip('/')}/models", api_key=ctx.api_key)
    if resp is None:
        _fail(f"请求失败: {err}")
        return False

    data = json.loads(resp.read().decode("utf-8"))
    _info(f"响应: {json.dumps(data, ensure_ascii=False)[:300]}")

    if data.get("object") != "list":
        _fail("object != list")
        return False

    ids = [m.get("id") for m in data.get("data", [])]
    if not ids:
        _fail("模型列表为空")
        return False

    _ok(f"返回 {len(ids)} 个模型: {ids}")
    return True


def test_single_turn_content(ctx: TestContext) -> bool:
    """Test 3: 单轮对话 — 验证内容完整性。"""
    _header("Test 3: 单轮对话（内容完整性）")

    prompt = "请写一句包含至少 15 个字的中文句子，并以句号结尾。"
    t0 = time.time()
    resp = chat_non_stream(
        ctx.base_url,
        [{"role": "user", "content": prompt}],
        model=ctx.model,
        api_key=ctx.api_key,
    )
    elapsed = time.time() - t0

    content = extract_content(resp)
    _info(f"耗时: {elapsed:.2f}s")
    _info(f"内容: {content!r}")

    passed = True
    if not content:
        _fail("响应内容为空")
        passed = False
    elif len(content) < 10:
        _fail(f"内容疑似被截断，长度仅 {len(content)}")
        passed = False
    else:
        _ok(f"内容长度 {len(content)} 字符")

    # 内容应包含中文标点或英文标点，提示未被截断
    if not any(p in content for p in "。！？.!?"):
        _warn("内容中没有标点符号，可能被截断")
    else:
        _ok("内容包含终止标点，拼接完整")

    return passed


def test_usage_tokens(ctx: TestContext) -> bool:
    """Test 4: usage.completion_tokens > 0。"""
    _header("Test 4: usage token 统计")

    resp = chat_non_stream(
        ctx.base_url,
        [{"role": "user", "content": "请回复：测试 token 统计。不要展开。"}],
        model=ctx.model,
        api_key=ctx.api_key,
    )
    usage = extract_usage(resp)
    content = extract_content(resp)
    _info(f"内容: {content!r}")
    _info(f"usage: {usage}")

    if not isinstance(usage, dict):
        _fail("usage 字段缺失或格式错误")
        return False

    ct = usage.get("completion_tokens", 0)
    if ct <= 0:
        _fail(f"completion_tokens = {ct}（应 > 0）")
        return False

    _ok(f"completion_tokens = {ct}")
    return True


def test_stream_mode(ctx: TestContext) -> bool:
    """Test 5: 流式 SSE 响应。"""
    _header("Test 5: 流式 SSE 响应")

    prompt = "请用一句话介绍你自己。"
    t0 = time.time()
    text, chunks = chat_stream(
        ctx.base_url,
        [{"role": "user", "content": prompt}],
        model=ctx.model,
        api_key=ctx.api_key,
    )
    elapsed = time.time() - t0

    _info(f"耗时: {elapsed:.2f}s | chunk 数: {chunks}")
    _info(f"拼接文本: {text!r}")

    passed = True
    if chunks <= 0:
        _fail("未收到任何 chunk")
        passed = False
    else:
        _ok(f"收到 {chunks} 个 chunk")

    if not text:
        _fail("拼接后的文本为空")
        passed = False
    else:
        _ok(f"拼接后长度 {len(text)} 字符")

    return passed


def test_multi_turn_context(ctx: TestContext) -> bool:
    """Test 6: 多轮对话 — 上下文保持。"""
    _header("Test 6: 多轮对话（上下文保持）")

    if ctx.delete_mode:
        _info("当前模式: delete_conversation=true（每次请求独立，靠 prompt 拼接历史）")
    else:
        _info("当前模式: delete_conversation=false（session 缓存复用）")

    # 第 1 轮：建立上下文
    msgs = [
        {"role": "system", "content": "你是一个数学助手。回答要简洁。"},
        {"role": "user", "content": "请记住这个数字：42"},
    ]

    t0 = time.time()
    resp1 = chat_non_stream(ctx.base_url, msgs, ctx.model, ctx.api_key)
    reply1 = extract_content(resp1)
    t1 = time.time() - t0
    _info(f"Turn 1 ({t1:.2f}s): {reply1!r}")

    if not reply1:
        _fail("Turn 1 返回空")
        return False

    # 第 2 轮：引用上下文
    msgs.append({"role": "assistant", "content": reply1})
    msgs.append({"role": "user", "content": "我刚才让你记住的数字是多少？只回答数字。"})

    t0 = time.time()
    resp2 = chat_non_stream(ctx.base_url, msgs, ctx.model, ctx.api_key)
    reply2 = extract_content(resp2)
    t2 = time.time() - t0
    _info(f"Turn 2 ({t2:.2f}s): {reply2!r}")

    # 第 3 轮：基于上下文计算
    msgs.append({"role": "assistant", "content": reply2})
    msgs.append({"role": "user", "content": "把这个数字乘以 2，只回答结果。"})

    t0 = time.time()
    resp3 = chat_non_stream(ctx.base_url, msgs, ctx.model, ctx.api_key)
    reply3 = extract_content(resp3)
    t3 = time.time() - t0
    _info(f"Turn 3 ({t3:.2f}s): {reply3!r}")

    passed = True
    if "42" in reply2:
        _ok("Turn 2 正确记住 42")
    else:
        _fail(f"Turn 2 未记住 42: {reply2!r}")
        passed = False

    if "84" in reply3:
        _ok("Turn 3 正确计算 42×2=84")
    else:
        _fail(f"Turn 3 未正确计算: {reply3!r}")
        passed = False

    return passed

def test_branch_isolation(ctx: TestContext) -> bool:
    """Test 7: 分支对话 — 不同前缀互不干扰。"""
    _header("Test 7: 分支对话隔离")

    msgs_a = [
        {"role": "user", "content": "我的名字是 Alice。"},
        {"role": "assistant", "content": "你好 Alice！"},
        {"role": "user", "content": "根据上面的对话，我叫什么名字？只回答名字。"},
    ]
    msgs_b = [
        {"role": "user", "content": "我的名字是 Bob。"},
        {"role": "assistant", "content": "你好 Bob！"},
        {"role": "user", "content": "根据上面的对话，我叫什么名字？只回答名字。"},
    ]

    resp_a = chat_non_stream(ctx.base_url, msgs_a, ctx.model, ctx.api_key)
    resp_b = chat_non_stream(ctx.base_url, msgs_b, ctx.model, ctx.api_key)

    reply_a = extract_content(resp_a)
    reply_b = extract_content(resp_b)
    _info(f"Branch A: {reply_a!r}")
    _info(f"Branch B: {reply_b!r}")

    passed = True
    if "Alice" in reply_a:
        _ok("Branch A 识别 Alice")
    else:
        _fail(f"Branch A 未识别 Alice: {reply_a!r}")
        passed = False

    if "Bob" in reply_b:
        _ok("Branch B 识别 Bob")
    else:
        _fail(f"Branch B 未识别 Bob: {reply_b!r}")
        passed = False

    # 用 "Bob 出现在 A 的回复里 / Alice 出现在 B 的回复里" 判断串扰
    if "Bob" not in reply_a and "Alice" not in reply_b:
        _ok("两分支互不串扰")
    else:
        _fail("分支间存在串扰")
        passed = False

    return passed


def test_error_handling(ctx: TestContext) -> bool:
    """Test 8: 错误处理 — 空 messages 应返回 400。"""
    _header("Test 8: 错误处理（空 messages）")

    payload = {"model": ctx.model, "messages": [], "stream": False}
    resp, err = _http(f"{ctx.base_url.rstrip('/')}/chat/completions", "POST", payload, ctx.api_key)

    if resp is not None:
        _fail("空 messages 却返回了成功状态")
        return False

    code, raw = err
    _info(f"HTTP {code}")
    _info(f"body: {raw[:200]}")

    if code != 400:
        _fail(f"期望 400，实际 {code}")
        return False

    try:
        body = json.loads(raw)
    except json.JSONDecodeError:
        _fail("错误响应不是合法 JSON")
        return False

    if "error" not in body:
        _fail("错误响应缺少 error 字段")
        return False

    _ok(f"正确返回 400: {body['error'].get('message', '')[:80]}")
    return True


def test_cache_behavior(ctx: TestContext) -> bool:
    """Test 9: 缓存行为 — 根据当前模式校验缓存统计。"""
    _header("Test 9: 缓存行为验证")

    health = get_health(ctx.base_url, ctx.api_key)
    cache = health.get("cache") or {}
    _info(f"cache 状态: {json.dumps(cache, ensure_ascii=False)}")

    if ctx.delete_mode:
        if cache.get("enabled") is False:
            _ok("delete_conversation=true → 缓存已禁用")
            return True
        _warn("delete_conversation=true 但缓存未禁用")
        return True  # 不视为失败，只是警告

    # 缓存启用模式：total > 0 说明有过请求
    total = cache.get("total", 0)
    if total > 0:
        _ok(f"缓存已存储 {total} 条记录（active={cache.get('active', 0)}）")
    else:
        _warn("缓存启用但 total=0（可能是初始状态）")
    return True


# ── Main ───────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="deepseek2api 完整功能测试")
    parser.add_argument("--base-url", default="http://localhost:8000/v1", help="API base URL")
    parser.add_argument("--api-key", default="", help="API key（若启用 SERVER_API_KEYS）")
    parser.add_argument("--model", default="deepseek-chat", help="模型名")
    parser.add_argument("--test", type=int, default=0, help="只跑指定测试 (1-9), 0=全部")
    args = parser.parse_args()

    print(f"\n{_BOLD}deepseek2api 完整功能测试{_RESET}")
    print(f"  Base URL: {_CYAN}{args.base_url}{_RESET}")
    print(f"  Model:    {_CYAN}{args.model}{_RESET}")
    print(f"  API Key:  {_CYAN}{'(none)' if not args.api_key else '***'}{_RESET}")

    ctx = TestContext(args.base_url, args.model, args.api_key)
    ctx.load_health()

    tests = [
        ("/health 端点",      test_health),
        ("/v1/models 端点",   test_models),
        ("单轮对话内容",      test_single_turn_content),
        ("usage token 统计",  test_usage_tokens),
        ("流式 SSE 响应",     test_stream_mode),
        ("多轮对话上下文",    test_multi_turn_context),
        ("分支对话隔离",      test_branch_isolation),
        ("错误处理",          test_error_handling),
        ("缓存行为验证",      test_cache_behavior),
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

    # 汇总
    _header("测试结果汇总")
    all_passed = True
    for name, passed in results:
        flag = f"{_GREEN}PASS{_RESET}" if passed else f"{_RED}FAIL{_RESET}"
        print(f"  [{flag}] {name}")
        if not passed:
            all_passed = False

    total = len(results)
    ok = sum(1 for _, p in results if p)
    color = _GREEN if all_passed else _RED
    print(f"\n  {color}{ok}/{total} 通过{_RESET}\n")

    sys.exit(0 if all_passed else 1)


if __name__ == "__main__":
    main()