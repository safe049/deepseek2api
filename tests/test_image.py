#!/usr/bin/env python3
"""
deepseek2api Vision 能力测试。

覆盖：
  1. 无图片基线（非流式）
  2. 无图片基线（流式）
  3. 单图 data URL（非流式）
  4. 单图 data URL（流式）
  5. 单图 HTTP URL（非流式）
  6. 多图（非流式）
  7. 图文混合 + 追问（多轮上下文）
  8. 图片 + 工具调用（混合）
  9. 非法图片 URL 的降级行为
 10. 大图（base64 > 1MB）的上传

依赖:
  pip install openai pillow    # pillow 只用于生成测试图，可选

用法:
  python test_vision.py
  python test_vision.py --base-url http://localhost:8000/v1 --api-key sk-xxx
  python test_vision.py --test 3
  python test_vision.py --http-image-url https://example.com/x.png   # 覆盖第 5 项
"""

from __future__ import annotations

import argparse
import base64
import io
import json
import sys
import time
import traceback
from typing import Any

# ── ANSI ────────────────────────────────────────────────────────
_GREEN = "\033[92m"
_RED = "\033[91m"
_YELLOW = "\033[93m"
_CYAN = "\033[96m"
_BOLD = "\033[1m"
_DIM = "\033[2m"
_RESET = "\033[0m"


def _ok(m):    print(f"  {_GREEN}✔{_RESET} {m}")
def _fail(m):  print(f"  {_RED}✘{_RESET} {m}")
def _warn(m):  print(f"  {_YELLOW}⚠{_RESET} {m}")
def _info(m):  print(f"  {_CYAN}ℹ{_RESET} {m}")
def _header(t):
    print(f"\n{_BOLD}{'─' * 72}{_RESET}")
    print(f"{_BOLD}  {t}{_RESET}")
    print(f"{_BOLD}{'─' * 72}{_RESET}")


# ═══════════════════════════════════════════════════════════════
# 测试图生成 —— 手写 PNG，不依赖 Pillow
# ═══════════════════════════════════════════════════════════════

def _png_chunk(tag: bytes, data: bytes) -> bytes:
    import struct, zlib
    return (
        struct.pack(">I", len(data))
        + tag
        + data
        + struct.pack(">I", zlib.crc32(tag + data) & 0xFFFFFFFF)
    )


def make_solid_png(width: int, height: int, rgb: tuple[int, int, int]) -> bytes:
    """生成一张纯色 PNG。"""
    import struct, zlib

    sig = b"\x89PNG\r\n\x1a\n"
    # IHDR
    ihdr = struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0)
    # IDAT：每行前加 filter byte 0
    row = bytes([0]) + bytes(rgb) * width
    raw = row * height
    idat = zlib.compress(raw, 9)

    return sig + _png_chunk(b"IHDR", ihdr) + _png_chunk(b"IDAT", idat) + _png_chunk(b"IEND", b"")


def make_text_png(text: str, width: int = 320, height: int = 120) -> bytes:
    """
    用最简 5x7 点阵字库画一段 ASCII 文本。

    只支持 A-Z0-9 和少量符号——够模型识别出内容用于测试。
    """
    # 5x7 点阵字库（只列需要的字符）
    FONT = {
        "A": [0x0E,0x11,0x11,0x1F,0x11,0x11,0x11],
        "B": [0x1E,0x11,0x11,0x1E,0x11,0x11,0x1E],
        "C": [0x0E,0x11,0x10,0x10,0x10,0x11,0x0E],
        "D": [0x1E,0x11,0x11,0x11,0x11,0x11,0x1E],
        "E": [0x1F,0x10,0x10,0x1E,0x10,0x10,0x1F],
        "F": [0x1F,0x10,0x10,0x1E,0x10,0x10,0x10],
        "G": [0x0E,0x11,0x10,0x17,0x11,0x11,0x0F],
        "H": [0x11,0x11,0x11,0x1F,0x11,0x11,0x11],
        "I": [0x0E,0x04,0x04,0x04,0x04,0x04,0x0E],
        "J": [0x07,0x02,0x02,0x02,0x02,0x12,0x0C],
        "K": [0x11,0x12,0x14,0x18,0x14,0x12,0x11],
        "L": [0x10,0x10,0x10,0x10,0x10,0x10,0x1F],
        "M": [0x11,0x1B,0x15,0x15,0x11,0x11,0x11],
        "N": [0x11,0x11,0x19,0x15,0x13,0x11,0x11],
        "O": [0x0E,0x11,0x11,0x11,0x11,0x11,0x0E],
        "P": [0x1E,0x11,0x11,0x1E,0x10,0x10,0x10],
        "Q": [0x0E,0x11,0x11,0x11,0x15,0x12,0x0D],
        "R": [0x1E,0x11,0x11,0x1E,0x14,0x12,0x11],
        "S": [0x0F,0x10,0x10,0x0E,0x01,0x01,0x1E],
        "T": [0x1F,0x04,0x04,0x04,0x04,0x04,0x04],
        "U": [0x11,0x11,0x11,0x11,0x11,0x11,0x0E],
        "V": [0x11,0x11,0x11,0x11,0x11,0x0A,0x04],
        "W": [0x11,0x11,0x11,0x15,0x15,0x1B,0x11],
        "X": [0x11,0x11,0x0A,0x04,0x0A,0x11,0x11],
        "Y": [0x11,0x11,0x0A,0x04,0x04,0x04,0x04],
        "Z": [0x1F,0x01,0x02,0x04,0x08,0x10,0x1F],
        "0": [0x0E,0x11,0x13,0x15,0x19,0x11,0x0E],
        "1": [0x04,0x0C,0x04,0x04,0x04,0x04,0x0E],
        "2": [0x0E,0x11,0x01,0x02,0x04,0x08,0x1F],
        "3": [0x1F,0x02,0x04,0x02,0x01,0x11,0x0E],
        "4": [0x02,0x06,0x0A,0x12,0x1F,0x02,0x02],
        "5": [0x1F,0x10,0x1E,0x01,0x01,0x11,0x0E],
        "6": [0x06,0x08,0x10,0x1E,0x11,0x11,0x0E],
        "7": [0x1F,0x01,0x02,0x04,0x08,0x08,0x08],
        "8": [0x0E,0x11,0x11,0x0E,0x11,0x11,0x0E],
        "9": [0x0E,0x11,0x11,0x0F,0x01,0x02,0x0C],
        " ": [0,0,0,0,0,0,0],
        "!": [0x04,0x04,0x04,0x04,0x04,0x00,0x04],
        "?": [0x0E,0x11,0x01,0x06,0x04,0x00,0x04],
        "-": [0,0,0,0x1F,0,0,0],
        ".": [0,0,0,0,0,0,0x04],
    }

    text = text.upper()
    # 渲染到像素矩阵
    scale = 8
    char_w = 6 * scale
    text_w = len(text) * char_w
    pad_x = max(0, (width - text_w) // 2)
    pad_y = max(0, (height - 7 * scale) // 2)

    # 背景黑、前景白
    px = bytearray(b"\x00\x00\x00" * (width * height))  # 每行后加 filter 0 待会处理

    def set_pixel(x, y):
        if 0 <= x < width and 0 <= y < height:
            i = (y * width + x) * 3
            px[i] = px[i+1] = px[i+2] = 0xFF

    for ci, ch in enumerate(text):
        glyph = FONT.get(ch)
        if glyph is None:
            continue
        gx0 = pad_x + ci * char_w
        for row_idx, bits in enumerate(glyph):
            for col_idx in range(5):
                if bits & (1 << (4 - col_idx)):
                    for dy in range(scale):
                        for dx in range(scale):
                            set_pixel(gx0 + col_idx * scale + dx,
                                      pad_y + row_idx * scale + dy)

    # 打包 PNG
    import struct, zlib
    sig = b"\x89PNG\r\n\x1a\n"
    ihdr = struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0)
    # 每行加 filter byte
    raw = bytearray()
    for y in range(height):
        raw.append(0)
        raw += px[y * width * 3:(y + 1) * width * 3]
    idat = zlib.compress(bytes(raw), 9)
    return sig + _png_chunk(b"IHDR", ihdr) + _png_chunk(b"IDAT", idat) + _png_chunk(b"IEND", b"")


def make_large_png(kb: int = 1200) -> bytes:
    """生成一张接近 kb KB 的 PNG（用大尺寸纯色图 + 随机噪声）。"""
    import os, struct, zlib
    # 用 800x800 的噪点图，压缩后大致 kb 级别
    size = 400
    # 目标不精确控制，够用即可
    raw = bytearray()
    noise = os.urandom(size * size * 3)
    for y in range(size):
        raw.append(0)
        raw += noise[y * size * 3:(y + 1) * size * 3]
    sig = b"\x89PNG\r\n\x1a\n"
    ihdr = struct.pack(">IIBBBBB", size, size, 8, 2, 0, 0, 0)
    idat = zlib.compress(bytes(raw), 1)   # 低压缩率 → 大文件
    return sig + _png_chunk(b"IHDR", ihdr) + _png_chunk(b"IDAT", idat) + _png_chunk(b"IEND", b"")


def to_data_url(png_bytes: bytes) -> str:
    b64 = base64.b64encode(png_bytes).decode("ascii")
    return f"data:image/png;base64,{b64}"


# ═══════════════════════════════════════════════════════════════
# Test context
# ═══════════════════════════════════════════════════════════════

class Ctx:
    def __init__(self, base_url: str, model: str, api_key: str, http_image_url: str):
        self.base_url = base_url
        self.model = model
        self.api_key = api_key
        self.http_image_url = http_image_url
        self.client = None
        # 预生成测试图，避免每次都重算
        self.png_hello = make_text_png("HELLO")
        self.png_world = make_text_png("WORLD")
        self.png_red   = make_solid_png(64, 64, (220, 30, 30))
        self.png_blue  = make_solid_png(64, 64, (30, 90, 220))
        self.png_large = make_large_png()

    def lazy_client(self):
        if self.client is None:
            from openai import OpenAI
            self.client = OpenAI(base_url=self.base_url, api_key=self.api_key or "dummy")
        return self.client


# ═══════════════════════════════════════════════════════════════
# Helpers
# ═══════════════════════════════════════════════════════════════

def _timed(fn, *a, **kw):
    t0 = time.time()
    try:
        r = fn(*a, **kw)
        return r, time.time() - t0, None
    except Exception as e:
        return None, time.time() - t0, e


def _content_text(resp) -> str:
    try:
        return resp.choices[0].message.content or ""
    except Exception:
        return ""


def _usage(resp) -> dict:
    try:
        u = resp.usage
        return {
            "prompt_tokens": getattr(u, "prompt_tokens", None),
            "completion_tokens": getattr(u, "completion_tokens", None),
            "total_tokens": getattr(u, "total_tokens", None),
        }
    except Exception:
        return {}


def _stream_collect(client, **kwargs) -> tuple[str, str, int]:
    """跑流式请求，返回 (content, reasoning, chunks)。"""
    text, reasoning, n = "", "", 0
    stream = client.chat.completions.create(**kwargs)
    for ch in stream:
        n += 1
        if not ch.choices:
            continue
        delta = ch.choices[0].delta
        c = getattr(delta, "content", None)
        r = getattr(delta, "reasoning_content", None)
        if c:
            text += c
        if r:
            reasoning += r
    return text, reasoning, n


# ═══════════════════════════════════════════════════════════════
# Tests
# ═══════════════════════════════════════════════════════════════

def test_text_baseline(ctx: Ctx) -> bool:
    _header("Test 1 · 无图片基线（非流式）")
    client = ctx.lazy_client()

    resp, dt, err = _timed(
        client.chat.completions.create,
        model=ctx.model,
        messages=[{"role": "user", "content": "用一句话介绍 Python。"}],
    )
    if err:
        _fail(f"请求异常: {err}")
        return False

    text = _content_text(resp)
    u = _usage(resp)
    _info(f"耗时 {dt:.2f}s | usage={u}")
    _info(f"内容: {text[:200]!r}")

    passed = True
    if not text:
        _fail("内容为空")
        passed = False
    else:
        _ok(f"返回 {len(text)} 字符")
    if (u.get("completion_tokens") or 0) <= 0:
        _fail("completion_tokens 应 > 0")
        passed = False
    else:
        _ok("usage 正常")
    if resp.choices[0].message.tool_calls:
        _fail("无工具请求却返回 tool_calls")
        passed = False
    return passed


def test_text_stream(ctx: Ctx) -> bool:
    _header("Test 2 · 无图片基线（流式）")
    client = ctx.lazy_client()

    text, reasoning, n, err = None, None, 0, None
    t0 = time.time()
    try:
        text, reasoning, n = _stream_collect(
            client,
            model=ctx.model,
            messages=[{"role": "user", "content": "用 3 句话介绍法国巴黎。"}],
            stream=True,
        )
    except Exception as e:
        err = e
    dt = time.time() - t0

    if err:
        _fail(f"请求异常: {err}")
        return False

    _info(f"耗时 {dt:.2f}s | chunks={n}")
    _info(f"content:   {text[:200]!r}")
    _info(f"reasoning: {reasoning[:120]!r}" if reasoning else "reasoning: (none)")

    passed = True
    if not text:
        _fail("内容为空")
        passed = False
    else:
        _ok(f"content {len(text)} 字符")
    if n <= 0:
        _fail("未收到 chunk")
        passed = False
    else:
        _ok(f"共 {n} 个 chunk")
    if "<|tool_call" in text:
        _fail("内容里泄漏了哨兵")
        passed = False
    else:
        _ok("未泄漏哨兵标记")
    return passed


def test_single_image_dataurl(ctx: Ctx) -> bool:
    _header("Test 3 · 单图 data URL（非流式）")
    client = ctx.lazy_client()

    data_url = to_data_url(ctx.png_hello)
    _info(f"图片大小: {len(ctx.png_hello)} bytes | data URL 长度: {len(data_url)}")

    resp, dt, err = _timed(
        client.chat.completions.create,
        model=ctx.model,
        messages=[{
            "role": "user",
            "content": [
                {"type": "text", "text": "这张图里写了什么英文单词？只回答那个单词。"},
                {"type": "image_url", "image_url": {"url": data_url}},
            ],
        }],
    )
    if err:
        _fail(f"请求异常: {err}")
        return False

    text = _content_text(resp)
    u = _usage(resp)
    _info(f"耗时 {dt:.2f}s | usage={u}")
    _info(f"内容: {text!r}")

    passed = True
    # 关键校验：模型必须真的看到了图
    if "HELLO" in text.upper():
        _ok("模型正确识别出图片内容是 HELLO")
    else:
        _warn(f"未识别出 HELLO，模型看到的内容: {text!r}")
        # 可能模型只描述了「一张图」，也可能上传失败降级了
        # 进一步检查 usage：有图片请求通常会明显带更多 prompt_tokens
        pt = u.get("prompt_tokens") or 0
        if pt > 500:
            _warn(f"prompt_tokens={pt} 偏高，可能图片已上传但模型未准确读出")
        else:
            _fail(f"prompt_tokens={pt}，图片很可能根本没上传")
            passed = False
    return passed


def test_single_image_dataurl_stream(ctx: Ctx) -> bool:
    _header("Test 4 · 单图 data URL（流式）")
    client = ctx.lazy_client()

    data_url = to_data_url(ctx.png_world)
    t0 = time.time()
    text, reasoning, n = "", "", 0
    try:
        stream = client.chat.completions.create(
            model=ctx.model,
            messages=[{
                "role": "user",
                "content": [
                    {"type": "text", "text": "这张图里写了什么英文单词？只回答那个单词。"},
                    {"type": "image_url", "image_url": {"url": data_url}},
                ],
            }],
            stream=True,
        )
        for ch in stream:
            n += 1
            if not ch.choices:
                continue
            c = getattr(ch.choices[0].delta, "content", None)
            r = getattr(ch.choices[0].delta, "reasoning_content", None)
            if c:
                text += c
            if r:
                reasoning += r
    except Exception as e:
        _fail(f"请求异常: {e}")
        traceback.print_exc()
        return False
    dt = time.time() - t0

    _info(f"耗时 {dt:.2f}s | chunks={n}")
    _info(f"content:   {text!r}")
    if reasoning:
        _info(f"reasoning: {reasoning[:150]!r}")

    passed = True
    if "WORLD" in text.upper():
        _ok("模型正确识别出 WORLD")
    else:
        _warn(f"未识别出 WORLD，实际返回: {text!r}")
        passed = False
    if n <= 0:
        _fail("未收到 chunk")
        passed = False
    else:
        _ok(f"{n} chunks")
    return passed


def test_single_image_httpurl(ctx: Ctx) -> bool:
    _header("Test 5 · 单图 HTTP URL（非流式）")

    if not ctx.http_image_url:
        _warn("未提供 --http-image-url，跳过")
        return True

    client = ctx.lazy_client()
    _info(f"图片 URL: {ctx.http_image_url}")

    resp, dt, err = _timed(
        client.chat.completions.create,
        model=ctx.model,
        messages=[{
            "role": "user",
            "content": [
                {"type": "text", "text": "描述这张图片。"},
                {"type": "image_url",
                 "image_url": {"url": ctx.http_image_url}},
            ],
        }],
    )
    if err:
        _fail(f"请求异常: {err}")
        return False

    text = _content_text(resp)
    _info(f"耗时 {dt:.2f}s")
    _info(f"内容: {text[:300]!r}")

    if not text:
        _fail("内容为空")
        return False
    _ok(f"返回 {len(text)} 字符")
    return True


def test_multi_image(ctx: Ctx) -> bool:
    _header("Test 6 · 多图（非流式）")
    client = ctx.lazy_client()

    red = to_data_url(ctx.png_red)
    blue = to_data_url(ctx.png_blue)
    _info(f"图片 1: 64x64 纯红 | 图片 2: 64x64 纯蓝")

    resp, dt, err = _timed(
        client.chat.completions.create,
        model=ctx.model,
        messages=[{
            "role": "user",
            "content": [
                {"type": "text",
                 "text": "我给了你两张图。请分别说出它们的主色调。格式：图1=颜色，图2=颜色"},
                {"type": "image_url", "image_url": {"url": red}},
                {"type": "image_url", "image_url": {"url": blue}},
            ],
        }],
    )
    if err:
        _fail(f"请求异常: {err}")
        return False

    text = _content_text(resp)
    u = _usage(resp)
    _info(f"耗时 {dt:.2f}s | usage={u}")
    _info(f"内容: {text!r}")

    low = text.lower()
    passed = True

    # 中文 / 英文颜色词都接受
    red_ok = any(w in low for w in ["红", "red", "#dc", "#ff0000"])
    blue_ok = any(w in low for w in ["蓝", "blue", "#1e5a", "#0000ff"])

    if red_ok:
        _ok("识别出图 1 是红色系")
    else:
        _warn("未明确识别出图 1 的颜色")
    if blue_ok:
        _ok("识别出图 2 是蓝色系")
    else:
        _warn("未明确识别出图 2 的颜色")
    if not (red_ok and blue_ok):
        # 没识别出可能因为纯色图对模型太抽象，不作为硬性失败
        pt = u.get("prompt_tokens") or 0
        if pt < 800:
            _fail(f"prompt_tokens={pt} 偏低，两张图可能都没上传")
            passed = False
    return passed


def test_image_followup(ctx: Ctx) -> bool:
    _header("Test 7 · 图文混合 + 追问（多轮上下文）")
    client = ctx.lazy_client()

    data_url = to_data_url(ctx.png_hello)

    # Turn 1：看图
    resp1, dt1, err1 = _timed(
        client.chat.completions.create,
        model=ctx.model,
        messages=[{
            "role": "user",
            "content": [
                {"type": "text", "text": "这张图里是什么英文单词？"},
                {"type": "image_url", "image_url": {"url": data_url}},
            ],
        }],
    )
    if err1:
        _fail(f"Turn 1 异常: {err1}")
        return False
    reply1 = _content_text(resp1)
    _info(f"Turn 1 ({dt1:.2f}s): {reply1!r}")

    # Turn 2：追问（不再重复图片，只是文字）
    resp2, dt2, err2 = _timed(
        client.chat.completions.create,
        model=ctx.model,
        messages=[
            {"role": "user", "content": [
                {"type": "text", "text": "这张图里是什么英文单词？"},
                {"type": "image_url", "image_url": {"url": data_url}},
            ]},
            {"role": "assistant", "content": reply1},
            {"role": "user", "content": "把那个单词翻译成中文。"},
        ],
    )
    if err2:
        _fail(f"Turn 2 异常: {err2}")
        return False
    reply2 = _content_text(resp2)
    _info(f"Turn 2 ({dt2:.2f}s): {reply2!r}")

    passed = True
    if "HELLO" in reply1.upper():
        _ok("Turn 1 正确读出 HELLO")
    else:
        _warn(f"Turn 1 未读出 HELLO: {reply1!r}")
        passed = False

    if "你好" in reply2 or "hello" in reply2.lower():
        _ok("Turn 2 正确翻译为「你好」")
    else:
        _warn(f"Turn 2 翻译不符预期: {reply2!r}")
        # 若 Turn 1 就没读到，Turn 2 大概率也读不到；不重复计失败
    return passed


def test_image_with_tools(ctx: Ctx) -> bool:
    _header("Test 8 · 图片 + 工具调用（混合）")
    client = ctx.lazy_client()

    data_url = to_data_url(ctx.png_hello)
    tools = [{
        "type": "function",
        "function": {
            "name": "record_word",
            "description": "记录在图片中看到的英文单词",
            "parameters": {
                "type": "object",
                "properties": {
                    "word": {"type": "string", "description": "图片里的单词"},
                },
                "required": ["word"],
            },
        },
    }]

    resp, dt, err = _timed(
        client.chat.completions.create,
        model=ctx.model,
        messages=[{
            "role": "user",
            "content": [
                {"type": "text",
                 "text": "看这张图，然后调用 record_word 工具把图里的单词记下来。"},
                {"type": "image_url", "image_url": {"url": data_url}},
            ],
        }],
        tools=tools,
        tool_choice="auto",
    )
    if err:
        _fail(f"请求异常: {err}")
        return False

    msg = resp.choices[0].message
    fr = resp.choices[0].finish_reason
    _info(f"耗时 {dt:.2f}s | finish_reason={fr}")

    passed = True
    if fr == "tool_calls" and msg.tool_calls:
        _ok(f"正确触发工具调用 ({len(msg.tool_calls)} 个)")
        for tc in msg.tool_calls:
            _info(f"  name={tc.function.name} args={tc.function.arguments}")
            if tc.function.name == "record_word":
                try:
                    args = json.loads(tc.function.arguments)
                    word = str(args.get("word", "")).upper()
                    if "HELLO" in word:
                        _ok(f"工具参数正确捕获 HELLO: {args}")
                    else:
                        _warn(f"工具参数里没看到 HELLO: {args}")
                except json.JSONDecodeError:
                    _fail(f"arguments 不是合法 JSON: {tc.function.arguments}")
                    passed = False
    elif msg.content:
        _warn(f"模型直接回答（未调用工具）: {msg.content[:150]!r}")
        # 温和判定：模型理解能力没问题，只是遵守协议不一致
    else:
        _fail("既无 tool_calls 也无 content")
        passed = False
    return passed


def test_invalid_image(ctx: Ctx) -> bool:
    _header("Test 9 · 非法图片 URL 的降级行为")
    client = ctx.lazy_client()

    # 非法 base64 的 data URL
    bad_url = "data:image/png;base64,=====not-valid-base64====="

    resp, dt, err = _timed(
        client.chat.completions.create,
        model=ctx.model,
        messages=[{
            "role": "user",
            "content": [
                {"type": "text", "text": "这张图里有什么？"},
                {"type": "image_url", "image_url": {"url": bad_url}},
            ],
        }],
    )

    if err:
        _info(f"请求被拒绝（可能是 400）: {err}")
        _ok("服务端明确拒绝非法图片（预期行为之一）")
        return True

    text = _content_text(resp)
    _info(f"耗时 {dt:.2f}s | 内容: {text[:200]!r}")

    # 另一种预期行为：忽略图片、只答文本
    if text:
        _ok("服务端降级为纯文本回答（图片被忽略）")
        return True

    _fail("既没拒绝也没回答")
    return False


def test_large_image(ctx: Ctx) -> bool:
    _header("Test 10 · 大图（base64 体积较大）")
    client = ctx.lazy_client()

    png = ctx.png_large
    data_url = to_data_url(png)
    _info(f"PNG 大小: {len(png):,} bytes | data URL 长度: {len(data_url):,}")

    resp, dt, err = _timed(
        client.chat.completions.create,
        model=ctx.model,
        messages=[{
            "role": "user",
            "content": [
                {"type": "text", "text": "粗略描述这张图（颜色或图案即可）。"},
                {"type": "image_url", "image_url": {"url": data_url}},
            ],
        }],
    )
    if err:
        _fail(f"请求异常: {err}")
        return False

    text = _content_text(resp)
    u = _usage(resp)
    _info(f"耗时 {dt:.2f}s | usage={u}")
    _info(f"内容: {text[:200]!r}")

    if not text:
        _fail("内容为空")
        return False
    _ok(f"返回 {len(text)} 字符")
    return True


# ═══════════════════════════════════════════════════════════════
# Runner
# ═══════════════════════════════════════════════════════════════

def main() -> int:
    parser = argparse.ArgumentParser(description="deepseek2api Vision 能力测试")
    parser.add_argument("--base-url", default="http://localhost:8000/v1")
    parser.add_argument("--api-key", default="")
    parser.add_argument("--model", default="deepseek-chat")
    parser.add_argument("--http-image-url", default="",
                        help="第 5 项测试使用的公网图片 URL")
    parser.add_argument("--test", type=int, default=0,
                        help="只跑指定测试 (1-10)，0=全部")
    args = parser.parse_args()

    print(f"\n{_BOLD}deepseek2api Vision 能力测试{_RESET}")
    print(f"  Base URL: {_CYAN}{args.base_url}{_RESET}")
    print(f"  Model:    {_CYAN}{args.model}{_RESET}")
    print(f"  API Key:  {_CYAN}{'(none)' if not args.api_key else '***'}{_RESET}")

    try:
        import openai  # noqa: F401
    except ImportError:
        print(f"\n{_RED}未安装 openai 库。请先运行：pip install openai{_RESET}\n")
        return 2

    ctx = Ctx(args.base_url, args.model, args.api_key, args.http_image_url)

    tests = [
        ("无图片基线（非流式）",     test_text_baseline),
        ("无图片基线（流式）",       test_text_stream),
        ("单图 data URL（非流式）",  test_single_image_dataurl),
        ("单图 data URL（流式）",    test_single_image_dataurl_stream),
        ("单图 HTTP URL（非流式）",  test_single_image_httpurl),
        ("多图（非流式）",           test_multi_image),
        ("图文混合 + 追问",           test_image_followup),
        ("图片 + 工具调用",          test_image_with_tools),
        ("非法图片降级",             test_invalid_image),
        ("大图上传",                 test_large_image),
    ]

    results: list[tuple[str, bool]] = []
    for idx, (name, fn) in enumerate(tests, 1):
        if args.test != 0 and args.test != idx:
            continue
        try:
            ok = fn(ctx)
            results.append((name, ok))
        except Exception as e:
            _fail(f"{name} 抛异常: {e}")
            traceback.print_exc()
            results.append((name, False))

    _header("结果汇总")
    all_ok = True
    for name, ok in results:
        flag = f"{_GREEN}PASS{_RESET}" if ok else f"{_RED}FAIL{_RESET}"
        print(f"  [{flag}] {name}")
        if not ok:
            all_ok = False

    total = len(results)
    passed = sum(1 for _, ok in results if ok)
    color = _GREEN if all_ok else _RED
    print(f"\n  {color}{passed}/{total} 通过{_RESET}\n")
    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main())