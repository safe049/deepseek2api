"""
OpenAI 兼容工具调用支持。

DeepSeek Web API 是纯文本流 —— 没有原生 function calling。
本模块通过「哨兵协议」+ DSML 在文本层模拟。

## 协议（两套并行，都兼容）

### 1. 哨兵格式（我们提示词教的）

    <|tool_calls|>
    {"name": "get_weather", "arguments": {"city": "Beijing"}}
    <|/tool_calls|>

### 2. DSML 格式（DeepSeek 原生，模型先验最强）

    <｜｜DSML｜｜ calls>
    <｜｜DSML｜｜ invoke name="get_weather">
    <｜｜DSML｜｜ parameter name="city" string="true">Beijing</｜｜DSML｜｜ parameter>
    </｜｜DSML｜｜ invoke>
    </｜｜DSML｜｜ calls>

## 解析容忍矩阵（本次修复）

| 维度        | 接受                                       |
|-------------|--------------------------------------------|
| 竖线        | `|` (U+007C)、`｜` (U+FF5C)，单/双/混合均可 |
| DSML 大小写 | `DSML` / `dsml` / `Dsml` …                 |
| 标签名大小写 | `invoke` / `INVOKE` / `Invoke` …           |
| 标签内空格  | `<｜｜DSML｜｜ invoke>` / `<｜｜DSML｜｜invoke>` 均可 |
| 属性值引号  | `name="x"` / `name='x'` / `name=x`         |
| 属性间隔    | `name = "x"` / `name="x"`                  |
| 属性值内 `>` | `command="a > b"` 安全剥离                  |
| 缺 `</invoke>` | 在下一个 `<invoke` 处隐式切分           |
| 缺 `</calls>`  | `finalize` 兜底解析                    |
| 孤立关闭标签  | SCAN 状态静默跳过，不吐到 content         |

## 关键设计：Artifacts 零容忍

客户端会把流里的 `content` 和 `tool_calls` 都渲染给用户。
**任何 DSML 相关的文本碎片出现在 content 里都是视觉污染**。

原则：

  1. 进入 DSML 状态后，只有两种出路：解析出 tool_calls 或静默丢弃
  2. SCAN 状态遇到孤立的 DSML 闭合标签，跳过不吐
  3. 空 DSML 块、残缺 DSML 块，静默丢弃
  4. `finalize` 的 DSML 残余也优先尝试解析，失败则丢弃

## 工具名匹配策略

litellm / Claude Code 侧传下来的工具名可能是 `Bash`（首字母大写）
或 `bash_command`（带下划线），而模型输出 `name="bash"`。
字符串不完全相等 ≠ 幻觉 → 宽容匹配 + debug 日志，绝不硬拒收。

## 解析优先级

    _try_dsml > _try_sentinel > _try_code_blocks > _try_embedded_json > _try_bare_json
"""

from __future__ import annotations

import json
import logging
import re
import uuid
from dataclasses import dataclass, field
from typing import Any, Optional

logger = logging.getLogger("deepseek2api.tool_calling")

# ═══════════════════════════════════════════════════════════════
# 哨兵常量
# ═══════════════════════════════════════════════════════════════

TOOL_OPEN_SINGULAR  = "<|tool_call|>"
TOOL_OPEN_PLURAL    = "<|tool_calls|>"
TOOL_CLOSE_SINGULAR = "<|/tool_call|>"
TOOL_CLOSE_PLURAL   = "<|/tool_calls|>"

_OPEN_VARIANTS  = (TOOL_OPEN_PLURAL, TOOL_OPEN_SINGULAR)
_CLOSE_VARIANTS = (TOOL_CLOSE_PLURAL, TOOL_CLOSE_SINGULAR)

# ═══════════════════════════════════════════════════════════════
# DSML 正则（★ 容忍所有常见变体 ★）
# ═══════════════════════════════════════════════════════════════

# 竖线：全角 U+FF5C 或半角 U+007C，允许混合、允许 1~2 个
_BAR = "[|\uff5c]"  

# DSML 标记：竖线 + DSML + 竖线，中间允许空格，大小写不敏感
_DSML_MARKER = rf'{_BAR}+\s*DSML\s*{_BAR}+'

# 标签属性：双引号 / 单引号 / 裸值 均可，值里可含 >（在引号内）
_ATTRS = r'(?:"[^"]*"|\'[^\']*\'|[^>"\'])*'

# 标签：< / </ + 标记 + 标签名 + 属性 + >
_CALLS_OPEN_RE = re.compile(
    rf'<\s*{_DSML_MARKER}\s*calls\b({_ATTRS})>',
    re.IGNORECASE,
)
_CALLS_CLOSE_RE = re.compile(
    rf'</\s*{_DSML_MARKER}\s*calls\s*>',
    re.IGNORECASE,
)
_INVOKE_OPEN_RE = re.compile(
    rf'<\s*{_DSML_MARKER}\s*invoke\b({_ATTRS})>',
    re.IGNORECASE,
)
_INVOKE_CLOSE_RE = re.compile(
    rf'</\s*{_DSML_MARKER}\s*invoke\s*>',
    re.IGNORECASE,
)
_PARAM_OPEN_RE = re.compile(
    rf'<\s*{_DSML_MARKER}\s*parameter\b({_ATTRS})>',
    re.IGNORECASE,
)
_PARAM_CLOSE_RE = re.compile(
    rf'</\s*{_DSML_MARKER}\s*parameter\s*>',
    re.IGNORECASE,
)

# 属性名值对：name="x" / name='x' / name=x
_ATTR_PAIR_RE = re.compile(
    r'([a-zA-Z_][a-zA-Z0-9_-]*)\s*=\s*(?:"([^"]*)"|\'([^\']*)\'|([^\s>]+))'
)

# 任意 DSML 起始标签（SCAN 状态检测用）
_ANY_DSML_OPEN_RE = re.compile(
    rf'<\s*{_DSML_MARKER}\s*(?:calls|invoke)\b',
    re.IGNORECASE,
)

# 孤立关闭标签（SCAN 状态剔除用）
_NOISE_RE = re.compile(
    rf'</\s*{_DSML_MARKER}\s*(?:calls|invoke|parameter)\s*>',
    re.IGNORECASE,
)

# 提示词里仍用全角，因为这是 DeepSeek 原生先验
_DSML = "\uff5c\uff5cDSML\uff5c\uff5c"

# ═══════════════════════════════════════════════════════════════
# 长度常量
# ═══════════════════════════════════════════════════════════════

_MAX_OPEN_LEN  = max(len(v) for v in _OPEN_VARIANTS)
_MAX_CLOSE_LEN = max(len(v) for v in _CLOSE_VARIANTS)

_ANCHORS = (
    '{"name"', '{"tool_calls"', '{"function"',
    '{ "name"', '{ "tool_calls"', '{ "function"',
)
_MAX_ANCHOR_LEN = max(len(a) for a in _ANCHORS)

# DSML 标签最长前缀的保守上界
_MAX_DSML_TAG_LEN = 64

_SAFE_KEEP = max(_MAX_OPEN_LEN, _MAX_ANCHOR_LEN, _MAX_DSML_TAG_LEN)

# ═══════════════════════════════════════════════════════════════
# 数据结构
# ═══════════════════════════════════════════════════════════════

@dataclass(slots=True)
class ParsedToolCall:
    id: str
    name: str
    arguments: str

    def to_openai(self) -> dict:
        return {
            "id": self.id,
            "type": "function",
            "function": {"name": self.name, "arguments": self.arguments},
        }


@dataclass(slots=True)
class ToolParseResult:
    calls: list[ParsedToolCall] = field(default_factory=list)
    clean_text: str = ""


@dataclass(slots=True)
class StreamAction:
    kind: str
    text: str = ""
    tool_calls: Optional[list[ParsedToolCall]] = None


# ═══════════════════════════════════════════════════════════════
# 属性解析
# ═══════════════════════════════════════════════════════════════

def _parse_attrs(s: str) -> dict[str, str]:
    """容忍 name="x" / name='x' / name=x 三种写法。"""
    result: dict[str, str] = {}
    if not s:
        return result
    for m in _ATTR_PAIR_RE.finditer(s):
        key = m.group(1)
        val = m.group(2)
        if val is None:
            val = m.group(3)
        if val is None:
            val = m.group(4)
        result[key] = val
    return result


def _decode_dsml_value(raw: str, is_string: bool):
    if is_string:
        return raw
    stripped = raw.strip()
    if not stripped:
        return ""
    try:
        return json.loads(stripped)
    except json.JSONDecodeError:
        return raw


# ═══════════════════════════════════════════════════════════════
# 提示词构建（保持全角竖线，与 DeepSeek 原生先验一致）
# ═══════════════════════════════════════════════════════════════

def _tool_name(tool: dict) -> str:
    fn = tool.get("function") if isinstance(tool.get("function"), dict) else tool
    return str(fn.get("name", "")).strip()


def build_tool_prompt(tools, tool_choice="auto"):
    if not tools:
        return ""

    D = _DSML

    lines = [
        "",
        "==== TOOL CALLING PROTOCOL ====",
        "",
        "!!! CRITICAL FORMAT RULE !!!",
        "When you need a tool, the FIRST characters of your response",
        f"MUST be the literal token `<{D} calls>`.",
        "If you write any sentence before it, the call is INVALID and",
        "the system WILL NOT execute the tool.",
        "",
        "WRONG (the tool will NOT run):",
        "  Sure, I'll check the weather for you.",
        '  {"name": "get_weather", "arguments": {"city": "Beijing"}}',
        "",
        "RIGHT:",
        f"  <{D} calls>",
        f'  <{D} invoke name="get_weather">',
        f'  <{D} parameter name="city" string="true">Beijing</{D} parameter>',
        f"  </{D} invoke>",
        f"  </{D} calls>",
        "",
        "Multi-tool example (multiple invokes INSIDE ONE calls block):",
        f"  <{D} calls>",
        f'  <{D} invoke name="read">',
        f'  <{D} parameter name="file_path" string="true">/etc/hosts</{D} parameter>',
        f"  </{D} invoke>",
        f'  <{D} invoke name="bash">',
        f'  <{D} parameter name="command" string="true">ls -la</{D} parameter>',
        f"  </{D} invoke>",
        f"  </{D} calls>",
        "",
        "Rules:",
        f"  - No text, no markdown, no code fences outside the <{D} calls> block.",
        f"  - ONE <{D} calls> block per response; wrap EACH tool call in its own <{D} invoke name=\"...\">.",
        f"  - Every argument uses <{D} parameter name=\"...\" string=\"true|false\">value</{D} parameter>.",
        f"  - Use string=\"true\" for text values; string=\"false\" (or omit) for numbers/objects/arrays.",
        f"  - Do NOT emit <|tool_calls|> or <|tool_call|> sentinel tokens.",
        f"  - Do NOT emit a trailing <{D} calls> block with no invokes.",
        f"  - Use EXACT tool names: " + ", ".join(_tool_name(t) for t in tools if isinstance(t, dict)),
        "",
        "## Tools",
        "",
    ]

    for tool in tools:
        if not isinstance(tool, dict):
            continue
        fn = tool.get("function") if isinstance(tool.get("function"), dict) else tool
        name = str(fn.get("name", "")).strip()
        if not name:
            continue
        desc = str(fn.get("description", "")).strip()
        params = fn.get("parameters") or {}
        props = params.get("properties") if isinstance(params, dict) else None
        required = set(params.get("required", [])) if isinstance(params, dict) else set()

        lines.append(f"### {name}")
        if desc:
            lines.append(desc)
        if props:
            lines.append("Parameters:")
            for pname, pinfo in props.items():
                pinfo = pinfo if isinstance(pinfo, dict) else {}
                ptype = pinfo.get("type", "string")
                pdesc = pinfo.get("description", "")
                req = " (required)" if pname in required else ""
                lines.append(f"  - {pname} ({ptype}{req}): {pdesc}")
        lines.append("")

    if tool_choice == "required":
        lines.append(">> You MUST call at least one tool in every response.")
    elif isinstance(tool_choice, dict):
        fn_name = (tool_choice.get("function") or {}).get("name", "")
        if fn_name:
            lines.append(f">> You MUST call the tool `{fn_name}`.")

    return "\n".join(lines)


# ═══════════════════════════════════════════════════════════════
# 工具名匹配
# ═══════════════════════════════════════════════════════════════

def _norm_name(name: str) -> str:
    return name.strip().lower().replace("-", "_")


def _match_tool_name(name: str, known: set[str]) -> bool:
    if not known:
        return True
    if name in known:
        return True
    n = _norm_name(name)
    return any(_norm_name(k) == n for k in known)


def _normalize_call(data: Any, tools: Optional[list[dict]]) -> Optional[ParsedToolCall]:
    """宽容策略：白名单只做软匹配，不匹配也放行（打 debug 日志）。"""
    if not isinstance(data, dict):
        return None

    name = data.get("name")
    if not isinstance(name, str) or not name:
        fn = data.get("function")
        if isinstance(fn, dict):
            name = fn.get("name")
    if not isinstance(name, str) or not name:
        return None

    if tools is not None:
        known = {_tool_name(t) for t in tools if isinstance(t, dict)}
        known.discard("")
        if known and not _match_tool_name(name, known):
            logger.debug(
                "tool name %r not in known set %r — accepted anyway (lenient match)",
                name, sorted(known),
            )

    args = data.get("arguments")
    if args is None:
        fn = data.get("function")
        if isinstance(fn, dict):
            args = fn.get("arguments")
    args_str = _coerce_arguments(args)

    return ParsedToolCall(
        id=str(data.get("id") or f"call_{uuid.uuid4().hex[:24]}"),
        name=name,
        arguments=args_str,
    )


def _coerce_arguments(args: Any) -> str:
    if args is None:
        return "{}"
    if isinstance(args, dict):
        return json.dumps(args, ensure_ascii=False)
    if isinstance(args, list):
        return json.dumps(args, ensure_ascii=False)
    if isinstance(args, str):
        s = args.strip()
        if not s:
            return "{}"
        try:
            json.loads(s)
            return s
        except json.JSONDecodeError:
            return json.dumps({"value": args}, ensure_ascii=False)
    try:
        return json.dumps(args, ensure_ascii=False)
    except (TypeError, ValueError):
        return "{}"


# ═══════════════════════════════════════════════════════════════
# DSML body 解析（★ 核心：缺闭合标签也能救 ★）
# ═══════════════════════════════════════════════════════════════

def _parse_dsml_body(body: str, tools: Optional[list[dict]] = None) -> list[ParsedToolCall]:
    """
    解析 DSML 块内部的若干 invoke。容忍：

      - 缺 </invoke>：在下一个 <invoke 处隐式切分；都没有则吃到末尾
      - 缺 </parameter>：吃到 invoke 末尾
      - 属性无引号：name=foo
    """
    calls: list[ParsedToolCall] = []
    if not body:
        return calls

    pos = 0
    while True:
        m = _INVOKE_OPEN_RE.search(body, pos)
        if not m:
            break

        attrs = _parse_attrs(m.group(1))
        name = attrs.get("name", "")

        # 定位 invoke 结束
        close_m = _INVOKE_CLOSE_RE.search(body, m.end())
        if close_m:
            invoke_inner = body[m.end():close_m.start()]
            pos = close_m.end()
        else:
            # 缺 </invoke>：下一个 <invoke 隐式切分
            next_inv = _INVOKE_OPEN_RE.search(body, m.end())
            if next_inv:
                invoke_inner = body[m.end():next_inv.start()]
                pos = next_inv.start()
            else:
                invoke_inner = body[m.end():]
                pos = len(body)

        arguments: dict = {}
        ppos = 0
        while True:
            pm = _PARAM_OPEN_RE.search(invoke_inner, ppos)
            if not pm:
                break

            p_attrs = _parse_attrs(pm.group(1))
            p_name = p_attrs.get("name", "")
            p_is_string = str(p_attrs.get("string", "")).lower() in ("true", "1", "yes", "")

            p_close = _PARAM_CLOSE_RE.search(invoke_inner, pm.end())
            if p_close:
                raw_value = invoke_inner[pm.end():p_close.start()]
                ppos = p_close.end()
            else:
                # 缺 </parameter>：吃到 invoke 末尾
                raw_value = invoke_inner[pm.end():]
                ppos = len(invoke_inner)

            if p_name:
                arguments[p_name] = _decode_dsml_value(raw_value, p_is_string)

        call = _normalize_call({"name": name, "arguments": arguments}, tools)
        if call:
            calls.append(call)

    return calls


# ═══════════════════════════════════════════════════════════════
# 非流式解析
# ═══════════════════════════════════════════════════════════════

def _try_dsml(text: str, tools: Optional[list[dict]]) -> ToolParseResult:
    """定位并解析 DSML 块（三种形态兼容）。"""
    if not text:
        return ToolParseResult()

    # ── 形态 ①：完整 <calls> ... </calls> ─────────────────
    m_open = _CALLS_OPEN_RE.search(text)
    if m_open:
        m_close = _CALLS_CLOSE_RE.search(text, m_open.end())
        if m_close:
            body = text[m_open.end():m_close.start()]
            calls = _parse_dsml_body(body, tools)
            if calls:
                clean = (text[:m_open.start()] + text[m_close.end():]).strip()
                return ToolParseResult(calls=calls, clean_text=clean)

        # 有 calls 开头但无 calls 闭合 → 用最后一个 invoke 收尾
        body = text[m_open.end():]
        calls = _parse_dsml_body(body, tools)
        if calls:
            last_close = None
            for m in _INVOKE_CLOSE_RE.finditer(text, m_open.end()):
                last_close = m
            if last_close:
                clean = (text[:m_open.start()] + text[last_close.end():]).strip()
            else:
                clean = text[:m_open.start()].strip()
            return ToolParseResult(calls=calls, clean_text=clean)

    # ── 形态 ②：缺头但有 calls 闭合 ──────────────────────
    m_close = _CALLS_CLOSE_RE.search(text)
    if m_close:
        m_inv = _INVOKE_OPEN_RE.search(text, 0, m_close.start())
        if m_inv:
            body = text[m_inv.start():m_close.start()]
            calls = _parse_dsml_body(body, tools)
            if calls:
                clean = (text[:m_inv.start()] + text[m_close.end():]).strip()
                return ToolParseResult(calls=calls, clean_text=clean)

    # ── 形态 ③：只有 invoke ──────────────────────────────
    m_inv = _INVOKE_OPEN_RE.search(text)
    if m_inv:
        body = text[m_inv.start():]
        calls = _parse_dsml_body(body, tools)
        if calls:
            last_close = None
            for m in _INVOKE_CLOSE_RE.finditer(text, m_inv.start()):
                last_close = m
            if last_close:
                clean = (text[:m_inv.start()] + text[last_close.end():]).strip()
            else:
                clean = text[:m_inv.start()].strip()
            return ToolParseResult(calls=calls, clean_text=clean)

    return ToolParseResult()


def _try_sentinel(text: str, tools: Optional[list[dict]]) -> ToolParseResult:
    for open_v in _OPEN_VARIANTS:
        open_idx = text.find(open_v)
        if open_idx == -1:
            continue
        close_idx = -1
        close_v = ""
        for cv in _CLOSE_VARIANTS:
            i = text.find(cv, open_idx + len(open_v))
            if i != -1:
                close_idx = i
                close_v = cv
                break
        if close_idx == -1:
            continue
        body = text[open_idx + len(open_v): close_idx]
        calls = parse_sentinel_body(body, tools)
        if calls:
            clean = (text[:open_idx] + text[close_idx + len(close_v):]).strip()
            return ToolParseResult(calls=calls, clean_text=clean)
    return ToolParseResult()


def parse_sentinel_body(body: str, tools: Optional[list[dict]] = None) -> list[ParsedToolCall]:
    calls: list[ParsedToolCall] = []
    for line in body.splitlines():
        line = line.strip()
        if not line or line.startswith("//"):
            continue
        try:
            data = json.loads(line)
        except json.JSONDecodeError:
            continue
        call = _normalize_call(data, tools)
        if call:
            calls.append(call)
    return calls


def _try_code_blocks(text: str, tools: Optional[list[dict]]) -> ToolParseResult:
    pattern = re.compile(r"```(?:json)?\s*\n?(.*?)\n?\s*```", re.DOTALL)
    for m in pattern.finditer(text):
        block = m.group(1).strip()
        calls = _parse_json_calls(block, tools)
        if calls:
            clean = (text[:m.start()] + text[m.end():]).strip()
            return ToolParseResult(calls=calls, clean_text=clean)
    return ToolParseResult()


def _try_embedded_json(text: str, tools: Optional[list[dict]]) -> ToolParseResult:
    if not text:
        return ToolParseResult()

    anchors = ('{"name"', '{"tool_calls"', '{"function"',
               '{ "name"', '{ "tool_calls"', '{ "function"')

    for anchor in anchors:
        idx = text.find(anchor)
        if idx == -1:
            continue

        depth, in_str, esc, end = 0, False, False, -1
        for i in range(idx, len(text)):
            ch = text[i]
            if esc:
                esc = False
                continue
            if ch == "\\":
                esc = True
                continue
            if ch == '"':
                in_str = not in_str
                continue
            if in_str:
                continue
            if ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    end = i + 1
                    break

        if end == -1:
            continue

        candidate = text[idx:end]
        calls = _parse_json_calls(candidate, tools)
        if calls:
            clean = (text[:idx] + text[end:]).strip()
            return ToolParseResult(calls=calls, clean_text=clean)

    return ToolParseResult()


def _try_bare_json(text: str, tools: Optional[list[dict]]) -> ToolParseResult:
    stripped = text.strip()
    calls = _parse_json_calls(stripped, tools)
    if calls:
        return ToolParseResult(calls=calls, clean_text="")
    return ToolParseResult()


def _parse_json_calls(raw: str, tools: Optional[list[dict]]) -> list[ParsedToolCall]:
    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        return []

    if isinstance(data, dict):
        if isinstance(data.get("tool_calls"), list):
            return [c for c in (_normalize_call(x, tools) for x in data["tool_calls"]) if c]
        single = _normalize_call(data, tools)
        if single:
            return [single]

    if isinstance(data, list):
        return [c for c in (_normalize_call(x, tools) for x in data) if c]

    return []


def parse_tool_calls(
    text: str,
    tools: Optional[list[dict]] = None,
    require_marker: bool = False,
) -> ToolParseResult:
    """
    require_marker=True 时，只有明确的 DSML / 哨兵标记才解析；
    不解析裸 JSON / 代码块 —— 用于「调用方没声明 tools，但模型仍输出了 DSML」的场景。
    """
    if not text:
        return ToolParseResult(clean_text=text)

    result = _try_dsml(text, tools)
    if result.calls:
        return result

    result = _try_sentinel(text, tools)
    if result.calls:
        return result

    if require_marker:
        return ToolParseResult(clean_text=text)

    result = _try_code_blocks(text, tools)
    if result.calls:
        return result

    result = _try_embedded_json(text, tools)
    if result.calls:
        return result

    result = _try_bare_json(text, tools)
    if result.calls:
        return result

    return ToolParseResult(clean_text=text)


# ═══════════════════════════════════════════════════════════════
# 流式检测器（★ 本次修复主体 ★）
# ═══════════════════════════════════════════════════════════════

class StreamingToolDetector:
    """
    有状态流式工具调用检测器。

    状态机：
      SCAN     —— 扫描 DSML / 哨兵 / JSON 锚点；尾部缓冲可能成为前缀的字符
      DSML     —— 累积直到 </…DSML… calls> 或 </…DSML… invoke>，然后整体解析
      SENTINEL —— 累积直到 </|tool_calls|>
      JSON     —— 累积直到括号平衡

    关键行为：
      - 遇到最早的结束标签（calls 或 invoke）就切块解析
      - 多 invoke 的 calls 块会被逐 invoke 切分处理
      - SCAN 状态剥掉孤立关闭标签，不吐到 content
      - 任何块解析失败都静默丢弃，绝不吐文本污染 content
    """

    def __init__(
        self,
        tools: Optional[list[dict]] = None,
        require_marker: bool = False,
    ) -> None:
        self._tools = tools
        self._require_marker = require_marker
        self._state = "SCAN"
        self._buf = ""
        self._closed = False
        self._saw_dsml = False
    # ── 主入口 ────────────────────────────────────────────

    def feed(self, delta: str) -> list[StreamAction]:
        if self._closed:
            return [StreamAction(kind="text", text=delta)] if delta else []

        self._buf += delta
        actions: list[StreamAction] = []

        guard = 0
        while self._buf and guard < 256:
            guard += 1
            if self._state == "DSML":
                if not self._step_dsml(actions):
                    break
            elif self._state == "SENTINEL":
                if not self._step_sentinel(actions):
                    break
            elif self._state == "JSON":
                if not self._step_json(actions):
                    break
            else:
                if not self._step_scan(actions):
                    break

        return actions

    def finalize(self) -> list[StreamAction]:
        actions: list[StreamAction] = []
        if not self._buf:
            self._closed = True
            return actions

        if self._state == "DSML":
            body = self._buf
            m_open = _CALLS_OPEN_RE.match(body)
            if m_open:
                body = body[m_open.end():]

            calls = _parse_dsml_body(body, self._tools)
            if calls:
                actions.append(StreamAction(kind="tool_calls", tool_calls=calls))
            else:
                logger.debug(
                    "finalize: discarding unparseable DSML residual len=%d",
                    len(self._buf),
                )
            self._buf = ""
            self._state = "SCAN"
            self._closed = True
            return actions

        if self._state == "SENTINEL":
            text = self._buf
            for v in _OPEN_VARIANTS:
                if text.startswith(v):
                    text = text[len(v):]
                    break
            if text:
                actions.append(StreamAction(kind="text", text=text))
            self._buf = ""
            self._state = "SCAN"
            self._closed = True
            return actions

        # JSON 或 SCAN：按文本吐出（可能是不完整的 JSON，客户端自行处理）
        if self._buf:
            actions.append(StreamAction(kind="text", text=self._buf))
        self._buf = ""
        self._state = "SCAN"
        self._closed = True
        return actions

    # ── SCAN ──────────────────────────────────────────────

    def _step_scan(self, actions: list[StreamAction]) -> bool:
        if not self._buf:
            return False

        idx, kind = self._find_earliest_scan_token()
        if idx == -1:
            # 没有锚点：吐出安全前缀（保留可能成为标签前缀的尾巴）
            safe_len = self._safe_prefix_len()
            if safe_len > 0:
                actions.append(StreamAction(kind="text", text=self._buf[:safe_len]))
                self._buf = self._buf[safe_len:]
            return False

        if idx > 0:
            prefix = self._buf[:idx]
            # 前缀里如果有孤立关闭标签，剥掉再吐
            if _NOISE_RE.search(prefix):
                prefix = _NOISE_RE.sub("", prefix)
            if prefix:
                actions.append(StreamAction(kind="text", text=prefix))
            self._buf = self._buf[idx:]

        if kind == "dsml_open":
            self._state = "DSML"
            self._saw_dsml = True
            return True
        if kind == "dsml_close":
            # 孤立关闭标签：剥掉不吐
            m = _NOISE_RE.match(self._buf)
            if m:
                self._buf = self._buf[m.end():]
                return True
            return False
        if kind == "sentinel_open":
            self._state = "SENTINEL"
            return True
        if kind == "json_open":
            self._state = "JSON"
            return True
        return False

    def _find_earliest_scan_token(self) -> tuple[int, str]:
        best_idx, best_kind = -1, ""

        # 哨兵始终识别
        for v in _OPEN_VARIANTS:
            i = self._buf.find(v)
            if i != -1 and (best_idx == -1 or i < best_idx):
                best_idx, best_kind = i, "sentinel_open"

        # DSML 始终识别
        m = _ANY_DSML_OPEN_RE.search(self._buf)
        if m and (best_idx == -1 or m.start() < best_idx):
            best_idx, best_kind = m.start(), "dsml_open"

        # 孤立关闭标签：只有 tools 模式才剥；无 tools 时当普通文本
        if not self._require_marker:
            m = _NOISE_RE.search(self._buf)
            if m and (best_idx == -1 or m.start() < best_idx):
                best_idx, best_kind = m.start(), "dsml_close"

        # JSON 锚点：require_marker=True 时忽略
        if not self._require_marker:
            for a in _ANCHORS:
                i = self._buf.find(a)
                if i != -1 and (best_idx == -1 or i < best_idx):
                    best_idx, best_kind = i, "json_open"

        return best_idx, best_kind

    def _safe_prefix_len(self) -> int:
        """
        返回可安全吐为文本的前缀长度。

        策略：
          - 如果 buf 末尾有未闭合的 "<..."（可能构成 DSML 标签或哨兵），保留它。
          - 如果末尾是 JSON 锚点的部分前缀（如 "{"、"{\"n"），保留它。
          - 否则整个 buf 都可以吐。

        关键：绝不能只看末字符。单个 "｜" 不是锚点开头，
        但它可能是 "<｜｜DSML｜｜" 的中间片段；一旦把它当独立字符吐出去，
        前面的 "<" 也保不住，整个锚点就永远错过。
        """
        buf = self._buf
        n = len(buf)
        if n == 0:
            return 0

        lower = max(0, n - _SAFE_KEEP)

        # ── 1. 找最长的、未闭合的 "<..." 后缀 ──
        # 从末尾往前扫，遇到第一个 "<" 就判断它到末尾是否闭合。
        #   - 未闭合 → 这就是我们要保留的起点
        #   - 已闭合 → 说明末尾的 "<...>" 是完整标签（不属于 DSML 的话，
        #             早就该被 _find_earliest_scan_token 处理掉了）；
        #             再往前找已经没意义，直接退出
        for i in range(n - 1, lower - 1, -1):
            if buf[i] == "<":
                if ">" not in buf[i:]:
                    return i
                break

        # ── 2. JSON 锚点的部分前缀 ──
        # 只在 require_marker=False（即调用方声明了 tools）时才会触发 JSON 检测，
        # 但为了对称处理，这里总是检查；不会误伤纯文本。
        if not self._require_marker:
            best_json = n
            for a in _ANCHORS:
                max_k = min(len(a), n)
                for k in range(1, max_k + 1):
                    suffix = buf[n - k:]
                    if a.startswith(suffix):
                        best_json = min(best_json, n - k)
            if best_json < n:
                # 如果 "<..." 保留点更靠前，取更早的那个
                return min(best_json, n) if "json_only" else best_json

        return n

    # ── DSML ──────────────────────────────────────────────

    def _step_dsml(self, actions: list[StreamAction]) -> bool:
        """
        DSML 状态：只找最早的结束标签（</…calls> 或 </…invoke>）。

        - 先出现 </…invoke> 就切一个 invoke，回到 SCAN（下一个 <invoke 会重新进入）
        - 先出现 </…calls> 就切整个块
        - 都不出现 → 继续累积，finalize 兜底
        """
        end_pos = -1
        end_len = 0

        m_calls = _CALLS_CLOSE_RE.search(self._buf)
        if m_calls:
            end_pos = m_calls.start()
            end_len = m_calls.end() - m_calls.start()

        m_invoke = _INVOKE_CLOSE_RE.search(self._buf)
        if m_invoke and (end_pos == -1 or m_invoke.start() < end_pos):
            end_pos = m_invoke.start()
            end_len = m_invoke.end() - m_invoke.start()

        if end_pos == -1:
            return False

        body = self._buf[:end_pos]
        rest = self._buf[end_pos + end_len:]

        # 若 body 以 <… calls> 开头，去掉外层 calls 开标签
        m_open = _CALLS_OPEN_RE.match(body)
        if m_open:
            body = body[m_open.end():]

        calls = _parse_dsml_body(body, self._tools)
        if calls:
            actions.append(StreamAction(kind="tool_calls", tool_calls=calls))
        else:
            logger.debug(
                "_step_dsml: DSML block yielded no calls, discarding body len=%d",
                len(body),
            )

        self._buf = rest
        self._state = "SCAN"
        return True

    # ── SENTINEL ──────────────────────────────────────────

    def _step_sentinel(self, actions: list[StreamAction]) -> bool:
        close_idx, close_v = self._find_close()
        if close_idx == -1:
            return False

        open_v = self._open_at(0)
        if open_v is None:
            actions.append(StreamAction(kind="text", text=self._buf))
            self._buf = ""
            self._state = "SCAN"
            return True

        body = self._buf[len(open_v):close_idx]
        rest = self._buf[close_idx + len(close_v):]

        calls = parse_sentinel_body(body, self._tools)
        if calls:
            actions.append(StreamAction(kind="tool_calls", tool_calls=calls))
        else:
            logger.debug("_step_sentinel: no calls, discarding block")

        self._buf = rest
        self._state = "SCAN"
        return True

    # ── JSON ──────────────────────────────────────────────

    def _step_json(self, actions: list[StreamAction]) -> bool:
        end = self._find_json_end(self._buf)
        if end == -1:
            return False

        block = self._buf[:end]
        rest = self._buf[end:]

        calls = _parse_json_calls(block, self._tools)
        if calls:
            actions.append(StreamAction(kind="tool_calls", tool_calls=calls))
        else:
            # 解析失败 → 当普通文本吐（不是工具调用）
            actions.append(StreamAction(kind="text", text=block))

        self._buf = rest
        self._state = "SCAN"
        return True

    # ── 内部工具 ──────────────────────────────────────────

    def _find_close(self) -> tuple[int, str]:
        start = 0
        for v in _OPEN_VARIANTS:
            if self._buf.startswith(v):
                start = len(v)
                break
        best_idx, best_v = -1, ""
        for v in _CLOSE_VARIANTS:
            i = self._buf.find(v, start)
            if i != -1 and (best_idx == -1 or i < best_idx):
                best_idx, best_v = i, v
        return best_idx, best_v

    def _open_at(self, idx: int) -> Optional[str]:
        for v in _OPEN_VARIANTS:
            if self._buf.startswith(v, idx):
                return v
        return None

    @staticmethod
    def _find_json_end(s: str) -> int:
        if not s.startswith("{"):
            return -1
        depth, in_str, esc = 0, False, False
        for i, ch in enumerate(s):
            if esc:
                esc = False
                continue
            if ch == "\\":
                esc = True
                continue
            if ch == '"':
                in_str = not in_str
                continue
            if in_str:
                continue
            if ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    return i + 1
        return -1