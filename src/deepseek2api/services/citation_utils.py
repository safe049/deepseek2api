"""剥离 DeepSeek Web 上游输出的 [citation:N] / [reference:N] 占位符。

上游把引用信息以两个通道分开吐：
  1. content 里内联 [reference:N] / [citation:N] 占位符
  2. references 事件里给 URL/title

Web 前端会合并渲染成角标；API 直出时占位符就是纯文本垃圾，
默认剥离。将来想做"真引用"可在此模块扩展为 markdown 链接。
"""
from __future__ import annotations

import re

# 完整标记：[citation:1] / [reference:0] / [citation：3]（中文冒号也兼容）
_CIT_INLINE = re.compile(r'\[(?:citation|reference)[:：]\s*\d+\]', re.IGNORECASE)

# 尾部"可能是未完成标记"的前缀：可跨 chunk 拆分
_CIT_TAIL = re.compile(r'\[(?:citation|reference)[:：]?\d*$', re.IGNORECASE)


def strip_citation_marks(text: str) -> str:
    """一次性剥离（非流式场景）。"""
    return _CIT_INLINE.sub("", text)


class CitationStripper:
    """
    流式剥离器。

    因为标记可能跨 chunk（如 "...答案[citation:" | "1]..."），
    需要在尾部缓存最长 ~24 字符的潜在未完成标记。
    """

    def __init__(self) -> None:
        self._tail = ""

    def feed(self, delta: str) -> str:
        if not delta:
            return ""
        text = self._tail + delta

        # 尾部形如 "[citation:" / "[reference:" / "[citation:"  → 缓冲
        m = _CIT_TAIL.search(text)
        if m:
            self._tail = text[m.start():]
            text = text[:m.start()]
        else:
            self._tail = ""

        return _CIT_INLINE.sub("", text)

    def finalize(self) -> str:
        """流结束时把缓冲里残余的（可能未闭合）标记清理后吐出。"""
        out = _CIT_INLINE.sub("", self._tail)
        self._tail = ""
        return out