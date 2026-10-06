# src/deepseek2api/services/pow_solver.py
from __future__ import annotations

import base64
import json
import threading
from logging import Logger

from deepseek_pow import Challenge, registry
from deepseek_pow.exceptions import DeepSeekPowError


class PoWSolver:
    """DeepSeek PoW 求解器，使用 deepseek-pow 库。"""

    def __init__(self, wasm_path: str, logger: Logger) -> None:
        self.logger = logger
        self._lock = threading.RLock()
        try:
            self._default_solver = registry.get("DeepSeekHashV1")
            self.logger.info("PoW 求解器初始化完成（deepseek-pow + Wasmtime）")
        except DeepSeekPowError as exc:
            self.logger.error("PoW 求解器初始化失败: %s", exc)
            raise RuntimeError(f"PoW 求解器初始化失败: {exc}") from exc

    def solve(
        self,
        challenge: str,
        salt: str,
        difficulty: int,
        algorithm: str = "DeepSeekHashV1",
        expire_at: int = 0,
        signature: str = "",
        target_path: str = "/api/v0/chat/completion",   # ← 新增
    ) -> str:
        """
        求解 PoW challenge，返回 Base64 编码的 x-ds-pow-response。
        """
        try:
            solver = registry.get(algorithm)
        except DeepSeekPowError:
            self.logger.warning("未注册的算法 %s，回退到 DeepSeekHashV1", algorithm)
            solver = self._default_solver

        prefix = f"{salt}_{expire_at}_"

        pow_challenge = Challenge(
            algorithm=algorithm,
            challenge=challenge,
            salt=salt,
            difficulty=difficulty,
            expire_at=expire_at,
            signature=signature,
        )

        self.logger.info(
            "开始 PoW 求解 algorithm=%s difficulty=%d expire_at=%d target=%s",
            algorithm, difficulty, expire_at, target_path,
        )

        try:
            solution = solver.solve(pow_challenge)
        except DeepSeekPowError as exc:
            self.logger.error("PoW 求解失败: %s", exc)
            raise RuntimeError(f"PoW 求解失败: {exc}") from exc

        payload = solution.to_answer_payload()
        answer_int = int(payload.get("answer", 0))

        self.logger.info("PoW 求解成功 answer=%d", answer_int)

        return self._make_header(
            algorithm=algorithm,
            challenge=challenge,
            salt=salt,
            answer=answer_int,
            signature=signature,
            target_path=target_path,
        )

    def _make_header(
        self,
        algorithm: str,
        challenge: str,
        salt: str,
        answer: int,
        signature: str,
        target_path: str = "/api/v0/chat/completion",
    ) -> str:
        """构造 Base64 编码的 x-ds-pow-response 头值。"""
        pow_payload = {
            "algorithm": algorithm,
            "challenge": challenge,
            "salt": salt,
            "answer": answer,          # 整数，不是字符串
            "signature": signature,
            "target_path": target_path,
        }
        pow_json = json.dumps(pow_payload, separators=(",", ":"))
        return base64.b64encode(pow_json.encode("utf-8")).decode("ascii")