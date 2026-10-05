from __future__ import annotations

import collections
import threading
import time
import uuid
from dataclasses import dataclass
from typing import Any, Deque, Dict, List, Optional


@dataclass(slots=True)
class RequestRecord:
    ts: float
    method: str
    path: str
    protocol: str
    model: str
    status: int
    duration_ms: int
    client_ip: str
    stream: bool
    error: str
    request_id: str


_MAX_LOG_RECORDS = 500
_HOURLY_BUCKETS = 48
_TOP_N_MODELS = 8


class AdminStore:
    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._request_log: Deque[RequestRecord] = collections.deque(maxlen=_MAX_LOG_RECORDS)
        self._total_requests = 0
        self._total_success = 0
        self._total_errors = 0
        self._total_client_errors = 0
        self._model_counter: Dict[str, int] = collections.Counter()
        self._protocol_counter: Dict[str, int] = collections.Counter()
        self._hourly: Deque[Dict[str, Any]] = collections.deque(maxlen=_HOURLY_BUCKETS)
        self._sessions: Dict[str, float] = {}
        self._started_at = time.time()

    def record_request(self, rec: RequestRecord) -> None:
        with self._lock:
            self._request_log.append(rec)
            self._total_requests += 1
            if 200 <= rec.status < 300:
                self._total_success += 1
            elif 400 <= rec.status < 500:
                self._total_client_errors += 1
            else:
                self._total_errors += 1

            if rec.model:
                self._model_counter[rec.model] += 1
            self._protocol_counter[rec.protocol] = self._protocol_counter.get(rec.protocol, 0) + 1
            self._hourly_append(rec)

    def _hourly_append(self, rec: RequestRecord) -> None:
        hour = int(rec.ts // 3600) * 3600
        if not self._hourly or self._hourly[-1]["hour"] != hour:
            self._hourly.append({"hour": hour, "total": 0, "success": 0, "error": 0})
        bucket = self._hourly[-1]
        bucket["total"] += 1
        if 200 <= rec.status < 300:
            bucket["success"] += 1
        elif rec.status >= 500:
            bucket["error"] += 1

    def create_session(self, ttl_seconds: int = 8 * 3600) -> str:
        token = uuid.uuid4().hex + uuid.uuid4().hex
        with self._lock:
            self._sessions[token] = time.time() + ttl_seconds
        return token

    def validate_session(self, token: Optional[str]) -> bool:
        if not token:
            return False
        with self._lock:
            exp = self._sessions.get(token)
            if exp is None:
                return False
            if exp < time.time():
                self._sessions.pop(token, None)
                return False
            return True

    def revoke_session(self, token: str) -> None:
        with self._lock:
            self._sessions.pop(token, None)

    def dashboard(self) -> Dict[str, Any]:
        with self._lock:
            now = time.time()
            cutoff_5m = now - 300
            recent = [r for r in self._request_log if r.ts >= cutoff_5m]
            recent_success = sum(1 for r in recent if 200 <= r.status < 300)
            recent_total = len(recent)
            recent_success_rate = (recent_success / recent_total * 100) if recent_total else 0.0
            recent_latencies = sorted(r.duration_ms for r in recent) if recent else []
            p50 = recent_latencies[len(recent_latencies) // 2] if recent_latencies else 0
            p95_idx = int(len(recent_latencies) * 0.95)
            p95 = recent_latencies[min(p95_idx, len(recent_latencies) - 1)] if recent_latencies else 0

            all_total = self._total_requests
            all_success_rate = (self._total_success / all_total * 100) if all_total else 0.0

            hourly = list(self._hourly)
            top_models = self._model_counter.most_common(_TOP_N_MODELS)
            protocols = dict(self._protocol_counter)

            return {
                "now": now,
                "uptime_seconds": now - self._started_at,
                "all_time": {
                    "total": all_total,
                    "success": self._total_success,
                    "client_errors": self._total_client_errors,
                    "server_errors": self._total_errors,
                    "success_rate": round(all_success_rate, 2),
                },
                "recent_5m": {
                    "total": recent_total,
                    "success": recent_success,
                    "success_rate": round(recent_success_rate, 2),
                    "p50_ms": p50,
                    "p95_ms": p95,
                },
                "hourly": hourly,
                "top_models": [{"model": m, "count": c} for m, c in top_models],
                "protocols": protocols,
            }

    def recent_logs(self, limit: int = 100, only_errors: bool = False) -> List[Dict[str, Any]]:
        with self._lock:
            records = list(self._request_log)
            if only_errors:
                records = [r for r in records if r.status >= 400]
            records = records[-limit:][::-1]
            return [
                {
                    "ts": r.ts,
                    "method": r.method,
                    "path": r.path,
                    "protocol": r.protocol,
                    "model": r.model,
                    "status": r.status,
                    "duration_ms": r.duration_ms,
                    "client_ip": r.client_ip,
                    "stream": r.stream,
                    "error": r.error,
                    "request_id": r.request_id,
                }
                for r in records
            ]


GLOBAL_STORE = AdminStore()


def get_store() -> AdminStore:
    return GLOBAL_STORE


def classify_protocol(path: str) -> str:
    if path.endswith("/v1/chat/completions") or path.endswith("/chat/completions"):
        return "openai-chat"
    if path.endswith("/v1/models"):
        return "meta"
    return "other"