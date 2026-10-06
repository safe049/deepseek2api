from __future__ import annotations

import os
import shutil
from dataclasses import dataclass
from pathlib import Path


class ConfigError(ValueError):
    pass

def parse_float(value: str | None, default: float) -> float:
    if value is None or value == "":
        return default
    try:
        return float(value)
    except (TypeError, ValueError) as exc:
        raise ConfigError(f"浮点配置值无效: {value}") from exc

def parse_dotenv(path: Path) -> dict[str, str]:
    values: dict[str, str] = {}
    if not path.exists():
        return values
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except UnicodeDecodeError as exc:
        raise ConfigError(f"配置文件不是有效的 UTF-8 编码: {path}") from exc
    except OSError as exc:
        raise ConfigError(f"读取配置文件失败: {path} error={exc}") from exc

    for raw_line in lines:
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, raw_value = line.split("=", 1)
        value = raw_value.strip()
        if value.startswith(("'", '"')) and value.endswith(("'", '"')) and len(value) >= 2:
            value = value[1:-1]
        values[key.strip()] = value
    return values


def parse_bool(value: str | None, default: bool = False) -> bool:
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def parse_int(value: str | None, default: int) -> int:
    if value is None or value == "":
        return default
    try:
        return int(value)
    except (TypeError, ValueError) as exc:
        raise ConfigError(f"整数配置值无效: {value}") from exc


def parse_list(value: str | None, default: tuple[str, ...] = ()) -> list[str]:
    if value is None or value.strip() == "":
        return list(default)
    return [item.strip() for item in value.split(",") if item.strip()]



EXPOSED_MODELS = (
    "deepseek-chat",
    "deepseek-reasoner",
    "default",
)

@dataclass(slots=True)
class AppConfig:
    env_file_path: Path
    env_file_created: bool

    host: str
    port: int
    api_prefix: str
    log_level: str
    debug_dump_all: bool
    request_timeout: int

    server_api_keys: list[str]

    auth_token: str
    device_id: str
    settings_token: str
    cookie: str   
    base_url: str
    wasm_path: str

    default_model: str
    delete_conversation: bool
    max_concurrency: int
    queue_wait_timeout: int

    admin_password: str

    # ── 防风控 ──
    anti_rate_limit_enabled: bool = True
    anti_rate_limit_min_delay: float = 1.5
    anti_rate_limit_max_delay: float = 3.0

    # ── 会话缓存 ──
    session_cache_enabled: bool = True
    session_cache_max_size: int = 500
    session_cache_ttl: int = 7200    

    search_enabled: bool = False
    strip_citations: bool = True

    file_upload_enabled: bool = True

    # ── DeepSeek API 端点 ──
    @property
    def pow_challenge_url(self) -> str:
        return f"{self.base_url}/api/v0/chat/create_pow_challenge"

    @property
    def session_create_url(self) -> str:
        return f"{self.base_url}/api/v0/chat_session/create"

    @property
    def chat_completion_url(self) -> str:
        return f"{self.base_url}/api/v0/chat/completion"

    @property
    def session_delete_url(self) -> str:
        return f"{self.base_url}/api/v0/chat_session/delete"

    @property
    def hif_leim_url(self) -> str:
        return "https://hif-leim.deepseek.com/query"

    @property
    def hif_dliq_url(self) -> str:
        return "https://hif-dliq.deepseek.com/query"


def ensure_env_file(env_path: Path) -> bool:
    if env_path.exists():
        return False

    cwd = Path.cwd()
    pkg_root = Path(__file__).resolve().parent
    repo_root = pkg_root.parent.parent

    example_candidates = [
        cwd / ".env.example",
        cwd / "configs" / "env.example",
        cwd / "configs" / ".env.example",
        repo_root / ".env.example",
        repo_root / "configs" / "env.example",
        repo_root / "configs" / ".env.example",
        env_path.with_name(".env.example"),
        env_path.parent / ".env.example",
    ]

    example_path = next((c for c in example_candidates if c.exists()), None)
    if example_path is None:
        return False

    try:
        env_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(example_path, env_path)
    except OSError as exc:
        raise ConfigError(f"自动创建配置文件失败: source={example_path} target={env_path} error={exc}") from exc
    return True


def load_config(env_file: str = ".env") -> AppConfig:
    import logging
    logger = logging.getLogger("deepseek2api.config")

    env_path = Path(env_file)
    env_file_created = ensure_env_file(env_path)
    file_values = parse_dotenv(env_path)
    values = {**file_values, **os.environ}

    host = values.get("HOST", "127.0.0.1").strip() or "127.0.0.1"

    api_prefix = values.get("API_PREFIX", "/v1").strip()
    if not api_prefix:
        api_prefix = "/v1"
    if not api_prefix.startswith("/"):
        api_prefix = f"/{api_prefix}"
    api_prefix = api_prefix.rstrip("/") or "/v1"

    log_level = values.get("LOG_LEVEL", "INFO").strip().upper() or "INFO"
    debug_dump_all = parse_bool(values.get("DEBUG_DUMP_ALL"), False)
    if debug_dump_all:
        log_level = "DEBUG"
    if log_level not in {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}:
        log_level = "INFO"

    delete_conversation = parse_bool(values.get("DEEPSEEK_DELETE_CONVERSATION"), True)

    session_cache_enabled = not delete_conversation
    if not session_cache_enabled:
        logger.info(
            "DEEPSEEK_DELETE_CONVERSATION=true → 已禁用 session 缓存，每次请求独立。"
        )    

    anti_rate_limit_enabled = parse_bool(
        values.get("DEEPSEEK_ANTI_RATE_LIMIT_ENABLED"), True
    )
    anti_rate_limit_min_delay = parse_float(
        values.get("DEEPSEEK_ANTI_RATE_LIMIT_MIN_DELAY_SECONDS"), 1.5
    )
    anti_rate_limit_max_delay = parse_float(
        values.get("DEEPSEEK_ANTI_RATE_LIMIT_MAX_DELAY_SECONDS"), 3.0
    )

    config = AppConfig(
        env_file_path=env_path,
        env_file_created=env_file_created,
        host=host,
        port=parse_int(values.get("PORT"), 8000),
        api_prefix=api_prefix,
        log_level=log_level,
        debug_dump_all=debug_dump_all,
        request_timeout=parse_int(values.get("REQUEST_TIMEOUT_SECONDS"), 180),
        server_api_keys=parse_list(values.get("SERVER_API_KEYS")),
        auth_token=values.get("DEEPSEEK_AUTH_TOKEN", "").strip(),
        device_id=values.get("DEEPSEEK_DEVICE_ID", "").strip(),
        settings_token=values.get("DEEPSEEK_SETTINGS_TOKEN", "").strip(),
        cookie=values.get("DEEPSEEK_COOKIE", "").strip(), 
        base_url=values.get("DEEPSEEK_BASE_URL", "https://chat.deepseek.com").rstrip("/"),
        wasm_path=values.get("DEEPSEEK_WASM_PATH", "./sha3_wasm_bg.wasm").strip(),
        default_model=values.get("DEEPSEEK_DEFAULT_MODEL", "deepseek-chat").strip(),
        delete_conversation=delete_conversation,
        max_concurrency=max(1, parse_int(values.get("DEEPSEEK_MAX_CONCURRENCY"), 1)),
        queue_wait_timeout=parse_int(values.get("DEEPSEEK_QUEUE_WAIT_TIMEOUT_SECONDS"), 60),
        admin_password=values.get("ADMIN_PASSWORD", "admin").strip(),  
        anti_rate_limit_enabled=anti_rate_limit_enabled,
        anti_rate_limit_min_delay=anti_rate_limit_min_delay,
        anti_rate_limit_max_delay=anti_rate_limit_max_delay,        
        session_cache_max_size=max(1, parse_int(values.get("DEEPSEEK_SESSION_CACHE_MAX_SIZE"), 500)),
        session_cache_enabled=session_cache_enabled,
        session_cache_ttl=max(60, parse_int(values.get("DEEPSEEK_SESSION_CACHE_TTL_SECONDS"), 7200)),
        search_enabled=parse_bool(values.get("DEEPSEEK_SEARCH_ENABLED"), False), 
        strip_citations=parse_bool(values.get("DEEPSEEK_STRIP_CITATIONS"), True),
        file_upload_enabled=parse_bool(
            values.get("DEEPSEEK_FILE_UPLOAD_ENABLED"), True
        ),        
    )

    if not (1 <= config.port <= 65535):
        raise ConfigError(f"端口配置超出范围: PORT={config.port}")
    if config.request_timeout <= 0:
        raise ConfigError(f"请求超时必须大于 0: REQUEST_TIMEOUT_SECONDS={config.request_timeout}")
    if config.anti_rate_limit_min_delay < 0:
        raise ConfigError(
            f"防风控最小延迟必须 >= 0: {config.anti_rate_limit_min_delay}"
        )
    if config.anti_rate_limit_max_delay < config.anti_rate_limit_min_delay:
        raise ConfigError(
            f"防风控最大延迟必须 >= 最小延迟: "
            f"min={config.anti_rate_limit_min_delay} "
            f"max={config.anti_rate_limit_max_delay}"
        )        
    if config.search_enabled:
        logger.info(
            "联网搜索已启用：所有模型请求将携带 search_enabled=true"
        )
    else:
        logger.info("联网搜索未启用（可通过 DEEPSEEK_SEARCH_ENABLED=true 开启）")
    if not config.auth_token:
        raise ConfigError("缺少 DEEPSEEK_AUTH_TOKEN 配置")

    logger.info(
        "配置加载完成 端口=%s 并发=%s 模型=%s 日志级别=%s",
        config.port,
        config.max_concurrency,
        config.default_model,
        config.log_level,
    )
    if config.anti_rate_limit_enabled:
        logger.info(
            "防风控已启用：每次上游聊天请求前随机延迟 %.2f~%.2fs",
            config.anti_rate_limit_min_delay,
            config.anti_rate_limit_max_delay,
        )
    else:
        logger.warning("防风控已禁用：请求将无间隔直发上游")    
    return config