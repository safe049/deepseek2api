# deepseek2api

将 **DeepSeek Web 端接口**（chat.deepseek.com）转换为 **OpenAI Chat Completions 兼容 API** 的轻量代理。

无需官方 API Key，直接复用浏览器会话凭证，即可让任何 OpenAI 兼容客户端（LangChain、LlamaIndex、OpenAI SDK、各类 ChatBox / NextChat / Cherry Studio 前端）调用 DeepSeek 的对话与推理模型。

> 仅供学习与个人研究使用。请遵守 DeepSeek 的服务条款，不要用于商业用途或高频滥用。

---

## 特性

- **OpenAI 兼容接口**：`/v1/chat/completions`、`/v1/models`，支持流式（SSE）与非流式。
- **多模型支持**：`deepseek-chat`、`deepseek-reasoner`、`default`。
- **推理内容透传**：`deepseek-reasoner` 的思维链以 `reasoning_content` 字段返回，兼容主流客户端。
- **原生工具调用（Function Calling）**：通过哨兵协议 + 流式检测器，把 DeepSeek Web 端的文本输出还原为标准 OpenAI `tool_calls` 结构。
- **联网搜索**：可全局开启（`DEEPSEEK_SEARCH_ENABLED=true`），也可单次请求通过 `extra_body.search_enabled` 临时开启；支持自动剥离 `[citation:N]` 引用角标。
- **PoW 求解**：内置 `DeepSeekHashV1` 工作量证明求解（基于 wasmtime + `deepseek-pow`），自动应对上游挑战。
- **会话缓存**：复用 `chat_session`，减少建会话开销；支持前缀哈希与上下文超限自动重开对话。
- **防风控**：上游请求前随机延迟，可配置上下限。
- **自动重试**：`rate_limit_reached` 冷却重试、`context_length_exceeded` 清理缓存后使用完整历史重开。
- **并发控制**：内置排队队列与等待超时，避免上游风控。
- **管理面板**：内置 Web 控制台 `/admin`，查看请求日志、模型、配置与运行状态。
- **零依赖框架**：基于标准库 `http.server`，无需 FastAPI / Flask。
- **自动生成配置**：首次启动若缺少 `.env`，会自动从 `configs/env.example` 复制。

---

## 项目结构

```
deepseek2api/
├── main.py                     # 便捷入口（等价于 python -m deepseek2api）
├── pyproject.toml              # 打包与依赖声明
├── configs/
│   └── env.example             # 配置模板（首次启动自动复制为 .env）
├── src/deepseek2api/
│   ├── __main__.py             # CLI 入口
│   ├── app.py                  # 应用装配、信号处理、优雅关闭
│   ├── config.py               # .env 解析与配置校验
│   ├── server.py               # HTTP 路由（OpenAI 兼容层）
│   ├── logging_utils.py        # 日志与调试转储
│   ├── admin/                  # 管理面板（API + 静态页面 + 指标存储）
│   ├── core/
│   │   ├── openai_compat.py    # OpenAI 响应/错误结构生成
│   │   └── model_profiles.py   # 模型能力描述
│   └── services/
│       ├── deepseek_client.py  # 上游客户端、队列、重试、流式处理
│       ├── deepseek_auth.py    # 凭证 / 请求头管理
│       ├── pow_solver.py       # DeepSeekHashV1 PoW 求解
│       ├── translator.py       # OpenAI ⇄ DeepSeek 消息与流式转换
│       ├── tool_calling.py     # 工具调用哨兵协议解析
│       ├── citation_utils.py   # 引用角标剥离
│       ├── session_cache.py    # 会话缓存
│       └── cookie_jar.py       # Cookie 处理
├── test_conversation.py        # 对话链路测试
└── test_tools.py               # 工具调用测试
```

---

## 快速开始

### 1. 环境要求

- Python **3.12+**
- 可访问 `chat.deepseek.com` 的网络环境
- DeepSeek 网页版登录后的会话凭证（见下方「获取凭证」）

### 2. 安装

```bash
git clone <your-repo-url> deepseek2api
cd deepseek2api

python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate

pip install -e .
```

依赖仅两项：`deepseek-pow`（PoW 计算）与 `wasmtime`（运行 WASM）。

### 3. 准备 PoW 所需的 WASM 文件

PoW 求解需要 DeepSeek 前端的 `sha3_wasm_bg.wasm`：

1. 打开浏览器访问 `https://chat.deepseek.com`，进入开发者工具 → Network；
2. 在加载的 JS 资源中找到 `sha3_wasm_bg.wasm`（或从打包的 JS 中定位其地址）并下载；
3. 放到项目根目录，路径与 `DEEPSEEK_WASM_PATH` 保持一致（默认 `./sha3_wasm_bg.wasm`）。

### 4. 获取凭证

登录 `https://chat.deepseek.com` 后打开开发者工具，在任意一个 `/api/v0/...` 请求的 **Request Headers** 中提取：

| 配置项 | 来源 |
| --- | --- |
| `DEEPSEEK_AUTH_TOKEN` | `Authorization: Bearer <token>`，去掉 `Bearer ` 前缀 |
| `DEEPSEEK_DEVICE_ID` | `x-device-id` |
| `DEEPSEEK_SETTINGS_TOKEN` | `x-settings-token`（JWE 加密串） |
| `DEEPSEEK_COOKIE` | 完整 `Cookie` 请求头 |

> 凭证会过期。若出现 401 / 认证失败，重新登录并更新 `.env`。

### 5. 配置

首次启动会自动从 `configs/env.example` 生成 `.env`，也可手动复制：

```bash
cp configs/env.example .env
```

编辑 `.env`，至少填写 `DEEPSEEK_AUTH_TOKEN`、`DEEPSEEK_DEVICE_ID`、`DEEPSEEK_SETTINGS_TOKEN`、`DEEPSEEK_COOKIE`。

### 6. 启动

```bash
# 方式一：控制台命令（安装后可用）
deepseek2api

# 方式二：模块方式
python -m deepseek2api

# 方式三：根目录脚本
python main.py
```

默认监听 `http://127.0.0.1:8000`，健康检查：

```bash
curl http://127.0.0.1:8000/health
```

---

## 配置项

完整说明见 [`configs/env.example`](configs/env.example)，常用项如下：

### 服务

| 变量 | 默认值 | 说明 |
| --- | --- | --- |
| `HOST` | `127.0.0.1` | 监听地址 |
| `PORT` | `8000` | 监听端口 |
| `API_PREFIX` | `/v1` | API 路径前缀 |
| `LOG_LEVEL` | `INFO` | 日志级别 |
| `DEBUG_DUMP_ALL` | `false` | 打开后转储完整请求/响应体，并强制 `DEBUG` 级别 |
| `REQUEST_TIMEOUT_SECONDS` | `180` | 上游请求超时 |

### 鉴权

| 变量 | 默认值 | 说明 |
| --- | --- | --- |
| `SERVER_API_KEYS` | 空 | 本服务自身的 API Key，逗号分隔；为空则不校验 |

### DeepSeek 上游

| 变量 | 默认值 | 说明 |
| --- | --- | --- |
| `DEEPSEEK_AUTH_TOKEN` | — | **必填**，Bearer Token |
| `DEEPSEEK_DEVICE_ID` | — | 设备 ID |
| `DEEPSEEK_SETTINGS_TOKEN` | — | `x-settings-token` |
| `DEEPSEEK_COOKIE` | — | 完整 Cookie |
| `DEEPSEEK_BASE_URL` | `https://chat.deepseek.com` | 上游基础地址 |
| `DEEPSEEK_WASM_PATH` | `./sha3_wasm_bg.wasm` | PoW WASM 文件路径 |
| `DEEPSEEK_DEFAULT_MODEL` | `deepseek-chat` | 未指定 model 时的默认模型 |
| `DEEPSEEK_DELETE_CONVERSATION` | `true` | 每次回答后删除会话；设为 `true` 会**禁用**会话缓存 |
| `DEEPSEEK_MAX_CONCURRENCY` | `1` | 最大并发 |
| `DEEPSEEK_QUEUE_WAIT_TIMEOUT_SECONDS` | `60` | 排队等待超时，超时返回 `503 queue_timeout` |

### 联网搜索与引用

| 变量 | 默认值 | 说明 |
| --- | --- | --- |
| `DEEPSEEK_SEARCH_ENABLED` | `false` | 全局开启联网搜索（响应更慢、更耗额度） |
| `DEEPSEEK_STRIP_CITATIONS` | `true` | 剥离上游 `[citation:N]` / `[reference:N]` 角标 |

### 防风控

| 变量 | 默认值 | 说明 |
| --- | --- | --- |
| `DEEPSEEK_ANTI_RATE_LIMIT_ENABLED` | `true` | 上游聊天请求前加入随机延迟 |
| `DEEPSEEK_ANTI_RATE_LIMIT_MIN_DELAY_SECONDS` | `1.5` | 随机延迟下限 |
| `DEEPSEEK_ANTI_RATE_LIMIT_MAX_DELAY_SECONDS` | `3.0` | 随机延迟上限（须 ≥ 下限） |

### 管理面板

| 变量 | 默认值 | 说明 |
| --- | --- | --- |
| `ADMIN_PASSWORD` | `admin` | 管理面板登录密码，**建议修改** |

---

## API 使用

### 模型列表

```bash
curl http://127.0.0.1:8000/v1/models \
  -H "Authorization: Bearer <SERVER_API_KEYS 中的任意一个>"
```

暴露的模型：`deepseek-chat`、`deepseek-reasoner`、`default`。

### 对话补全（非流式）

```bash
curl http://127.0.0.1:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer <your-server-api-key>" \
  -d '{
    "model": "deepseek-chat",
    "messages": [
      {"role": "system", "content": "你是一个乐于助人的助手。"},
      {"role": "user", "content": "用一句话介绍你自己。"}
    ]
  }'
```

### 对话补全（流式）

```bash
curl -N http://127.0.0.1:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "deepseek-reasoner",
    "messages": [{"role": "user", "content": "9.11 和 9.8 哪个大？"}],
    "stream": true
  }'
```

### OpenAI Python SDK

```python
from openai import OpenAI

client = OpenAI(
    base_url="http://127.0.0.1:8000/v1",
    api_key="your-server-api-key",  # 若未配置 SERVER_API_KEYS 可随意填写
)

resp = client.chat.completions.create(
    model="deepseek-chat",
    messages=[{"role": "user", "content": "你好"}],
)
print(resp.choices[0].message.content)
```

### 工具调用

```python
tools = [{
    "type": "function",
    "function": {
        "name": "get_weather",
        "description": "查询指定城市的天气",
        "parameters": {
            "type": "object",
            "properties": {"city": {"type": "string", "description": "城市名"}},
            "required": ["city"],
        },
    },
}]

resp = client.chat.completions.create(
    model="deepseek-chat",
    messages=[{"role": "user", "content": "北京现在天气怎么样？"}],
    tools=tools,
)
print(resp.choices[0].message.tool_calls)
```

### 单次开启联网搜索

```python
resp = client.chat.completions.create(
    model="deepseek-chat",
    messages=[{"role": "user", "content": "今天的科技新闻有哪些？"}],
    extra_body={"search_enabled": True},
)
```

---

## 管理面板

浏览器访问 `http://127.0.0.1:8000/admin`，使用 `ADMIN_PASSWORD` 登录。可查看：

- 请求量、成功率、平均耗时等概览指标
- 请求日志（方法、路径、协议、模型、状态码、耗时、客户端 IP、错误）
- 当前模型与配置快照

管理面板 API：`/admin/api/login`、`/logout`、`/dashboard`、`/logs`、`/models`、`/config`。

---

## 端点一览

| 方法 | 路径 | 说明 |
| --- | --- | --- |
| `GET` | `/health` | 健康检查，含配置摘要与缓存统计 |
| `GET` | `/v1/models` | 模型列表 |
| `GET` | `/v1/models/{id}` | 单个模型详情 |
| `POST` | `/v1/chat/completions` | 对话补全（流式 / 非流式） |
| `GET` | `/admin` | 管理面板页面 |
| `*` | `/admin/api/*` | 管理面板接口 |

所有响应带 CORS 头（`Access-Control-Allow-Origin: *`）。

---

## 错误处理

响应遵循 OpenAI 错误结构：

```json
{
  "error": {
    "message": "Incorrect API key provided.",
    "type": "authentication_error",
    "code": "invalid_api_key",
    "param": null
  }
}
```

| 场景 | 状态码 | `code` |
| --- | --- | --- |
| 本服务鉴权失败 | `401` | `invalid_api_key` |
| 请求体非法 JSON / 非 UTF-8 / 非对象 | `400` | `invalid_json` / `invalid_encoding` / `invalid_payload` |
| 缺少 `messages` | `400` | `invalid_request` |
| 模型不存在 | `404` | `model_not_found` |
| 排队超时 | `503` | `queue_timeout` |
| 上游返回错误 | 透传上游状态码 | `upstream_error` |
| 其他内部异常 | `500` / `502` | `internal_error` / `upstream_error` |

---

## 测试

```bash
pip install pytest
pytest -q
```

- [`test_conversation.py`](test_conversation.py)：对话链路、流式输出、上下文超限等。
- [`test_tools.py`](test_tools.py)：工具调用解析与流式检测。

---

## 常见问题

**Q：启动报 `缺少 DEEPSEEK_AUTH_TOKEN 配置`？**
A：`.env` 未填写或未被读取。确认工作目录下存在 `.env` 且 `DEEPSEEK_AUTH_TOKEN` 非空。

**Q：上游返回 401 / 认证失败？**
A：网页端凭证已过期。重新登录 `chat.deepseek.com`，更新 `AUTH_TOKEN`、`DEVICE_ID`、`SETTINGS_TOKEN`、`COOKIE`。

**Q：PoW 求解失败或找不到 WASM？**
A：确认 `sha3_wasm_bg.wasm` 已下载且 `DEEPSEEK_WASM_PATH` 指向正确路径。

**Q：频繁 `rate_limit_reached`？**
A：降低 `DEEPSEEK_MAX_CONCURRENCY`（建议 `1`），保持防风控开启并适当加大随机延迟。

**Q：会话缓存为什么没生效？**
A：`DEEPSEEK_DELETE_CONVERSATION=true` 会强制禁用缓存（每次请求独立会话）。需要多轮上下文复用时设为 `false`。

**Q：如何对外提供服务？**
A：默认只监听 `127.0.0.1`。如需暴露，请设置 `SERVER_API_KEYS` 并配合反向代理与 HTTPS，切勿裸奔。

---

## 免责声明

本项目通过逆向 Web 端接口实现，**非 DeepSeek 官方产品**，与 DeepSeek 官方无任何关联。使用本项目可能违反上游服务条款，由此产生的一切后果由使用者自行承担。请勿用于商业用途、批量滥用或任何违法场景。

---

## License

[MIT](LICENSE)
