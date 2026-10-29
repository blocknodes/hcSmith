# hcProxy

OpenAI-format 兼容代理：hcAgent / hcTools 把 LLM 网关指到本服务，即可在**不改调用方代码**的前提下：

1. **捕获**每次调用的完整请求/响应（含 `metadata` 业务标签）落盘到 `data/captures/<app>/YYYY-MM-DD.jsonl`；
2. **透传**到后端真实 vllm server（`HC_PROXY_UPSTREAM`），请求/响应均不改写。

## 快速开始

```bash
cd hcProxy
pip install -r requirements.txt
./run.sh          # 监听 0.0.0.0:8091，日志 run.log
```

调用方只需把 API base 指到代理（hcAgent 示例）：

```bash
export HC_LLM_API_BASE=http://127.0.0.1:8091/v1
```

## 配置（环境变量）

| 变量 | 默认 | 说明 |
|---|---|---|
| `HC_PROXY_HOST` / `HC_PROXY_PORT` | `0.0.0.0:8091` | 监听地址 |
| `HC_PROXY_UPSTREAM` | `http://10.19.96.219:4003/v1` | 上游真实 vllm（OpenAI-format /v1） |
| `HC_PROXY_UPSTREAM_KEY` | 取 `HC_LLM_API_KEY` | 上游 Bearer key |
| `HC_PROXY_DATA_DIR` | `<项目>/data/captures` | 捕获落盘目录 |
| `HC_PROXY_CAPTURE` | `1` | 设 `0` 关闭落盘（只代理） |
| `HC_PROXY_TIMEOUT` | `120` | 上游超时秒数 |

## 接口

- `POST /v1/chat/completions` — 主链路：透传 + 捕获（非流式）
- `GET  /v1/models` — 静态模型列表（探活/兼容）
- `GET  /healthz` — 存活探活

## 捕获格式

`data/captures/<app标记>/YYYY-MM-DD.jsonl`，每行一个调用：

```json
{"ts": "2026-09-20 12:00:00",
 "request": {"model": "baseline", "messages": [...], "metadata": {"app": "hcAgent", "purpose": "plan", ...}},
 "response": {"role": "assistant", "content": "..."},
 "status": 200, "latency_ms": 321.0, "error": null}
```

- `<app标记>` 取 `metadata.app`（如 `hcAgent`）；无 metadata 的裸请求落 `_raw/`。
- 上游失败也落盘（`status`/`error` 记录失败原因），便于排查网关问题。
- 落盘是旁路（尽力而为）：磁盘异常只告警，不影响代理返回。

## 设计约定

- **不改写**：请求/响应原样透传，metadata 只是搭车字段（上游忽略也不影响）。
- **非流式**：调用方（hcAgent/hcTools）均为非 streaming 请求，暂不做 SSE 转发。
- hcAgent 侧的 metadata schema 见 `hcAgent/app/prompts.py::build_llm_metadata()`。
