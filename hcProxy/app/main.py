"""hcProxy：对外提供 OpenAI-format 接口，捕获请求/响应后透传到真实 vllm server。

链路：hcAgent/hcTools → hcProxy(:8091) → 真实 vllm（HC_PROXY_UPSTREAM）

- POST /v1/chat/completions  主链路：body 原样（含 metadata）转发上游；
  响应原样回给调用方，同时落一条捕获记录（data/captures/<app>/YYYY-MM-DD.jsonl）。
- GET  /v1/models            静态模型列表（探活/客户端兼容用），不转发上游。
- GET  /healthz              存活探针。

透传语义：**不改写请求、不改写响应**——metadata 只是搭车字段，上游多数实现会忽略；
捕获是旁路（尽力而为），落盘失败不影响主链路。
"""
from __future__ import annotations

import logging
import time

import httpx
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from . import capture, config

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(name)s %(levelname)s %(message)s",
)
logger = logging.getLogger("hcProxy.main")

app = FastAPI(title="hcProxy", docs_url=None, redoc_url=None)

# 与 hcAgent 同款：进程级共享连接池。
_client: httpx.AsyncClient | None = None


def _get_client() -> httpx.AsyncClient:
    global _client
    if _client is None:
        _client = httpx.AsyncClient(timeout=config.TIMEOUT)
    return _client


@app.on_event("shutdown")
async def _shutdown() -> None:
    global _client
    if _client is not None:
        await _client.aclose()
        _client = None


@app.get("/healthz")
async def healthz() -> dict[str, str]:
    return {"status": "ok", "upstream": config.UPSTREAM}


@app.get("/v1/models")
async def models() -> dict[str, object]:
    """静态模型列表：探活/客户端兼容用。真实可用性以上游为准。"""
    return {
        "object": "list",
        "data": [{"id": "baseline", "object": "model", "owned_by": "hcProxy"}],
    }


@app.post("/v1/chat/completions")
async def chat_completions(request: Request) -> JSONResponse:
    """透传 + 捕获。请求/响应均不改写；metadata 随 body 原样到上游。"""
    started = time.perf_counter()
    try:
        payload = await request.json()
    except Exception:  # noqa: BLE001
        return JSONResponse(
            status_code=400,
            content={"error": {"message": "request body is not valid JSON",
                               "type": "invalid_request_error"}},
        )
    if not isinstance(payload, dict):
        return JSONResponse(
            status_code=400,
            content={"error": {"message": "request body must be a JSON object",
                               "type": "invalid_request_error"}},
        )

    headers = {"Content-Type": "application/json"}
    if config.API_KEY:
        headers["Authorization"] = f"Bearer {config.API_KEY}"

    status: int | None = None
    data: dict[str, object] | None = None
    error: str | None = None
    try:
        resp = await _get_client().post(
            f"{config.UPSTREAM}/chat/completions", json=payload, headers=headers
        )
        status = resp.status_code
        try:
            data = resp.json()
        except Exception:  # noqa: BLE001
            data = None
            error = f"upstream non-JSON response: {resp.text[:200]}"
            return JSONResponse(
                status_code=502,
                content={"error": {"message": error, "type": "upstream_error"}},
            )
        resp.raise_for_status()
        return JSONResponse(status_code=resp.status_code, content=data)
    except httpx.HTTPStatusError as exc:
        # 上游 4xx/5xx：原样把错误体回给调用方
        status = exc.response.status_code
        try:
            return JSONResponse(status_code=status, content=exc.response.json())
        except Exception:  # noqa: BLE001
            error = f"HTTP {status}: {exc.response.text[:200]}"
            return JSONResponse(
                status_code=status,
                content={"error": {"message": error, "type": "upstream_error"}},
            )
    except Exception as exc:  # noqa: BLE001
        error = f"{type(exc).__name__}: {exc}"
        logger.error("上游调用失败：%s", error)
        return JSONResponse(
            status_code=502,
            content={"error": {"message": error, "type": "upstream_error"}},
        )
    finally:
        capture.save(
            payload,
            response=data,
            status=status,
            latency_ms=(time.perf_counter() - started) * 1000,
            error=error,
        )
