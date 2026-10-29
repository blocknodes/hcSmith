"""hcProxy 测试：用 FastAPI TestClient + 桩上游（不依赖真实网关）。

覆盖：透传、捕获落盘（带/不带 metadata）、上游错误透传、健康检查。
"""
from __future__ import annotations

import json
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

import pytest
from fastapi.testclient import TestClient

from app import capture, config
from app.capture import _acc_for, _LATENCY_ACC, _log_summary, _stats_tail, reset_stats
from app.main import app


# --------------------------------------------------------------------------
# 桩上游：一个固定返回 OpenAI-format 响应的本地 HTTP 服务
# --------------------------------------------------------------------------
class _Upstream(BaseHTTPRequestHandler):
    def do_POST(self):  # noqa: N802
        length = int(self.headers.get("Content-Length", 0))
        body = json.loads(self.rfile.read(length) or b"{}")
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        resp = {"id": "chatcmpl-stub", "object": "chat.completion",
                "choices": [{"index": 0, "message": {"role": "assistant",
                                                     "content": f"echo:{body.get('model')}"},
                             "finish_reason": "stop"}]}
        self.wfile.write(json.dumps(resp).encode())

    def log_message(self, *args):  # 静默
        pass


@pytest.fixture(scope="module")
def upstream():
    server = HTTPServer(("127.0.0.1", 0), _Upstream)
    port = server.server_address[1]
    t = threading.Thread(target=server.serve_forever, daemon=True)
    t.start()
    # 把代理上游指到桩服务
    config.UPSTREAM = f"http://127.0.0.1:{port}"
    yield port
    server.shutdown()


@pytest.fixture()
def tmp_data(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "DATA_DIR", tmp_path)
    monkeypatch.setattr(capture, "_handles", {})
    return tmp_path


@pytest.fixture()
def client():
    return TestClient(app, raise_server_exceptions=False)


META = {"app": "hcAgent", "purpose": "plan", "same_turn": "true", "op": "split"}


def test_healthz(client):
    resp = client.get("/healthz")
    assert resp.status_code == 200
    assert resp.json()["status"] == "ok"


def test_models(client):
    resp = client.get("/v1/models")
    assert resp.status_code == 200
    assert resp.json()["object"] == "list"


def test_passthrough_and_capture(client, upstream, tmp_data):
    payload = {"model": "baseline", "messages": [{"role": "user", "content": "hi"}],
               "metadata": dict(META)}
    resp = client.post("/v1/chat/completions", json=payload)
    assert resp.status_code == 200
    assert resp.json()["choices"][0]["message"]["content"] == "echo:baseline"

    f = tmp_data / "hcAgent" / f"{capture.datetime.now().strftime('%Y-%m-%d')}.jsonl"
    assert f.exists(), "hcAgent 标记目录应落盘"
    record = json.loads(f.read_text(encoding="utf-8").strip())
    assert record["request"]["metadata"]["app"] == "hcAgent"
    assert record["status"] == 200
    assert record["error"] is None
    assert "chatcmpl" in json.dumps(record["response"], ensure_ascii=False)


def test_capture_raw_without_metadata(client, upstream, tmp_data):
    payload = {"model": "baseline", "messages": [{"role": "user", "content": "hi"}]}
    resp = client.post("/v1/chat/completions", json=payload)
    assert resp.status_code == 200
    f = tmp_data / config.RAW_APP / f"{capture.datetime.now().strftime('%Y-%m-%d')}.jsonl"
    assert f.exists(), "裸请求应落 _raw 目录"
    record = json.loads(f.read_text(encoding="utf-8").strip())
    assert "metadata" not in record["request"]


def test_invalid_json_body(client, tmp_data):
    resp = client.post("/v1/chat/completions", content=b"not-json",
                       headers={"Content-Type": "application/json"})
    assert resp.status_code == 400


def test_upstream_error_passthrough(client, upstream, tmp_data, monkeypatch):
    # 指向不存在的端口：应 502 且仍落盘
    config.UPSTREAM = "http://127.0.0.1:1"
    payload = {"model": "baseline", "messages": [{"role": "user", "content": "x"}],
               "metadata": dict(META)}  # tag=hcAgent（含 app 键，否则落 _raw）
    resp = client.post("/v1/chat/completions", json=payload)
    assert resp.status_code == 502
    f = tmp_data / "hcAgent" / f"{capture.datetime.now().strftime('%Y-%m-%d')}.jsonl"
    records = [json.loads(l) for l in f.read_text(encoding="utf-8").strip().splitlines()]
    last = records[-1]
    assert last["status"] is None and last["error"]
    config.UPSTREAM = f"http://127.0.0.1:{upstream}"  # 恢复


# --------------------------------------------------------------------------
# 调用时长统计
# --------------------------------------------------------------------------
def test_latency_stats_accumulate(client, upstream, tmp_data):
    """每次调用都累计 (app,purpose) 时长，行尾统计随调用数增长。"""
    reset_stats()
    payload = {"model": "baseline", "messages": [{"role": "user", "content": "hi"}],
               "metadata": dict(META)}  # app=hcAgent purpose=plan
    for _ in range(3):
        resp = client.post("/v1/chat/completions", json=payload)
        assert resp.status_code == 200
    acc = _LATENCY_ACC.get("hcAgent", {}).get("hcAgent:plan")
    assert acc is not None and acc.count == 3
    r = acc.report()
    assert r["count"] == 3
    assert r["mean_ms"] >= 0 and r["median_ms"] >= 0 and r["max_ms"] >= r["mean_ms"]
    tail = _stats_tail("hcAgent", "plan")
    assert "3次" in tail and "均" in tail and "峰" in tail


def test_latency_stats_periodic_summary(caplog, monkeypatch):
    """到间隔后，打一行累计LLM调用时长汇总（按 app:purpose 分桶）。"""
    reset_stats()
    import time as _time
    monkeypatch.setattr(capture, "_last_stats_ts", _time.monotonic() - 1000)  # 假装很久没打
    acc = _acc_for("hcAgent", "plan")
    acc.add(100.0); acc.add(200.0); acc.add(300.0)
    with caplog.at_level("INFO", logger="hcProxy.capture"):
        capture._log_periodic_stats()  # 间隔已到 → 应打印表行
        capture._log_periodic_stats()  # 第二次（间隔未到）→ 不打
    lines = [r.message for r in caplog.records if "累计LLM调用时长" in r.message]
    assert len(lines) == 1
    assert "hcAgent:plan" in lines[0] and "3次" in lines[0]

    # 间隔内不重复打
    capture._log_periodic_stats()
    assert len([r for r in caplog.records if "累计LLM调用时长" in r.message]) == 1
