"""捕获落盘：每次 LLM 调用一行 JSONL，按天分目录。

布局：data/captures/<app标记>/YYYY-MM-DD.jsonl
  - <app标记> = metadata.app（如 hcAgent / hcTools）；无 metadata 的裸请求 → "_raw"
  - 每行一个 JSON 对象：
      {"ts": ..., "request": {...含 metadata...}, "response": {...}, "status": 200,
       "latency_ms": 123.4, "error": null}
  - 同一天同进程只开一次文件句柄（append），行内自带 ts，天然可 grep。

尽力而为：落盘抛异常只记 warning，绝不影响代理主链路。
"""
from __future__ import annotations

import json
import logging
import os
import time
from datetime import datetime
from pathlib import Path
from typing import Any

from . import config

logger = logging.getLogger("hcProxy.capture")

# {app标记: (date, file_handle)} —— 同一 app 同一天复用一个句柄
_handles: dict[str, tuple[str, Any]] = {}


# ---------------------------------------------------------------------------
# 调用时长统计（打屏用，独立于落盘）
# ---------------------------------------------------------------------------
#: 按 (app, purpose) 分桶累计 latency；结构 {app: {purpose: _LatencyAcc}}
_LATENCY_ACC: dict[str, dict[str, "_LatencyAcc"]] = {}
_last_stats_ts = time.monotonic()  # 上一次周期汇总时间（模块加载时固定）


class _LatencyAcc:
    """一个 (app,purpose) 桶的累计：计数/总和/平方和/最大/窗口内原始值（算中位）。"""

    __slots__ = ("count", "total_sum", "sum_sq", "max", "samples")

    def __init__(self) -> None:
        self.count = 0
        self.total_sum = 0.0
        self.sum_sq = 0.0
        self.max = 0.0
        self.samples: list[float] = []

    def add(self, ms: float) -> None:
        self.count += 1
        self.total_sum += ms
        self.sum_sq += ms * ms
        if ms > self.max:
            self.max = ms
        self.samples.append(ms)

    def report(self) -> dict[str, Any]:
        n = self.count
        if n == 0:
            return {"n": 0}
        mean = self.total_sum / n
        ddof = n - 1 if n > 1 else 1  # 样本标准差；n=1 时不除 0
        std = (max(0.0, self.sum_sq / n - mean * mean)) ** 0.5
        s = sorted(self.samples)
        mid = n // 2
        median = s[mid] if n % 2 else (s[mid - 1] + s[mid]) / 2
        return {"count": n, "mean_ms": round(mean, 1), "median_ms": round(median, 1),
                "max_ms": round(self.max, 1), "std_ms": round(std, 1)}


def _stats_key(tag: str, purpose: str) -> str:
    return f"{tag}:{purpose or '-'}"


def reset_stats() -> None:
    """清空累计统计（测试用）。"""
    global _last_stats_ts
    _LATENCY_ACC.clear()
    _last_stats_ts = time.monotonic()


def _acc_for(tag: str, purpose: str) -> _LatencyAcc:
    d = _LATENCY_ACC.setdefault(tag, {})
    key = _stats_key(tag, purpose)
    acc = d.get(key)
    if acc is None:
        acc = _LatencyAcc()
        d[key] = acc
    return acc


def _stats_tail(tag: str, purpose: str) -> str:
    """单条调用日志行尾的累计统计：count/avg/max/median。"""
    acc = _LATENCY_ACC.get(tag, {}).get(_stats_key(tag, purpose))
    if acc is None or acc.count == 0:
        return "0 次"
    return (f"{acc.count}次 均{acc.total_sum / acc.count:.0f}ms "
            f"中{acc.report()['median_ms']:.0f}ms 峰{acc.max:.0f}ms")


def _log_periodic_stats() -> None:
    """达到周期间隔时，打一行（app, purpose）× latency 汇总表。"""
    global _last_stats_ts
    now = time.monotonic()
    if config.STATS_INTERVAL <= 0 or now - _last_stats_ts < config.STATS_INTERVAL:
        return
    _last_stats_ts = now
    rows: list[str] = []
    for tag in sorted(_LATENCY_ACC):
        for key, acc in sorted(_LATENCY_ACC[tag].items()):
            r = acc.report()
            rows.append(f"{key}[{r['count']:>4}次 均{r['mean_ms']:.0f}ms "
                        f"中{r['median_ms']:.0f}ms 峰{r['max_ms']:.0f}ms "
                        f"σ{r['std_ms']:.0f}ms]")
    if rows:
        logger.info("累计LLM调用时长 | %s", " ".join(rows))


def app_tag(payload: dict[str, Any]) -> str:
    """从请求体提取 app 标记（metadata.app），无则归入 _raw。"""
    metadata = payload.get("metadata")
    if isinstance(metadata, dict):
        app = str(metadata.get("app", "")).strip()
        if app:
            # 防路径穿越：只留安全字符
            return "".join(c if c.isalnum() or c in "-_." else "_" for c in app)[:64]
    return config.RAW_APP


def _last_user_text(payload: dict[str, Any]) -> str:
    """取最后一条 user 消息文本做预览（没有则空串）。"""
    messages = payload.get("messages")
    if isinstance(messages, list):
        for msg in reversed(messages):
            if isinstance(msg, dict) and msg.get("role") == "user":
                content = msg.get("content", "")
                if isinstance(content, str):
                    return content
                if isinstance(content, list):  # 多模态：拼接 text 片段
                    return " ".join(
                        p.get("text", "") for p in content
                        if isinstance(p, dict) and p.get("type") == "text"
                    )
    return ""


def _log_summary(
    payload: dict[str, Any],
    *,
    tag: str,
    status: int | None,
    latency_ms: float,
    error: str | None,
) -> None:
    """每次调用打一行屏幕摘要：app/purpose/status/latency + 累计统计 + query 预览。"""
    metadata = payload.get("metadata")
    purpose = ""
    if isinstance(metadata, dict):
        purpose = str(metadata.get("purpose", "") or metadata.get("op", ""))
    query = _last_user_text(payload).replace("\n", " ")
    if len(query) > 60:
        query = query[:60] + "…"
    logger.info(
        "调用 app=%s purpose=%s status=%s %.0fms [%s]%s | %s",
        tag, purpose or "-", status, latency_ms,
        _stats_tail(tag, purpose),
        f" error={error}" if error else "", query,
    )


def _file_for(tag: str) -> Any:
    """拿到该 app 今天对应的 append 文件句柄（跨天自动切换）。"""
    today = datetime.now().strftime("%Y-%m-%d")
    entry = _handles.get(tag)
    if entry is not None and entry[0] == today:
        return entry[1]
    if entry is not None:
        try:  # 跨天：关旧句柄
            entry[1].close()
        except Exception:  # noqa: BLE001
            pass
    config.DATA_DIR.mkdir(parents=True, exist_ok=True)
    path = config.DATA_DIR / tag / f"{today}.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    fh = open(path, "a", encoding="utf-8")  # noqa: SIM115
    _handles[tag] = (today, fh)
    logger.info("捕获文件：%s", path)
    return fh


def save(
    payload: dict[str, Any],
    *,
    response: dict[str, Any] | None,
    status: int | None,
    latency_ms: float,
    error: str | None = None,
) -> None:
    """落一条捕获记录。request 原样保存（含 metadata）。"""
    tag = app_tag(payload)
    # 累计调用时长统计（打屏用）
    purpose = ""
    if isinstance(payload.get("metadata"), dict):
        purpose = str(payload["metadata"].get("purpose", "") or payload["metadata"].get("op", ""))
    _acc_for(tag, purpose).add(latency_ms)
    if config.LOG_STDOUT:  # 打屏独立于落盘：CAPTURE=0 时屏幕摘要仍出
        _log_summary(payload, tag=tag, status=status, latency_ms=latency_ms, error=error)
        _log_periodic_stats()
    if not config.CAPTURE:
        return
    record = {
        "ts": time.strftime("%Y-%m-%d %H:%M:%S"),
        "request": payload,
        "response": response,
        "status": status,
        "latency_ms": round(latency_ms, 1),
        "error": error,
    }
    try:
        fh = _file_for(tag)
        fh.write(json.dumps(record, ensure_ascii=False) + "\n")
        fh.flush()
        os.fsync(fh.fileno())
    except Exception as exc:  # noqa: BLE001
        logger.warning("捕获落盘失败（不影响代理）：%s", exc)
