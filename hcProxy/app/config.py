"""hcProxy 配置：全部走环境变量，其余为常量默认值。

环境变量：
  HC_PROXY_HOST / HC_PROXY_PORT      监听地址（默认 0.0.0.0:8091）
  HC_PROXY_UPSTREAM                  上游真实 vllm 网关（默认 juagent 网关）
  HC_PROXY_UPSTREAM_KEY              上游鉴权 key（默认取 HC_LLM_API_KEY）
  HC_PROXY_DATA_DIR                  捕获落盘目录（默认 <项目>/data/captures）
  HC_PROXY_CAPTURE                   设 "0" 关闭落盘（只代理不捕获）
  HC_PROXY_LOG_STDOUT                设 "0" 关闭每次调用摘要打屏（默认开）
  HC_PROXY_STATS_INTERVAL            打屏统计汇总的间隔秒数（默认 60；设 0 每次调用都打汇总）
  HC_PROXY_TIMEOUT                   上游超时秒数（默认 120）
"""
from __future__ import annotations

import os
from pathlib import Path

# ---- 监听 ----
HOST = os.environ.get("HC_PROXY_HOST", "0.0.0.0")
PORT = int(os.environ.get("HC_PROXY_PORT", "8091"))

# ---- 上游真实 vllm server（OpenAI-format）----
# 默认指回 juagent 网关；切真实 vllm 时改这一个变量即可。
UPSTREAM = os.environ.get(
    "HC_PROXY_UPSTREAM", "http://10.19.96.219:4003/v1"
).rstrip("/")
API_KEY = os.environ.get("HC_PROXY_UPSTREAM_KEY", os.environ.get("HC_LLM_API_KEY", ""))

# ---- 超时 ----
TIMEOUT = float(os.environ.get("HC_PROXY_TIMEOUT", "120"))

# ---- 捕获落盘 ----
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = Path(os.environ.get("HC_PROXY_DATA_DIR", PROJECT_ROOT / "data" / "captures"))
#: 设 "0" 关闭落盘（代理仍工作）
CAPTURE = os.environ.get("HC_PROXY_CAPTURE", "1") not in {"0", "false", "off", "no"}

# ---- 调用摘要打屏 ----
#: 默认开：每次调用在屏幕打一行摘要（app/purpose/status/latency/query 预览）。设 "0" 关闭。
LOG_STDOUT = os.environ.get("HC_PROXY_LOG_STDOUT", "1") not in {"0", "false", "off", "no"}

# ---- 调用时长统计打屏 ----
#: 每次调用都累计 latency，并打印累计均值/中位/峰值。间隔秒数（默认 60s 打一行周期汇总，
#: 且每条调用日志行尾带累计统计便于 grep）；设 "0" 关闭周期汇总（只保留调用行尾统计）。
STATS_INTERVAL = float(os.environ.get("HC_PROXY_STATS_INTERVAL", "60"))

# ---- 捕获文件按 app 标记分流 ----
# 有 metadata.app 的请求落到 captures/<app>/，裸请求落到 captures/_raw/。
RAW_APP = "_raw"
