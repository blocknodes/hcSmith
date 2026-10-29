#!/usr/bin/env bash
# hcProxy 启动脚本：OpenAI-format 兼容代理（捕获 + 透传真实 vllm）
set -euo pipefail
cd "$(dirname "$0")"

PORT="${HC_PROXY_PORT:-8080}"
HOST="${HC_PROXY_HOST:-0.0.0.0}"

# 先杀掉占用同一端口的旧进程（往往是上次没退干净的 uvicorn）
if command -v lsof >/dev/null 2>&1; then
  pids="$(lsof -t -iTCP:"$PORT" -sTCP:LISTEN 2>/dev/null || true)"
  if [ -n "$pids" ]; then
    echo "[run.sh] 端口 $PORT 被以下进程占用，先结束：$pids"
    kill $pids 2>/dev/null || true
    sleep 1
  fi
fi

# 统一日志文件：所有输出(stdout+stderr)都追加到这里。可用 HC_PROXY_LOG_FILE 覆盖。
LOG_FILE="${HC_PROXY_LOG_FILE:-run.log}"
exec uvicorn app.main:app --host "$HOST" --port "$PORT" 2>&1 | tee -a "$LOG_FILE"
