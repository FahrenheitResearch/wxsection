#!/bin/bash
# HRRR Cross-Section Dashboard - Production Startup
# Starts: dashboard + auto-update daemon + cloudflare tunnel
#
# Usage: ./deploy/run_production.sh
#   Stop: ./deploy/run_production.sh stop

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_DIR"

PORT=${PORT:-5559}
MAX_HOURS=${MAX_HOURS:-18}
LOG_DIR="/tmp"

# Admin key for archive access — change this to your own secret
export WXSECTION_KEY="${WXSECTION_KEY:-}"
if [ -z "$WXSECTION_KEY" ]; then
    echo "WARNING: WXSECTION_KEY not set. Archive access will be unrestricted."
    echo "  Set it: export WXSECTION_KEY=your_secret_key"
    echo ""
fi

# Stop everything
stop_all() {
    echo "Stopping services..."
    pkill -f "unified_dashboard.py --port $PORT" 2>/dev/null && echo "  Dashboard stopped" || echo "  Dashboard not running"
    pkill -f "auto_update.py" 2>/dev/null && echo "  Auto-update stopped" || echo "  Auto-update not running"
    pkill -f "cloudflared tunnel run wxsection" 2>/dev/null && echo "  Tunnel stopped" || echo "  Tunnel not running"
}

if [ "${1:-}" = "stop" ]; then
    stop_all
    exit 0
fi

# Stop any existing instances first
stop_all
sleep 2

echo "=============================================="
echo "  wxsection.com - HRRR Cross-Section Dashboard"
echo "=============================================="
echo ""

# 1. Auto-update daemon
echo "Starting auto-update daemon (interval=2min, max F${MAX_HOURS})..."
nohup python tools/auto_update.py --interval 2 --max-hours "$MAX_HOURS" \
    > "$LOG_DIR/auto_update.log" 2>&1 &
echo "  PID: $! (log: $LOG_DIR/auto_update.log)"

# 2. Dashboard
echo "Starting dashboard on port $PORT..."
nohup python tools/unified_dashboard.py \
    --port "$PORT" \
    --preload 2 \
    --production \
    > "$LOG_DIR/dashboard.log" 2>&1 &
echo "  PID: $! (log: $LOG_DIR/dashboard.log)"

# 3. Cloudflare tunnel
echo "Starting Cloudflare tunnel (wxsection.com)..."
nohup cloudflared tunnel run wxsection \
    > "$LOG_DIR/cloudflared.log" 2>&1 &
echo "  PID: $! (log: $LOG_DIR/cloudflared.log)"

echo ""
echo "All services started."
echo "  Local:  http://localhost:$PORT"
echo "  Public: https://wxsection.com"
echo ""
echo "Logs:"
echo "  tail -f $LOG_DIR/dashboard.log"
echo "  tail -f $LOG_DIR/auto_update.log"
echo "  tail -f $LOG_DIR/cloudflared.log"
echo ""
echo "Stop all: $0 stop"
