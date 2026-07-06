#!/usr/bin/env bash
#
# Stop the FDE Email Agent.
#
#   ./stop.sh         Normal shutdown: stop the worker + Slack listener only.
#                     The Gmail watch is LEFT ACTIVE on purpose, so mail that
#                     arrives while you're off keeps queuing in Pub/Sub and is
#                     pulled on the next start. Postgres/Redis/Colima are left
#                     running (they also stop on laptop shutdown).
#
#   ./stop.sh --all   Full teardown: also stop the Gmail watch, bring docker
#                     compose down (data volumes are KEPT), and stop Colima.
set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"
export CLOUDSDK_CONFIG="$HOME/.gcloud"
PY="$SCRIPT_DIR/.venv/bin/python"
MODE="${1:-}"

echo "==> Stopping worker + Slack listener…"
pkill -f "app.worker"         2>/dev/null && echo "   worker stopped"   || echo "   worker not running"
pkill -f "app.slack_approval" 2>/dev/null && echo "   listener stopped" || echo "   listener not running"

if [ "$MODE" = "--all" ]; then
  echo "==> Stopping Gmail watch (Pub/Sub will NOT queue new mail until you start again)…"
  "$PY" -m app.gmail_client stop || echo "   (watch stop failed or already stopped)"
  echo "==> docker compose down (keeping data volumes)…"
  docker compose down
  echo "==> Stopping Colima…"
  colima stop || true
  echo "Full teardown complete."
else
  echo
  echo "Processes stopped. Gmail watch LEFT ACTIVE — mail keeps queuing in Pub/Sub"
  echo "while you're off and will be pulled on the next ./start.sh."
  echo "For a full teardown (also stops the watch + docker + Colima): ./stop.sh --all"
fi
