#!/usr/bin/env bash
#
# Start the FDE Email Agent on this laptop:
#   1. export CLOUDSDK_CONFIG (gcloud config lives in ~/.gcloud here)
#   2. start Colima if it isn't running
#   3. bring up Postgres + Redis (docker compose) and wait for Postgres
#   4. re-register the Gmail watch (refreshes the ~7-day expiry every start)
#   5. launch the worker + Slack listener (skips any already running)
#
# Safe to run repeatedly. Logs go to worker.log / slack_listener.log.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

export CLOUDSDK_CONFIG="$HOME/.gcloud"
PY="$SCRIPT_DIR/.venv/bin/python"

echo "==> FDE Email Agent: starting"

# 1) Colima ------------------------------------------------------------------
if colima status >/dev/null 2>&1; then
  echo "==> Colima already running"
else
  echo "==> Starting Colima…"
  colima start
fi

# 2) Postgres + Redis --------------------------------------------------------
echo "==> docker compose up -d (Postgres + Redis)…"
docker compose up -d

echo -n "==> Waiting for Postgres to be healthy"
hs="starting"
for _ in $(seq 1 30); do
  hs="$(docker inspect --format '{{.State.Health.Status}}' fde_postgres 2>/dev/null || echo starting)"
  [ "$hs" = "healthy" ] && { echo " ✓"; break; }
  echo -n "."
  sleep 2
done
if [ "$hs" != "healthy" ]; then
  echo; echo "ERROR: Postgres did not become healthy" >&2; exit 1
fi

# 3) Re-register the Gmail watch (refreshes 7-day expiry) --------------------
echo "==> Re-registering Gmail watch…"
"$PY" -m app.gmail_client watch

# 4) Launch worker + Slack listener ------------------------------------------
start_proc() {
  local label="$1" module="$2" log="$3"
  if pgrep -f "$module" >/dev/null 2>&1; then
    echo "==> $label already running (pid $(pgrep -f "$module" | tr '\n' ' '))"
  else
    echo "==> Starting $label → $log"
    CLOUDSDK_CONFIG="$HOME/.gcloud" PYTHONUNBUFFERED=1 \
      nohup "$PY" -m "$module" >"$log" 2>&1 &
    sleep 1
  fi
}

start_proc "Slack listener" "app.slack_approval" "$SCRIPT_DIR/slack_listener.log"
start_proc "Gmail worker"   "app.worker"         "$SCRIPT_DIR/worker.log"

echo
echo "==> Up. Verify with:"
echo "    pgrep -fl 'app.worker|app.slack_approval'"
echo "    tail -f worker.log         # watch ingest + startup catch-up"
echo "    tail -n 30 slack_listener.log"
