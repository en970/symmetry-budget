#!/usr/bin/env bash
# Overnight loop. One turn every 30 minutes, matching the standing commit rule so
# each turn produces at most one checkpoint.
#
# Stops on exit code 3, which tick.py returns for the two cases that need a
# human: a phase that closed (the auditor must run before any prose exists about
# the numbers) and a phase past 20% failures (inconclusive under PROTOCOL §5).
cd "$(dirname "$0")/.." || exit 1
PHASE="${1:-1}"
LOG="logs/loop.log"
echo "=== loop started $(date '+%F %H:%M') phase=$PHASE ===" >> "$LOG"

for turn in $(seq 1 48); do          # 48 turns x 30 min = 24 h ceiling
  echo "--- turn $turn $(date '+%F %H:%M') ---" >> "$LOG"
  python3 tools/tick.py --phase "$PHASE" --limit 2 >> "$LOG" 2>&1
  rc=$?
  if [ $rc -eq 3 ]; then
    echo "=== loop stopped by design (exit 3) at $(date '+%F %H:%M') ===" >> "$LOG"
    exit 0
  fi
  sleep 1800
done
echo "=== loop hit its 24h ceiling $(date '+%F %H:%M') ===" >> "$LOG"
