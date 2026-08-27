#!/usr/bin/env bash
# Continuous loop: runs until the experiment is finished or something needs a person.
#
# One turn every 30 minutes, matching the standing commit rule so each turn
# produces at most one checkpoint. tick.py follows whichever phase is open and
# crosses phase boundaries on its own, auditing each one as it closes.
#
# It stops only on exit code 3, which covers exactly three cases:
#   - a phase past 20% failures (inconclusive under PROTOCOL §5)
#   - a phase whose audit could not reach a verdict
#   - all three phases complete: the final interpretation is not the loop's to write
cd "$(dirname "$0")/.." || exit 1
LOG="logs/loop.log"
mkdir -p logs
echo "=== loop started $(date '+%F %H:%M') ===" >> "$LOG"

while true; do
  echo "--- turn $(date '+%F %H:%M') ---" >> "$LOG"
  python3 tools/tick.py --limit 2 >> "$LOG" 2>&1
  if [ $? -eq 3 ]; then
    echo "=== loop stopped by design at $(date '+%F %H:%M') ===" >> "$LOG"
    tail -20 "$LOG" > logs/STOPPED.txt
    exit 0
  fi
  sleep 1800
done
