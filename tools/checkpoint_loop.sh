#!/usr/bin/env bash
# Commit whatever has changed, every 15 minutes.
#
# Separate from loop.sh on purpose. loop.sh's cadence is set by Kaggle, not by
# git: each of its turns polls kernel status, and Kaggle bills that against a
# quota, so halving its sleep to get more frequent commits would spend GPU hours
# on nothing. This loop touches no network — it only checkpoints the working
# tree, including tools/ and src/, which loop.sh never stages.
#
# It commits but does not push. loop.sh pushes every 30 minutes and carries these
# commits with it, so nothing here reaches the remote on its own.
cd "$(dirname "$0")/.." || exit 1
LOG="logs/checkpoint.log"
mkdir -p logs
echo "=== checkpoint loop started $(date '+%F %H:%M') ===" >> "$LOG"

while true; do
  sleep 900
  # Never race loop.sh's own commit, and never interrupt an index operation.
  if [ -e .git/index.lock ]; then
    echo "$(date '+%F %H:%M') index locked, skipping" >> "$LOG"
    continue
  fi
  if [ -z "$(git status --porcelain)" ]; then
    continue
  fi
  files=$(git status --porcelain | wc -l | tr -d ' ')
  git add -A
  if git diff --cached --quiet; then
    continue
  fi
  git commit -q -m "Checkpoint: $files path(s) changed

Written by tools/checkpoint_loop.sh. Cadence only — this commit asserts nothing
about whether the work in it is finished.

Co-Authored-By: Claude Opus 5 (1M context) <noreply@anthropic.com>" \
    && echo "$(date '+%F %H:%M') committed $files path(s)" >> "$LOG"
done
