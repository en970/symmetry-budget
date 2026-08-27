#!/usr/bin/env bash
# PostToolUse hook: validate any result file the moment it is written.
# Reads the hook payload on stdin, acts only on results/*.json, stays silent otherwise.
set -uo pipefail
payload=$(cat)
path=$(printf '%s' "$payload" | python3 -c 'import json,sys; print(json.load(sys.stdin).get("tool_input",{}).get("file_path",""))' 2>/dev/null)
case "$path" in
  */results/*.json)
    cd "$(dirname "$0")/.." || exit 0
    if ! out=$(python3 tools/validate_result.py "$path" 2>&1); then
      echo "Result rejected by PROTOCOL validation — fix the run, do not edit the file:" >&2
      echo "$out" >&2
      exit 2   # exit 2 feeds stderr back to the model as a blocking error
    fi
    ;;
esac
exit 0
