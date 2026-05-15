#!/usr/bin/env bash
# Claude Code PostToolUse hook: runs the skill-sync check when an edit
# touches either the ModelArch schema or the adding-a-model SKILL.md.
#
# Exit codes:
#   0 — not relevant or check passed
#   2 — drift detected; stderr is fed back to Claude as feedback

set -euo pipefail

INPUT=$(cat)

# Extract tool_input.file_path. python3 is on every macOS/Linux dev box;
# avoiding the jq dependency.
FILE_PATH=$(printf '%s' "$INPUT" | python3 -c \
  'import json,sys; d=json.load(sys.stdin); print(d.get("tool_input",{}).get("file_path",""))' \
  2>/dev/null || true)

case "$FILE_PATH" in
  */calc/src/engine/types.ts|*/.claude/skills/adding-a-model/SKILL.md)
    exec node "$(dirname "$0")/check-skill-sync.mjs"
    ;;
  *)
    exit 0
    ;;
esac
