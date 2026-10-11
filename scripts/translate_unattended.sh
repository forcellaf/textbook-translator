#!/usr/bin/env bash
# No-API mode launcher: translate one book with headless Claude Code.
#
#     scripts/translate_unattended.sh data/input/physics.pdf [book_name]
#
# Runs the /translate-book skill (.claude/skills/translate-book/SKILL.md) as
# the orchestrator -- Claude Opus 5.5 at high effort -- on a Claude Code
# subscription. No Anthropic API key is used: stage 2's LLM calls go through
# the exchange (LLM_BACKEND=exchange) and every chunk is translated by a
# chunk-translator subagent.
#
# The run settings overlay (.claude/run-settings.json) denies edits to the
# tool itself (src/, assets/, tests/, scripts/, .claude/) for the duration of
# the run, and raises the output-token cap so one chunk-translator reply fits.
# It is an overlay, not the shared settings, so development sessions can
# still edit code.
#
# The session transcript (stream-json) goes to logs/; the deliverables to
# data/output/<book>/.

set -euo pipefail

if [[ $# -lt 1 || $# -gt 2 ]]; then
  echo "usage: $0 <pdf> [book_name]" >&2
  exit 2
fi

cd "$(dirname "$0")/.."

pdf="$1"
if [[ ! -f "$pdf" ]]; then
  echo "error: $pdf not found" >&2
  exit 1
fi
stem="$(basename "$pdf")"
stem="${stem%.*}"
book="${2:-$(printf '%s' "$stem" | tr '[:upper:]' '[:lower:]' | tr ' ' '_')}"

mkdir -p logs
transcript="logs/translate-${book}-$(date +%Y%m%d-%H%M%S).jsonl"
echo "translating $pdf as '$book'; transcript: $transcript" >&2

# A key in the environment would make Claude Code bill the API instead of
# the subscription, and make the pipeline pick the API backend.
unset ANTHROPIC_API_KEY
export LLM_BACKEND=exchange

claude -p "/translate-book $pdf $book" \
  --model claude-opus-5-5 \
  --effort high \
  --settings .claude/run-settings.json \
  --permission-mode dontAsk \
  --output-format stream-json \
  --verbose \
  > "$transcript"

status=$?
echo "done (exit $status); output in data/output/$book/" >&2
exit $status
