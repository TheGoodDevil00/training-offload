#!/usr/bin/env bash
set -e
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_ROOT"

echo -e "\033]0;Overnight Drone Model Tournament (keep window open)\007"
bash "$REPO_ROOT/scripts/tournament.sh" "$@"
RC=$?
echo ""
if [ $RC -eq 0 ]; then echo "RESULT: SUCCESS"; else echo "RESULT: FAILED - read the messages above"; fi
read -p "Press Enter to continue..."
exit $RC
