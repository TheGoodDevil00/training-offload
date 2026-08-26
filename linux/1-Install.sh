#!/usr/bin/env bash
set -e
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_ROOT"

echo -e "\033]0;STEP 1/5 - Install (one-time setup)\007"
bash "$REPO_ROOT/scripts/1_setup.sh" "$@"
RC=$?
echo ""
if [ $RC -eq 0 ]; then echo "RESULT: SUCCESS"; else echo "RESULT: FAILED - read the messages above"; fi
read -p "Press Enter to continue..."
exit $RC
