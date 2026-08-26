#!/usr/bin/env bash
set -e
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_ROOT"

echo -e "\033]0;QUICK TEST - verify pipeline end to end\007"
bash "$REPO_ROOT/scripts/4_train.sh" -SmokeTest "$@"
RC=$?
echo ""
if [ $RC -eq 0 ]; then echo "RESULT: SUCCESS"; else echo "RESULT: FAILED - read the messages above"; fi
read -p "Press Enter to continue..."
exit $RC
