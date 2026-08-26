#!/usr/bin/env bash
set -e
cd "$(dirname "$0")"
echo -e "\033]0;QUICK TEST - verify pipeline end to end\007"
bash scripts/4_train.sh "$@" -SmokeTest
RC=$?
echo ""
if [ $RC -eq 0 ]; then echo "RESULT: SUCCESS"; else echo "RESULT: FAILED - read the messages above"; fi
read -p "Press Enter to continue..."
exit $RC
