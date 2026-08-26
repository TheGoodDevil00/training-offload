#!/usr/bin/env bash
set -e
cd "$(dirname "$0")"
echo -e "\033]0;STEP 5/5 - Package results\007"
bash scripts/5_package_results.sh "$@"
RC=$?
echo ""
if [ $RC -eq 0 ]; then echo "RESULT: SUCCESS"; else echo "RESULT: FAILED - read the messages above"; fi
read -p "Press Enter to continue..."
exit $RC
