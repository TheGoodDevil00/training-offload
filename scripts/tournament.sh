#!/usr/bin/env bash
# Overnight Drone-Footage Model Tournament (Linux)

source "$(dirname "${BASH_SOURCE[0]}")/_common.sh"

write_header "OVERNIGHT DRONE-FOOTAGE MODEL TOURNAMENT (RTX 4050 6GB)"

PY=$(get_venv_py)

# Pass all incoming flags to tournament_runner.py
$PY training/tournament_runner.py "$@"
RC=$?

if [ $RC -eq 0 ]; then
    write_good "Tournament execution completed successfully."
    write_info "Check results in runs/tournament/LEADERBOARD.md"
else
    fail "Tournament execution failed with exit code $RC."
fi
