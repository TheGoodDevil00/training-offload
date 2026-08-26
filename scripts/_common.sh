#!/usr/bin/env bash
# Shared helpers for the training-offload step scripts.

set -e
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

write_header() {
    echo -e "\n\033[1;36m================================================================\033[0m"
    echo -e "\033[1;36m  $1\033[0m"
    echo -e "\033[1;36m================================================================\033[0m"
}

write_info() { echo -e "\033[0;36m[i] $1\033[0m"; }
write_good() { echo -e "\033[0;32m[OK] $1\033[0m"; }
write_warn() { echo -e "\033[1;33m[!] $1\033[0m"; }

fail() {
    echo -e "\n\033[0;31m[X] $1\033[0m\n"
    exit 1
}

get_venv_py() {
    local p="$REPO_ROOT/.venv/bin/python"
    if [ ! -f "$p" ]; then
        fail "Python environment not found. Run 1-Install.sh first."
    fi
    echo "$p"
}

assert_nvidia_gpu() {
    if ! command -v nvidia-smi &> /dev/null; then
        fail "No NVIDIA GPU driver detected (nvidia-smi not found).\nTraining needs an NVIDIA GPU. Install/update the driver, reboot, and try again."
    fi
    GPU_LINE=$(nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader | head -n 1)
    GPU_NAME=$(echo "$GPU_LINE" | cut -d, -f1 | xargs)
    GPU_DRIVER=$(echo "$GPU_LINE" | cut -d, -f2 | xargs)
    GPU_VRAM=$(echo "$GPU_LINE" | cut -d, -f3 | xargs)
    
    write_good "GPU: $GPU_NAME  (driver $GPU_DRIVER, VRAM $GPU_VRAM)"
    
    DRIVER_MAJOR=$(echo "$GPU_DRIVER" | cut -d. -f1)
    if [ -z "$DRIVER_MAJOR" ] || [ "$DRIVER_MAJOR" -lt 580 ]; then
        fail "This GPU driver ($GPU_DRIVER) is too old for the PyTorch build used here (needs CUDA 13, i.e. driver 580 or newer).\nFix: install the latest driver from NVIDIA, reboot, and run this step again."
    fi
}
