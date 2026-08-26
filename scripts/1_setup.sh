#!/usr/bin/env bash
# Step 1 - one-time machine setup: GPU check, Python, dependencies, sanity checks.

source "$(dirname "${BASH_SOURCE[0]}")/_common.sh"
write_header "STEP 1 of 5 - Install (Linux one-time setup)"

# ---------------------------------------------------------------- GPU + driver
write_info "Checking for an NVIDIA GPU..."
assert_nvidia_gpu

# ---------------------------------------------------------------- Base tools
for cmd in curl unzip zip; do
    if ! command -v $cmd &>/dev/null; then
        if grep -qi ubuntu /etc/os-release 2>/dev/null; then
            fail "Missing $cmd. Please run:  sudo apt update && sudo apt install $cmd"
        elif grep -qi arch /etc/os-release 2>/dev/null; then
            fail "Missing $cmd. Please run:  sudo pacman -S $cmd"
        else
            fail "Missing $cmd. Please install it using your package manager."
        fi
    fi
done

# ---------------------------------------------------------------- Python
if [ -f "$REPO_ROOT/.venv/bin/python" ]; then
    write_good "Existing Python environment found - skipping installation."
else
    SYS_PY=""
    for cmd in python3.12 python3.13 python3.11 python3; do
        if command -v $cmd &>/dev/null; then
            V=$($cmd -c "import sys; print('%d.%d' % sys.version_info[:2])")
            if [[ "$V" == "3.10" || "$V" == "3.11" || "$V" == "3.12" || "$V" == "3.13" ]]; then
                SYS_PY=$(command -v $cmd)
                break
            fi
        fi
    done

    if [ -z "$SYS_PY" ]; then
        if grep -qi ubuntu /etc/os-release 2>/dev/null; then
            fail "Suitable Python 3.10-3.13 not found.\nPlease run:  sudo apt update && sudo apt install python3.12-venv python3.12-dev"
        elif grep -qi arch /etc/os-release 2>/dev/null; then
            fail "Suitable Python 3.10-3.13 not found.\nPlease run:  sudo pacman -S python"
        else
            fail "Suitable Python 3.10-3.13 not found. Please install Python 3.12."
        fi
    fi
    write_good "Using Python at $SYS_PY"

    write_info "Creating isolated environment (.venv)..."
    $SYS_PY -m venv "$REPO_ROOT/.venv" || {
        if grep -qi ubuntu /etc/os-release 2>/dev/null; then
            fail "venv creation failed. You might be missing python3-venv.\nPlease run:  sudo apt install python3-venv"
        else
            fail "Failed to create python virtual environment."
        fi
    }
fi

# ---------------------------------------------------------------- Dependencies
PY=$(get_venv_py)
write_info "Installing PyTorch (CUDA), Ultralytics YOLO and export tools..."
write_info "(This downloads several GB and can take 10-30 minutes. Only happens once.)"
$PY -m pip install --upgrade pip -q
$PY -m pip install -r requirements.txt

# ---------------------------------------------------------------- Sanity checks
write_info "Verifying that PyTorch can actually see the GPU..."
cat << 'EOF' > "$REPO_ROOT/.chk.py"
import torch
assert torch.cuda.is_available(), 'torch.cuda.is_available() is False'
print('CUDA_OK ' + torch.cuda.get_device_name(0))
props = torch.cuda.get_device_properties(0)
print('VRAM_GB %.1f' % (props.total_memory / 2**30))
EOF

if ! out=$($PY "$REPO_ROOT/.chk.py"); then
    rm -f "$REPO_ROOT/.chk.py"
    fail "PyTorch is installed but CANNOT use your GPU.\nFixes that usually work:\n  1. Update your NVIDIA driver (need 580 or newer), then reboot.\n  2. Reboot once after driver install before giving up.\nThen run 1-Install.sh again."
fi
rm -f "$REPO_ROOT/.chk.py"
write_good "$(echo "$out" | tr '\n' ' | ')"

write_info "Pre-downloading the base model weights (yolo11n.pt)..."
$PY -c "from ultralytics import YOLO; YOLO('yolo11n.pt'); print('weights_ok')" > /dev/null

write_header "STEP 1 FINISHED - everything installed. Next: run ./2-Download-Dataset.sh"
