# Step 1 - one-time machine setup: GPU check, Python, dependencies, sanity checks.
# Safe to re-run; skips work that is already done.

$ErrorActionPreference = 'Stop'
. (Join-Path $PSScriptRoot '_common.ps1')

Write-Header "STEP 1 of 5 - Install (one time only)"

# ---------------------------------------------------------------- GPU + driver
Write-Info "Checking for an NVIDIA GPU..."
Assert-NvidiaGpu

# ---------------------------------------------------------------- Python
function Get-UsablePython {
    # Probe the py launcher (preferred) and plain python on PATH.
    $probes = @(
        @{ Exe = 'py';   Pre = @('-3.12') },
        @{ Exe = 'py';   Pre = @('-3.13') },
        @{ Exe = 'py';   Pre = @('-3.11') },
        @{ Exe = 'py';   Pre = @() },
        @{ Exe = 'python'; Pre = @() }
    )
    foreach ($pr in $probes) {
        try {
            $out = & $pr.Exe @($pr.Pre + @('-c', 'import sys; print(sys.executable)')) 2>$null
            if ($LASTEXITCODE -eq 0 -and $out -and (Test-Path ($out | Select-Object -Last 1))) {
                return ($out | Select-Object -Last 1)
            }
        } catch { }
    }
    return $null
}

$venvPy = Join-Path $RepoRoot '.venv\Scripts\python.exe'
if (Test-Path $venvPy) {
    Write-Good "Existing Python environment found - skipping installation."
} else {
    $sysPy = Get-UsablePython
    if (-not $sysPy) {
        Write-Warn "No suitable Python found. Installing Python 3.12 (a few minutes, no input needed)..."
        if (Get-Command winget.exe -ErrorAction SilentlyContinue) {
            & winget install --id Python.Python.3.12 -e --silent --accept-package-agreements --accept-source-agreements
            Write-Info "winget finished (exit $LASTEXITCODE)."
        } else {
            Write-Info "winget not available - downloading the official Python installer instead..."
            $inst = Join-Path $env:TEMP 'python-3.12.10-amd64.exe'
            Run 'curl.exe' @('-fsSL', '-o', $inst, 'https://www.python.org/ftp/python/3.12.10/python-3.12.10-amd64.exe')
            Write-Info "Running the Python installer silently (takes a few minutes)..."
            Run $inst @('/quiet', 'InstallAllUsers=0', 'PrependPath=1', 'Include_test=0')
        }
        foreach ($cand in @(
            "$env:LOCALAPPDATA\Programs\Python\Python312\python.exe",
            'C:\Program Files\Python312\python.exe'
        )) {
            if (Test-Path $cand) { $sysPy = $cand; break }
        }
        if (-not $sysPy) {
            Fail "Python was installed but could not be located. Reboot and run 1-Install.bat again."
        }
    }
    $v = & $sysPy -c "import sys; print('%d.%d' % sys.version_info[:2])"
    if ([version]($v + '.0') -lt [version]'3.10.0' -or [version]($v + '.0') -ge [version]'3.14.0') {
        Fail "Found Python $v but this pipeline needs Python 3.10-3.13. Install Python 3.12 from https://www.python.org/downloads/ and run this step again."
    }
    Write-Good "Using Python $v at $sysPy"

    Write-Info "Creating isolated environment (.venv)..."
    Run $sysPy @('-m', 'venv', (Join-Path $RepoRoot '.venv'))
}

# ---------------------------------------------------------------- Dependencies
Write-Info "Installing PyTorch (CUDA), Ultralytics YOLO and export tools..."
Write-Info "(This downloads several GB and can take 10-30 minutes. Only happens once.)"
Run $venvPy @('-m', 'pip', 'install', '--upgrade', 'pip')
Run $venvPy @('-m', 'pip', 'install', '-r', (Join-Path $RepoRoot 'requirements.txt'))

# ---------------------------------------------------------------- Sanity checks
Write-Info "Verifying that PyTorch can actually see the GPU..."
$chk = @"
import torch
assert torch.cuda.is_available(), 'torch.cuda.is_available() is False'
print('CUDA_OK ' + torch.cuda.get_device_name(0))
props = torch.cuda.get_device_properties(0)
print('VRAM_GB %.1f' % (props.total_memory / 2**30))
"@
$out = & $venvPy @('-c', $chk)
if ($LASTEXITCODE -ne 0) {
    Fail @"
PyTorch is installed but CANNOT use your GPU.
Fixes that usually work:
  1. Update your NVIDIA driver: https://www.nvidia.com/drivers  (need 580 or newer), then reboot.
  2. Reboot once after driver install before giving up.
Then run 1-Install.bat again.
"@
}
Write-Good ($out -join ' | ')

Write-Info "Pre-downloading the base model weights (yolo11n.pt)..."
Run $venvPy @('-c', "from ultralytics import YOLO; YOLO('yolo11n.pt'); print('weights_ok')")

Write-Header "STEP 1 FINISHED - everything installed. Next: double-click 2-Download-Dataset.bat"
