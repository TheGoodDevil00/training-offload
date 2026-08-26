# Shared helpers for the training-offload step scripts.
# Each step script starts with:  . (Join-Path $PSScriptRoot '_common.ps1')

$RepoRoot = (Resolve-Path (Join-Path $PSScriptRoot '..')).Path
Set-Location $RepoRoot

function Write-Header([string]$Msg) {
    Write-Host ''
    Write-Host ('=' * 64) -ForegroundColor Cyan
    Write-Host ("  " + $Msg) -ForegroundColor Cyan
    Write-Host ('=' * 64) -ForegroundColor Cyan
}

function Write-Info([string]$Msg)  { Write-Host "[i] $Msg" }
function Write-Good([string]$Msg)  { Write-Host "[OK] $Msg" -ForegroundColor Green }
function Write-Warn2([string]$Msg) { Write-Host "[!] $Msg" -ForegroundColor Yellow }

function Fail([string]$Msg) {
    Write-Host ''
    Write-Host "[X] $Msg" -ForegroundColor Red
    Write-Host ''
    exit 1
}

# Run an external command; abort the script if it exits non-zero.
function Run([string]$Exe, [string[]]$CmdArgs) {
    & $Exe @CmdArgs
    if ($LASTEXITCODE -ne 0) {
        Fail "Command failed (exit code $LASTEXITCODE): $Exe $($CmdArgs -join ' ')"
    }
}

# Path to the venv python created by step 1.
function Get-VenvPy {
    $p = Join-Path $RepoRoot '.venv\Scripts\python.exe'
    if (-not (Test-Path $p)) {
        Fail "Python environment not found. Double-click 1-Install.bat first (it creates everything this step needs)."
    }
    return $p
}

# Abort unless an NVIDIA GPU with a CUDA-13-capable driver is present.
function Assert-NvidiaGpu {
    $smi = Get-Command nvidia-smi.exe -ErrorAction SilentlyContinue
    if (-not $smi) {
        Fail @"
No NVIDIA GPU driver detected (nvidia-smi not found).
Training needs an NVIDIA GPU. Install/update the driver from https://www.nvidia.com/drivers ,
reboot, then run this step again.
"@
    }
    $gpuLine = (& nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader | Select-Object -First 1)
    $parts = $gpuLine -split ','
    $script:GpuName   = $parts[0].Trim()
    $script:GpuDriver = $parts[1].Trim()
    $script:GpuVram   = $parts[2].Trim()
    $driverMajor = 0
    if ($GpuDriver -match '^(\d+)\.') { $driverMajor = [int]$Matches[1] }
    Write-Good ("GPU: {0}  (driver {1}, VRAM {2})" -f $GpuName, $GpuDriver, $GpuVram)
    if ($driverMajor -lt 580) {
        Fail @"
This GPU driver ($GpuDriver) is too old for the PyTorch build used here (needs CUDA 13, i.e. driver 580 or newer).
Fix: install the latest driver from https://www.nvidia.com/drivers , reboot, and run this step again.
"@
    }
}
