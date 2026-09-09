# Step 4.5 / Tournament - Overnight Drone-Footage Model Tournament
# Sequential successive-halving training, evaluation, and ranking on RTX 4050.
# Prioritizes high recall and small-object detection for offline annotation.

param(
    [switch]$SmokeTest,
    [switch]$TUI,
    [switch]$Mock,
    [switch]$Leaderboard,
    [switch]$BuildHardVal,
    [double]$Hours = 0,
    [string]$Preset = ""
)

$ErrorActionPreference = 'Stop'
. (Join-Path $PSScriptRoot '_common.ps1')

if ($TUI) {
    Write-Header "DRONE MODEL TOURNAMENT — INTERACTIVE TUI WORKSTATION"
} elseif ($SmokeTest) {
    Write-Header "DRONE MODEL TOURNAMENT — QUICK SMOKE TEST (~10-15 min)"
} elseif ($Mock) {
    Write-Header "DRONE MODEL TOURNAMENT — MOCK TEST RUN"
} else {
    Write-Header "OVERNIGHT DRONE-FOOTAGE MODEL TOURNAMENT (Keep this window open overnight!)"
}

if (-not $Mock -and -not $Leaderboard) {
    Assert-NvidiaGpu
}

$py = Get-VenvPy
$yaml = Join-Path $RepoRoot 'datasets\usable\yolo-human\data.yaml'

if (-not $Mock -and -not $Leaderboard -and -not (Test-Path $yaml)) {
    Fail "YOLO dataset not found at $yaml. Run 2-Download-Dataset.bat and 3-Prepare-Dataset.bat first."
}

$cmdArgs = @('training\tournament_runner.py')

if ($TUI) {
    $cmdArgs += '--tui'
} elseif ($SmokeTest) {
    $cmdArgs += '--smoke-test'
}

if ($Mock) {
    $cmdArgs += '--mock'
}

if ($Leaderboard) {
    $cmdArgs += '--leaderboard'
}

if ($BuildHardVal) {
    $cmdArgs += '--build-hard-val'
}

if ($Hours -gt 0) {
    $cmdArgs += @('--hours', "$Hours")
}

if ($Preset -ne "") {
    $cmdArgs += @('--preset', "$Preset")
}

$sw = [System.Diagnostics.Stopwatch]::StartNew()
Run $py $cmdArgs
$sw.Stop()

$durationHours = [math]::Round($sw.Elapsed.TotalHours, 2)
Write-Good "Tournament command finished in $durationHours hours ($([int]$sw.Elapsed.TotalMinutes) minutes)."
Write-Info "Results, leaderboard, and winner checkpoint are located in runs\tournament\"
