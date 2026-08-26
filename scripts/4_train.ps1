# Step 4 - train the YOLO11n human detector, two-stage recipe:
#   Stage 1: COCO backbone frozen (--freeze 10), fast high-quality transfer.
#   Stage 2: everything unfrozen (--freeze 0) + cosine LR, warm-started
#            from Stage 1's best.pt.
#
# Tunables (set environment variables before launching, or edit here):
#   EPOCHS_STAGE1 / EPOCHS_STAGE2 / IMGSZ_TRAIN / WORKERS

param([switch]$SmokeTest)

$ErrorActionPreference = 'Stop'
. (Join-Path $PSScriptRoot '_common.ps1')

if ($SmokeTest) {
    Write-Header "QUICK TEST - tiny training run to verify the whole pipeline (~5-10 min)"
} else {
    Write-Header "STEP 4 of 5 - Train YOLO11n (two stages, several hours. Keep this window open!)"
}

Assert-NvidiaGpu

$py = Get-VenvPy
$yaml = Join-Path $RepoRoot 'datasets\usable\yolo-human\data.yaml'
if (-not (Test-Path $yaml)) {
    Fail "YOLO dataset not found. Run 2-Download-Dataset.bat and 3-Prepare-Dataset.bat first."
}

# Defaults tuned for a strong GPU: batch=-1 auto-fits VRAM.
$E1   = if ($env:EPOCHS_STAGE1) { [int]$env:EPOCHS_STAGE1 } else { 100 }
$E2   = if ($env:EPOCHS_STAGE2) { [int]$env:EPOCHS_STAGE2 } else { 40 }
$Imgsz = if ($env:IMGSZ_TRAIN)  { [int]$env:IMGSZ_TRAIN } else { 512 }
$W    = if ($env:WORKERS)       { [int]$env:WORKERS }      else { 8 }

$sw = [System.Diagnostics.Stopwatch]::StartNew()

if ($SmokeTest) {
    # Tiny end-to-end validation run; writes to runs\detect\smoke-test only.
    Run $py @('training\train.py', '--data', $yaml,
              '--epochs', '2', '--fraction', '0.05', '--imgsz', '320', '--batch', '8',
              '--freeze', '10', '--single-cls', '--workers', '4',
              '--project', 'runs/detect', '--name', 'smoke-test')
    Write-Good "Smoke test passed - the pipeline works end to end."
} else {
    # ---------------- Stage 1 ----------------
    Write-Info ("Stage 1/2: frozen-backbone fine-tune ({0} epochs @ {1}px)..." -f $E1, $Imgsz)
    Run $py @('training\train.py', '--data', $yaml,
              '--epochs', "$E1", '--batch', '-1', '--imgsz', "$Imgsz",
              '--freeze', '10', '--single-cls', '--workers', "$W")

    $best1 = Join-Path $RepoRoot 'runs\detect\train\weights\best.pt'
    if (-not (Test-Path $best1)) {
        Fail "Stage 1 finished but best.pt is missing at $best1"
    }
    Write-Good "Stage 1 done in $([int]$sw.Elapsed.TotalMinutes) min -> $best1"

    # ---------------- Stage 2 ----------------
    # NOTE: --name train-stage2 on purpose; train.py purges its target run dir,
    # so reusing runs/detect/train would delete the Stage-1 weights we load here.
    Write-Info ("Stage 2/2: full fine-tune with cosine LR ({0} epochs)..." -f $E2)
    Run $py @('training\train.py', '--data', $yaml,
              '--weights', $best1,
              '--freeze', '0', '--epochs', "$E2", '--imgsz', "$Imgsz",
              '--single-cls', '--cos-lr', '--workers', "$W",
              '--project', 'runs/detect', '--name', 'train-stage2')

    $final = Join-Path $RepoRoot 'runs\detect\train-stage2\weights\best.pt'
    if (-not (Test-Path $final)) {
        Fail "Stage 2 finished but best.pt is missing at $final"
    }
    $sw.Stop()
    Write-Good "Stage 2 done. Total training time: $([int]$sw.Elapsed.TotalHours) h $([int]($sw.Elapsed.Minutes)) min"
    Write-Info "Final weights: $final"

    Write-Header "TRAINING FINISHED. Next: double-click 5-Package-Results.bat"
}
