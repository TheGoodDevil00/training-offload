# Step 5 - export the trained model (NCNN fp16/int8, ONNX int8), evaluate it
# (mAP + FPS benchmark), collect everything into .\output\ and zip it up for
# sending back.

$ErrorActionPreference = 'Stop'
. (Join-Path $PSScriptRoot '_common.ps1')

Write-Header "STEP 5 of 5 - Export, evaluate and package results (~15-30 minutes)"

$py = Get-VenvPy
Assert-NvidiaGpu

# ---------------------------------------------------------------- locate weights
$best2   = Join-Path $RepoRoot 'runs\detect\train-stage2\weights\best.pt'
$best1   = Join-Path $RepoRoot 'runs\detect\train\weights\best.pt'
if     (Test-Path $best2) { $FinalPt = $best2;  $StageName = 'train-stage2' }
elseif (Test-Path $best1) { $FinalPt = $best1;  $StageName = 'train' }
else {
    Fail "No trained weights found (looked for train-stage2 and train run dirs). Run 4-Train.bat first."
}
$WDir   = Split-Path $FinalPt -Parent
$RunDir = Split-Path $WDir -Parent
Write-Good "Using final weights: $FinalPt"

$yaml   = Join-Path $RepoRoot 'datasets\usable\yolo-human\data.yaml'
$valDir = Join-Path $RepoRoot 'datasets\usable\yolo-human\val'
$out    = Join-Path $RepoRoot 'output'

# ---------------------------------------------------------------- exports @416px
function Rename-Ncnn([string]$From, [string]$To) {
    $src = Join-Path $WDir $From
    $dst = Join-Path $WDir $To
    if (-not (Test-Path $src)) { return $false }
    if (Test-Path $dst) { Remove-Item -Recurse -Force $dst }
    Move-Item $src $dst
    return $true
}

Write-Info "Exporting NCNN fp16 (the recommended Raspberry Pi deployment format)..."
Run $py @('training\export.py', '--model', $FinalPt, '--format', 'ncnn', '--imgsz', '416', '--half')
if (-not (Rename-Ncnn 'best_ncnn_model' 'best_fp16_ncnn_model')) {
    Fail "NCNN fp16 export did not produce $WDir\best_ncnn_model"
}
$fp16Dir = Join-Path $WDir 'best_fp16_ncnn_model'

$int8Dir = $null
try {
    Write-Info "Exporting NCNN int8 (needs a short calibration pass over the dataset)..."
    Run $py @('training\export.py', '--model', $FinalPt, '--format', 'ncnn', '--imgsz', '416', '--int8', '--data', $yaml)
    if (Rename-Ncnn 'best_ncnn_model' 'best_int8_ncnn_model') { $int8Dir = Join-Path $WDir 'best_int8_ncnn_model' }
} catch {
    Write-Warn2 "NCNN int8 export failed ($($_.Exception.Message)) - continuing without it."
}

$onnxPath = Join-Path $WDir 'best.onnx'
try {
    Write-Info "Exporting ONNX int8..."
    Run $py @('training\export.py', '--model', $FinalPt, '--format', 'onnx', '--imgsz', '416', '--int8', '--data', $yaml)
    if (-not (Test-Path $onnxPath)) { Write-Warn2 "ONNX file not found at $onnxPath - continuing without it."; $onnxPath = $null }
} catch {
    Write-Warn2 "ONNX export failed ($($_.Exception.Message)) - continuing without it."
    $onnxPath = $null
}

# ---------------------------------------------------------------- accuracy (CPU, Pi-like)
New-Item -ItemType Directory -Force -Path $out | Out-Null
Write-Info "Evaluating accuracy (mAP50 / mAP50-95 on the val split, CPU like a Raspberry Pi...)"
Run $py @('training\eval\evaluate.py', 'accuracy',
          '--model', $FinalPt, '--data', $yaml,
          '--imgsz', '416', '--threads', '4',
          '--out', (Join-Path $out 'eval_accuracy.json'))

# ---------------------------------------------------------------- speed benchmark
$speedJson = Join-Path $out 'eval_speed.json'
try {
    Write-Info "Benchmarking speed on a synthesized clip from val frames..."
    $clip = Join-Path $RepoRoot 'sample_drone.mp4'
    Run $py @('scripts\package_helpers.py', 'video', '--val-dir', $valDir, '--out', $clip)

    $models = @($FinalPt, $fp16Dir) + $(if ($int8Dir) { @($int8Dir) } else { @() })
    Run $py @('training\eval\evaluate.py', 'speed',
              '--models', @($models),
              '--video', $clip, '--imgsz', '640,416,352', '--threads', '4',
              '--out', $speedJson)
} catch {
    Write-Warn2 "Speed benchmark failed ($($_.Exception.Message)) - accuracy numbers are still packaged."
} finally {
    if (Test-Path (Join-Path $RepoRoot 'sample_drone.mp4')) { Remove-Item -Force (Join-Path $RepoRoot 'sample_drone.mp4') }
}

# ---------------------------------------------------------------- preview grid
try {
    Write-Info "Rendering prediction preview over 6 val frames..."
    Run $py @('scripts\package_helpers.py', 'preview', '--model', $FinalPt,
              '--val-dir', $valDir, '--out', (Join-Path $out 'preview_predictions.jpg'))
} catch {
    Write-Warn2 "Preview rendering failed ($($_.Exception.Message)) - skipping."
}

# ---------------------------------------------------------------- collect artifacts
Write-Info "Collecting weights, curves and configs into output\ ..."
Copy-Item $FinalPt (Join-Path $out 'final_best.pt')
$lastPt = Join-Path $WDir 'last.pt'
if (Test-Path $lastPt) { Copy-Item $lastPt (Join-Path $out 'last.pt') }
foreach ($d in @($fp16Dir, $int8Dir)) {
    if ($d -and (Test-Path $d)) { Copy-Item $d (Join-Path $out (Split-Path $d -Leaf)) -Recurse }
}
if ($onnxPath) { Copy-Item $onnxPath (Join-Path $out 'best.onnx') }

$plots = @('results.png','results.csv','args.yaml','confusion_matrix.png',
           'confusion_matrix_normalized.png','PR_curve.png','F1_curve.png','P_curve.png',
           'R_curve.png','labels.jpg','labels_correlogram.jpg','train_batch0.jpg',
           'val_batch0_labels.jpg','val_batch0_pred.jpg')
foreach ($p in $plots) {
    $src = Join-Path $RunDir $p
    if (Test-Path $src) { Copy-Item $src (Join-Path $out $p) }
}
Copy-Item $yaml (Join-Path $out 'data.yaml')

# ---------------------------------------------------------------- RESULTS.txt
$torchVer = (& $py @('-c', 'import torch; print(torch.__version__)')).Trim()
$acc = $null; $spd = $null
if (Test-Path (Join-Path $out 'eval_accuracy.json')) { $acc = Get-Content (Join-Path $out 'eval_accuracy.json') -Raw | ConvertFrom-Json }
if (Test-Path $speedJson)                            { $spd = Get-Content $speedJson -Raw | ConvertFrom-Json }

$lines = @(
  'TRAINING RESULTS - YOLO11n human detector (VisDrone2019-DET)',
  ('Generated : {0}' -f (Get-Date -Format 'yyyy-MM-dd HH:mm')),
  ("GPU       : {0} (driver {1})" -f $GpuName, $GpuDriver),
  ("Torch     : {0}" -f $torchVer),
  ("Weights   : runs/detect/{0}/weights/best.pt" -f $StageName),
  ''
)
if ($acc) {
    $lines += @('--- accuracy @ imgsz 416 (CPU, Pi-5-simulated threads) ---',
      ('mAP50={0}  mAP50-95={1}  precision={2}  recall={3}' -f $acc.'mAP50', $acc.'mAP50-95', $acc.precision, $acc.recall), '')
}
if ($spd -and $spd.results.Count -gt 0) {
    $lines += '--- speed (CPU-only simulation; pi5_fps = laptop fps * 0.55 estimate) ---'
    foreach ($r in $spd.results) {
        $lines += ('{0} @ {1}px : {2} fps  (est Pi5 {3} fps)' -f (Split-Path $r.model -Leaf), $r.imgsz, $r.fps, $r.pi5_fps)
    }
    $lines += ''
}
$lines += '--- contents ---'
Get-ChildItem $out | ForEach-Object {
    if ($_.PSIsContainer) { $lines += ('{0}/  ({1} items)' -f $_.Name, (Get-ChildItem $_.FullName).Count) }
    else { $lines += ('{0}  ({1:N1} MB)' -f $_.Name, ($_.Length / 1MB)) }
}
Set-Content -Path (Join-Path $out 'RESULTS.txt') -Value $lines -Encoding ASCII

# ---------------------------------------------------------------- zip it
$stamp = Get-Date -Format 'yyyyMMdd-HHmm'
$zip = Join-Path $RepoRoot ("training-results-$stamp.zip")
if (Test-Path $zip) { Remove-Item -Force $zip }
Write-Info "Zipping output -> $(Split-Path $zip -Leaf) (a few minutes)..."
Compress-Archive -Path "$out\*" -DestinationPath $zip -CompressionLevel Optimal
$zipMb = (Get-Item $zip).Length / 1MB

Write-Header ("ALL DONE! Send back the file: {0}  ({1:N0} MB)" -f (Split-Path $zip -Leaf), $zipMb)
Write-Info "The unzipped copy also stays in the 'output' folder if you want to look first."
