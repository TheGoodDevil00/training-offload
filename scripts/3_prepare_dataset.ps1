# Step 3 - convert raw VisDrone annotations into the single-class YOLO
# dataset that training consumes (class 0 = human).

$ErrorActionPreference = 'Stop'
. (Join-Path $PSScriptRoot '_common.ps1')

Write-Header "STEP 3 of 5 - Convert VisDrone to YOLO format (~2-5 minutes)"

$py      = Get-VenvPy
$DataDir = Join-Path $RepoRoot 'datasets\usable'
$train   = Join-Path $DataDir 'VisDrone2019-DET-train'
$val     = Join-Path $DataDir 'VisDrone2019-DET-val'
$out     = Join-Path $DataDir 'yolo-human'

if (-not ((Test-Path (Join-Path $train 'annotations')) -and (Test-Path (Join-Path $val 'annotations')))) {
    Fail "Raw VisDrone folders not found. Double-click 2-Download-Dataset.bat first."
}

# --copy: plain copies instead of symlinks (symlinks on Windows need admin/Developer Mode).
Run $py @('training\prepare_dataset.py', '--train', $train, '--val', $val, '--out', $out, '--copy')

$yaml = Join-Path $out 'data.yaml'
if (-not (Test-Path $yaml)) {
    Fail "Conversion finished but data.yaml is missing at $yaml"
}
$nTrain = (Get-ChildItem (Join-Path $out 'train\images') -File).Count
$nVal   = (Get-ChildItem (Join-Path $out 'val\images') -File).Count
Write-Good "YOLO dataset ready: $nTrain train images, $nVal val images"
Write-Info "data.yaml: $yaml"

Write-Header "STEP 3 FINISHED. Optional: QUICK-TEST.bat, then 4-Train.bat for the real run."
