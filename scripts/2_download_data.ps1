# Step 2 - download the VisDrone2019-DET dataset (train + val) and unzip it.
# Skips splits that are already downloaded AND extracted. Resumable at the
# granularity of one zip: an interrupted download is restarted cleanly.

$ErrorActionPreference = 'Stop'
. (Join-Path $PSScriptRoot '_common.ps1')

Write-Header "STEP 2 of 5 - Download the VisDrone2019-DET dataset (~2.5 GB total)"

$DataDir = Join-Path $RepoRoot 'datasets\usable'
$Base    = 'https://github.com/ultralytics/assets/releases/download/v0.0.0'

New-Item -ItemType Directory -Force -Path $DataDir | Out-Null

foreach ($split in @('train', 'val')) {
    $zip = Join-Path $DataDir ("VisDrone2019-DET-{0}.zip" -f $split)
    $dir = Join-Path $DataDir ("VisDrone2019-DET-" + $split)

    if ((Test-Path (Join-Path $dir 'images')) -and (Test-Path (Join-Path $dir 'annotations'))) {
        Write-Good "Split '$split' already extracted - skipping."
    } else {
        if (Test-Path $zip) {
            Write-Warn "Found a leftover partial zip for '$split' - restarting its download."
            Remove-Item -Force $zip
        }
        Write-Info "Downloading split '$split'..."
        Run 'curl.exe' @('-L', '--fail', '--retry', '3', '--retry-delay', '5',
                         '--retry-all-errors', '-o', $zip, "$Base/VisDrone2019-DET-$split.zip")
        Write-Info "Extracting..."
        Run 'tar.exe' @('-xf', $zip, '-C', $DataDir)
        if (-not ((Test-Path (Join-Path $dir 'images')) -and (Test-Path (Join-Path $dir 'annotations')))) {
            Fail "Extraction of $zip did not produce the expected images/annotations folders."
        }
        Remove-Item -Force $zip   # free ~2 GB once safely extracted
        Write-Good "Split '$split' ready."
    }

    $nImg = (Get-ChildItem (Join-Path $dir 'images') -File).Count
    $nAnn = (Get-ChildItem (Join-Path $dir 'annotations') -File).Count
    Write-Info ("  {0}: {1} images, {2} annotation files" -f $split, $nImg, $nAnn)
}

Write-Header "STEP 2 FINISHED - dataset ready. Next: double-click 3-Prepare-Dataset.bat"
