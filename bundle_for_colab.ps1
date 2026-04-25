# bundle_for_colab.ps1
#
# Creates TWO zip files in the repo root, ready to upload to Colab:
#   videos.zip      — the 216 extracted mp4s
#   fitnova_src.zip — source code + warm-start weights
#
# Run from the repo root:
#   powershell -ExecutionPolicy Bypass -File bundle_for_colab.ps1

$ErrorActionPreference = 'Stop'

$repoRoot  = "C:\Users\tsh_x\Desktop\FitNova Application"
$videoRoot = "C:\Users\tsh_x\Desktop\FitNova Datasets\fit3d\train\train"

Set-Location $repoRoot
Write-Host "[bundle] repo root : $repoRoot"
Write-Host "[bundle] video root: $videoRoot"
Write-Host ""

# ── Check prerequisites ──────────────────────────────────────────────────────
if (-not (Test-Path $videoRoot)) {
    Write-Host "ERROR: videos not found at $videoRoot" -ForegroundColor Red
    Write-Host "Run this first:" -ForegroundColor Yellow
    Write-Host "  python -m backend.training.preprocessing.extract_fit3d_videos"
    exit 1
}

$weightsPath = "$repoRoot\backend\models\form_model\mt_tcn_weights.weights.h5"
if (-not (Test-Path $weightsPath)) {
    Write-Host "ERROR: warm-start weights not found at $weightsPath" -ForegroundColor Red
    exit 1
}

$videoCount = (Get-ChildItem -Path $videoRoot -Recurse -Filter "*.mp4" -File).Count
Write-Host "[bundle] found $videoCount mp4 files"
Write-Host ""

# ── Zip 1: videos.zip ────────────────────────────────────────────────────────
$videosZip = "$repoRoot\videos.zip"
if (Test-Path $videosZip) { Remove-Item $videosZip -Force }

Write-Host "[bundle] creating videos.zip (this takes 3-10 min, 2.4 GB) ..."
$videoContents = Get-ChildItem -Path $videoRoot -Directory
Compress-Archive -Path $videoContents.FullName -DestinationPath $videosZip -CompressionLevel Fastest

$videosZipMB = [math]::Round((Get-Item $videosZip).Length / 1MB, 1)
Write-Host "[bundle] OK: videos.zip ($videosZipMB MB)" -ForegroundColor Green
Write-Host ""

# ── Zip 2: fitnova_src.zip ───────────────────────────────────────────────────
$stage = "$env:TEMP\fitnova_src_stage"
if (Test-Path $stage) { Remove-Item $stage -Recurse -Force }
New-Item -ItemType Directory -Path $stage | Out-Null

# Copy source files to staging
$srcItems = @(
    "backend\__init__.py",
    "backend\training\__init__.py",
    "backend\training\train_form_model.py",
    "backend\training\models",
    "backend\training\preprocessing",
    "backend\models\form_model\mt_tcn_weights.weights.h5"
)

foreach ($item in $srcItems) {
    $src = "$repoRoot\$item"
    $dst = "$stage\$item"
    $dstParent = Split-Path -Parent $dst

    if (-not (Test-Path $dstParent)) {
        New-Item -ItemType Directory -Path $dstParent -Force | Out-Null
    }

    if (Test-Path $src) {
        if ((Get-Item $src).PSIsContainer) {
            Copy-Item -Path $src -Destination $dst -Recurse -Force
        } else {
            Copy-Item -Path $src -Destination $dst -Force
        }
        Write-Host "  + $item"
    } else {
        Write-Host "  MISSING: $item" -ForegroundColor Yellow
    }
}

# Remove __pycache__ dirs from staging
Get-ChildItem -Path $stage -Recurse -Directory -Filter "__pycache__" -ErrorAction SilentlyContinue |
    Remove-Item -Recurse -Force -ErrorAction SilentlyContinue

$srcZip = "$repoRoot\fitnova_src.zip"
if (Test-Path $srcZip) { Remove-Item $srcZip -Force }

Write-Host ""
Write-Host "[bundle] creating fitnova_src.zip (with forward-slash paths for Linux) ..."
# Use Python's zipfile — PowerShell Compress-Archive stores Windows backslashes
# which Linux/Colab cannot unpack as directory trees.
$py = @"
import zipfile, os
stage = r'$stage'
out   = r'$srcZip'
with zipfile.ZipFile(out, 'w', compression=zipfile.ZIP_DEFLATED) as zf:
    for root, dirs, files in os.walk(stage):
        for fname in files:
            abs_path = os.path.join(root, fname)
            # arc name: relative to stage, forward slashes
            arc_name = os.path.relpath(abs_path, stage).replace(os.sep, '/')
            zf.write(abs_path, arc_name)
print('done')
"@
python -c $py

$srcZipMB = [math]::Round((Get-Item $srcZip).Length / 1MB, 1)
Write-Host "[bundle] OK: fitnova_src.zip ($srcZipMB MB)" -ForegroundColor Green

Remove-Item $stage -Recurse -Force

Write-Host ""
Write-Host "============================================================" -ForegroundColor Cyan
Write-Host "DONE. Two files are ready in the repo root:" -ForegroundColor Cyan
Write-Host "  $videosZip  ($videosZipMB MB)"
Write-Host "  $srcZip  ($srcZipMB MB)"
Write-Host ""
Write-Host "Next: Go to https://colab.research.google.com" -ForegroundColor Cyan
Write-Host "============================================================" -ForegroundColor Cyan
