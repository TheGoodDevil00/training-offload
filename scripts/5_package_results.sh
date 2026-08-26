#!/usr/bin/env bash
# Step 5 - export, evaluate, and package results (Linux)

source "$(dirname "${BASH_SOURCE[0]}")/_common.sh"
write_header "STEP 5 of 5 - Export, evaluate and package results (~15-30 minutes)"

PY=$(get_venv_py)
assert_nvidia_gpu

# ---------------------------------------------------------------- locate weights
BEST2="$REPO_ROOT/runs/detect/train-stage2/weights/best.pt"
BEST1="$REPO_ROOT/runs/detect/train/weights/best.pt"

if [ -f "$BEST2" ]; then
    FINAL_PT="$BEST2"
    STAGE_NAME="train-stage2"
elif [ -f "$BEST1" ]; then
    FINAL_PT="$BEST1"
    STAGE_NAME="train"
else
    fail "No trained weights found. Run 4-Train.sh first."
fi

WDIR="$(dirname "$FINAL_PT")"
RUNDIR="$(dirname "$WDIR")"
write_good "Using final weights: $FINAL_PT"

YAML_PATH="$REPO_ROOT/datasets/usable/yolo-human/data.yaml"
VAL_DIR="$REPO_ROOT/datasets/usable/yolo-human/val"
OUT_DIR="$REPO_ROOT/output"

# ---------------------------------------------------------------- exports @416px
rename_ncnn() {
    local src="$WDIR/$1"
    local dst="$WDIR/$2"
    if [ ! -d "$src" ]; then return 1; fi
    rm -rf "$dst"
    mv "$src" "$dst"
    return 0
}

write_info "Exporting NCNN fp16..."
$PY training/export.py --model "$FINAL_PT" --format ncnn --imgsz 416 --half || fail "fp16 export failed"
if ! rename_ncnn "best_ncnn_model" "best_fp16_ncnn_model"; then
    fail "NCNN fp16 export did not produce best_ncnn_model"
fi
FP16_DIR="$WDIR/best_fp16_ncnn_model"

INT8_DIR=""
write_info "Exporting NCNN int8..."
if $PY training/export.py --model "$FINAL_PT" --format ncnn --imgsz 416 --int8 --data "$YAML_PATH"; then
    if rename_ncnn "best_ncnn_model" "best_int8_ncnn_model"; then
        INT8_DIR="$WDIR/best_int8_ncnn_model"
    fi
else
    write_warn "NCNN int8 export failed - continuing without it."
fi

ONNX_PATH="$WDIR/best.onnx"
write_info "Exporting ONNX int8..."
if ! $PY training/export.py --model "$FINAL_PT" --format onnx --imgsz 416 --int8 --data "$YAML_PATH"; then
    write_warn "ONNX export failed - continuing without it."
    ONNX_PATH=""
elif [ ! -f "$ONNX_PATH" ]; then
    write_warn "ONNX file not found - continuing without it."
    ONNX_PATH=""
fi

# ---------------------------------------------------------------- accuracy
mkdir -p "$OUT_DIR"
write_info "Evaluating accuracy (CPU, Pi-like)..."
$PY training/eval/evaluate.py accuracy --model "$FINAL_PT" --data "$YAML_PATH" --imgsz 416 --threads 4 --out "$OUT_DIR/eval_accuracy.json" || fail "Accuracy eval failed"

# ---------------------------------------------------------------- speed benchmark
SPEED_JSON="$OUT_DIR/eval_speed.json"
write_info "Benchmarking speed..."
CLIP="$REPO_ROOT/sample_drone.mp4"
$PY scripts/package_helpers.py video --val-dir "$VAL_DIR" --out "$CLIP" || true

MODELS=("$FINAL_PT" "$FP16_DIR")
[ -n "$INT8_DIR" ] && MODELS+=("$INT8_DIR")

$PY training/eval/evaluate.py speed --models "${MODELS[@]}" --video "$CLIP" --imgsz 640,416,352 --threads 4 --out "$SPEED_JSON" || write_warn "Speed benchmark failed"
rm -f "$CLIP"

# ---------------------------------------------------------------- preview grid
write_info "Rendering prediction preview..."
$PY scripts/package_helpers.py preview --model "$FINAL_PT" --val-dir "$VAL_DIR" --out "$OUT_DIR/preview_predictions.jpg" || write_warn "Preview rendering failed"

# ---------------------------------------------------------------- collect artifacts
write_info "Collecting artifacts..."
cp "$FINAL_PT" "$OUT_DIR/final_best.pt"
[ -f "$WDIR/last.pt" ] && cp "$WDIR/last.pt" "$OUT_DIR/"
[ -d "$FP16_DIR" ] && cp -r "$FP16_DIR" "$OUT_DIR/"
[ -n "$INT8_DIR" ] && [ -d "$INT8_DIR" ] && cp -r "$INT8_DIR" "$OUT_DIR/"
[ -n "$ONNX_PATH" ] && [ -f "$ONNX_PATH" ] && cp "$ONNX_PATH" "$OUT_DIR/"

PLOTS=(results.png results.csv args.yaml confusion_matrix.png confusion_matrix_normalized.png PR_curve.png F1_curve.png P_curve.png R_curve.png labels.jpg labels_correlogram.jpg train_batch0.jpg val_batch0_labels.jpg val_batch0_pred.jpg)
for p in "${PLOTS[@]}"; do
    [ -f "$RUNDIR/$p" ] && cp "$RUNDIR/$p" "$OUT_DIR/"
done
cp "$YAML_PATH" "$OUT_DIR/"

# ---------------------------------------------------------------- RESULTS.txt
cat <<EOF > "$OUT_DIR/RESULTS.txt"
TRAINING RESULTS - YOLO11n human detector (VisDrone2019-DET)
Generated : $(date '+%Y-%m-%d %H:%M')
GPU       : $GPU_NAME (driver $GPU_DRIVER)
Torch     : $($PY -c 'import torch; print(torch.__version__)')
Weights   : runs/detect/$STAGE_NAME/weights/best.pt

--- contents ---
EOF
ls -lh "$OUT_DIR" >> "$OUT_DIR/RESULTS.txt"

# ---------------------------------------------------------------- zip it
STAMP=$(date '+%Y%m%d-%H%M')
ZIP_FILE="$REPO_ROOT/training-results-$STAMP.zip"
rm -f "$ZIP_FILE"
write_info "Zipping output -> $(basename "$ZIP_FILE")..."
cd "$OUT_DIR"
zip -q -r "$ZIP_FILE" .
cd "$REPO_ROOT"

write_header "ALL DONE! Send back the file: $(basename "$ZIP_FILE")"
