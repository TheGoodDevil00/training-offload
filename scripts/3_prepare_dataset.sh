#!/usr/bin/env bash
# Step 3 - convert raw VisDrone annotations into the single-class YOLO format

source "$(dirname "${BASH_SOURCE[0]}")/_common.sh"
write_header "STEP 3 of 5 - Convert VisDrone to YOLO format (~2-5 minutes)"

PY=$(get_venv_py)
DATA_DIR="$REPO_ROOT/datasets/usable"
TRAIN_DIR="$DATA_DIR/VisDrone2019-DET-train"
VAL_DIR="$DATA_DIR/VisDrone2019-DET-val"
OUT_DIR="$DATA_DIR/yolo-human"

if [ ! -d "$TRAIN_DIR/annotations" ] || [ ! -d "$VAL_DIR/annotations" ]; then
    fail "Raw VisDrone folders not found. Run 2-Download-Dataset.sh first."
fi

# --copy: plain copies instead of symlinks
$PY training/prepare_dataset.py --train "$TRAIN_DIR" --val "$VAL_DIR" --out "$OUT_DIR" --copy || fail "Prepare script failed."

YAML_PATH="$OUT_DIR/data.yaml"
if [ ! -f "$YAML_PATH" ]; then
    fail "Conversion finished but data.yaml is missing at $YAML_PATH"
fi

n_train=$(ls -1q "$OUT_DIR/train/images" 2>/dev/null | wc -l || echo 0)
n_val=$(ls -1q "$OUT_DIR/val/images" 2>/dev/null | wc -l || echo 0)

write_good "YOLO dataset ready: $n_train train images, $n_val val images"
write_info "data.yaml: $YAML_PATH"

write_header "STEP 3 FINISHED. Optional: QUICK-TEST.sh, then 4-Train.sh for the real run."
