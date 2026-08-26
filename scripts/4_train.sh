#!/usr/bin/env bash
# Step 4 - train the YOLO11n human detector (Linux)

SMOKE_TEST=0
if [ "$1" == "-SmokeTest" ]; then
    SMOKE_TEST=1
fi

source "$(dirname "${BASH_SOURCE[0]}")/_common.sh"

if [ "$SMOKE_TEST" -eq 1 ]; then
    write_header "QUICK TEST - tiny training run to verify the whole pipeline (~5-10 min)"
else
    write_header "STEP 4 of 5 - Train YOLO11n (two stages, several hours. Keep this window open!)"
fi

assert_nvidia_gpu
PY=$(get_venv_py)

YAML_PATH="$REPO_ROOT/datasets/usable/yolo-human/data.yaml"
if [ ! -f "$YAML_PATH" ]; then
    fail "YOLO dataset not found. Run 2-Download-Dataset.sh and 3-Prepare-Dataset.sh first."
fi

E1=${EPOCHS_STAGE1:-100}
E2=${EPOCHS_STAGE2:-40}
IMGSZ=${IMGSZ_TRAIN:-512}
W=${WORKERS:-8}

if [ "$SMOKE_TEST" -eq 1 ]; then
    $PY training/train.py --data "$YAML_PATH" \
        --epochs 2 --fraction 0.05 --imgsz 320 --batch 8 \
        --freeze 10 --single-cls --workers 4 \
        --project runs/detect --name smoke-test || fail "Smoke test failed."
    write_good "Smoke test passed - the pipeline works end to end."
else
    # ---------------- Stage 1 ----------------
    write_info "Stage 1/2: frozen-backbone fine-tune ($E1 epochs @ ${IMGSZ}px)..."
    
    start_s1=$(date +%s)
    $PY training/train.py --data "$YAML_PATH" \
        --epochs "$E1" --batch -1 --imgsz "$IMGSZ" \
        --freeze 10 --single-cls --workers "$W" || fail "Stage 1 failed."
    end_s1=$(date +%s)
    
    BEST1="$REPO_ROOT/runs/detect/train/weights/best.pt"
    if [ ! -f "$BEST1" ]; then
        fail "Stage 1 finished but best.pt is missing at $BEST1"
    fi
    mins=$(( (end_s1 - start_s1) / 60 ))
    write_good "Stage 1 done in ${mins} min -> $BEST1"

    # ---------------- Stage 2 ----------------
    write_info "Stage 2/2: full fine-tune with cosine LR ($E2 epochs)..."
    
    start_s2=$(date +%s)
    $PY training/train.py --data "$YAML_PATH" \
        --weights "$BEST1" \
        --freeze 0 --epochs "$E2" --imgsz "$IMGSZ" \
        --single-cls --cos-lr --workers "$W" \
        --project runs/detect --name train-stage2 || fail "Stage 2 failed."
    end_s2=$(date +%s)

    FINAL="$REPO_ROOT/runs/detect/train-stage2/weights/best.pt"
    if [ ! -f "$FINAL" ]; then
        fail "Stage 2 finished but best.pt is missing at $FINAL"
    fi
    total_mins=$(( (end_s2 - start_s1) / 60 ))
    hrs=$(( total_mins / 60 ))
    mins_rem=$(( total_mins % 60 ))
    write_good "Stage 2 done. Total training time: ${hrs} h ${mins_rem} min"
    write_info "Final weights: $FINAL"

    write_header "TRAINING FINISHED. Next: run ./5-Package-Results.sh"
fi
