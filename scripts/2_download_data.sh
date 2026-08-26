#!/usr/bin/env bash
# Step 2 - download the VisDrone2019-DET dataset (train + val) and unzip it.

source "$(dirname "${BASH_SOURCE[0]}")/_common.sh"
write_header "STEP 2 of 5 - Download the VisDrone2019-DET dataset (~2.5 GB total)"

DATA_DIR="$REPO_ROOT/datasets/usable"
BASE_URL="https://github.com/ultralytics/assets/releases/download/v0.0.0"

mkdir -p "$DATA_DIR"

for split in train val; do
    zip_path="$DATA_DIR/VisDrone2019-DET-${split}.zip"
    dir_path="$DATA_DIR/VisDrone2019-DET-${split}"

    if [ -d "$dir_path/images" ] && [ -d "$dir_path/annotations" ]; then
        write_good "Split '$split' already extracted - skipping."
    else
        if [ -f "$zip_path" ]; then
            write_warn "Found a leftover partial zip for '$split' - restarting its download."
            rm -f "$zip_path"
        fi
        write_info "Downloading split '$split'..."
        curl -L --fail --retry 3 --retry-delay 5 -o "$zip_path" "$BASE_URL/VisDrone2019-DET-${split}.zip"
        
        write_info "Extracting..."
        unzip -q -o "$zip_path" -d "$DATA_DIR"
        
        if [ ! -d "$dir_path/images" ] || [ ! -d "$dir_path/annotations" ]; then
            fail "Extraction of $zip_path did not produce the expected images/annotations folders."
        fi
        rm -f "$zip_path"
        write_good "Split '$split' ready."
    fi

    n_img=$(ls -1q "$dir_path/images" | wc -l)
    n_ann=$(ls -1q "$dir_path/annotations" | wc -l)
    write_info "  $split: $n_img images, $n_ann annotation files"
done

write_header "STEP 2 FINISHED - dataset ready. Next: run ./3-Prepare-Dataset.sh"
