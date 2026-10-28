#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"

IMAGE_FEATURE_DIR="${IMAGE_FEATURE_DIR:-./data/things_eeg/image_feature/internvit_multilevel_20_24_28_32_36}"
TEXT_FEATURE_DIR="${TEXT_FEATURE_DIR:-}"
EEG_DATA_DIR="${EEG_DATA_DIR:-./data/things_eeg/preprocessed_eeg}"
GPU_ID="${GPU_ID:-0}"
DEVICE="${DEVICE:-cuda:${GPU_ID}}"
EEG_ENCODER_TYPE="${EEG_ENCODER_TYPE:-TSConv}"
BATCH_SIZE="${BATCH_SIZE:-1024}"
LEARNING_RATE="${LEARNING_RATE:-1e-4}"
STAGE2_LEARNING_RATE="1e-5"
SEED="${SEED:-2025}"
NUM_EPOCHS=30
STAGE1_EPOCHS=20
PROJECTOR="linear"
FEATURE_DIM=512
OUTPUT_DIR="${OUTPUT_DIR:-./results/things_eeg/inter"
SUBJECTS="${SUBJECTS:-1 2 3 4 5 6 7 8 9 10}"
mkdir -p "$OUTPUT_DIR"

echo "[SAMGA] Stage-1=20 epochs; Stage-2=10 epochs; Stage-2 LR=${STAGE2_LEARNING_RATE}; seed=${SEED}; device=${DEVICE}"
echo "[SAMGA] Subjects: ${SUBJECTS}; output=${OUTPUT_DIR}"

for SUB_ID in $SUBJECTS; do
    printf -v OUTPUT_NAME 'sub-%02d' "$SUB_ID"
    TRAIN_IDS=()
    for i in {1..10}; do
        if [[ "$i" -ne "$SUB_ID" ]]; then
            TRAIN_IDS+=("$i")
        fi
    done

    echo "[LOSO] target=${SUB_ID}; training sources=${TRAIN_IDS[*]}"
    python -u train.py \
        --batch_size "$BATCH_SIZE" \
        --learning_rate "$LEARNING_RATE" \
        --stage2_learning_rate "$STAGE2_LEARNING_RATE" \
        --stage2_lr_schedule constant \
        --stage2_min_lr 1e-6 \
        --output_name "$OUTPUT_NAME" \
        --eeg_encoder_type "$EEG_ENCODER_TYPE" \
        --train_subject_ids "${TRAIN_IDS[@]}" \
        --test_subject_ids "$SUB_ID" \
        --softplus \
        --num_epochs "$NUM_EPOCHS" \
        --stage1_epochs "$STAGE1_EPOCHS" \
        --image_feature_dir "$IMAGE_FEATURE_DIR" \
        --text_feature_dir "$TEXT_FEATURE_DIR" \
        --eeg_data_dir "$EEG_DATA_DIR" \
        --device "$DEVICE" \
        --output_dir "$OUTPUT_DIR" \
        --selected_channels \
        --eeg_aug \
        --eeg_aug_type smooth \
        --frozen_eeg_prior \
        --img_l2norm \
        --eeg_feature_dim 1024 \
        --projector "$PROJECTOR" \
        --feature_dim "$FEATURE_DIM" \
        --data_average \
        --save_weights \
        --stage1_mmd_start 0.9 \
        --stage1_mmd_end 0.5 \
        --use_multilayer_router \
        --layer_ids 20 24 28 32 36 \
        --layer_prior_center 28 \
        --layer_prior_strength 1.0 \
        --router_eval_mode global \
        --seed "$SEED"
done

python - "$OUTPUT_DIR" <<'PY'
from pathlib import Path
import pandas as pd
import re
import sys

root = Path(sys.argv[1])
rows = []
for path in sorted(root.glob('*/result.csv')):
    match = re.search(r'sub-(\d{2})$', path.parent.name)
    if match:
        row = pd.read_csv(path).iloc[0].to_dict()
        rows.append({'sub': f'sub-{int(match.group(1)):02d}', **row})
if rows:
    df = pd.DataFrame(rows).sort_values('sub')
    numeric = df.select_dtypes(include='number').columns
    avg = {'sub': 'Average', **df[numeric].mean().to_dict()}
    pd.concat([df, pd.DataFrame([avg])], ignore_index=True).to_csv(
        root / 'avg_results.csv', index=False, float_format='%.4f'
    )
    print(f"Saved: {root / 'avg_results.csv'}")
PY
