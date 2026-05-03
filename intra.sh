#!/bin/bash
set -e
trap 'echo "Script Error"' ERR

IMAGE_FEATURE_DIR="./data/things_eeg/image_feature/internvit_multilevel_20_24_28_32_36"
TEXT_FEATURE_DIR=""
EEG_DATA_DIR="./data/things_eeg/preprocessed_eeg"
DEVICE="cuda:0"
EEG_ENCODER_TYPE="EEGProject"
BATCH_SIZE=512
LEARNING_RATE=1e-4
NUM_EPOCHS=60
SELECTED_CHANNELS=('P7' 'P5' 'P3' 'P1' 'Pz' 'P2' 'P4' 'P6' 'P8' 'PO7' 'PO3' 'POz' 'PO4' 'PO8' 'O1' 'Oz' 'O2')
PROJECTOR="linear"
FEATURE_DIM=512
OUTPUT_DIR="./results/things_eeg/intra"

for SUB_ID in {1..10}
do
    OUTPUT_NAME=$(printf "sub-%02d" $SUB_ID)
    echo "Training subject ${SUB_ID}..."
    python train.py \
        --batch_size "$BATCH_SIZE" \
        --stage2_learning_rate 5e-5 \
        --learning_rate "$LEARNING_RATE" \
        --output_name "$OUTPUT_NAME" \
        --eeg_encoder_type "$EEG_ENCODER_TYPE" \
        --train_subject_ids $SUB_ID \
        --test_subject_ids $SUB_ID \
        --softplus \
        --num_epochs "$NUM_EPOCHS" \
        --image_feature_dir "$IMAGE_FEATURE_DIR" \
        --text_feature_dir "$TEXT_FEATURE_DIR" \
        --eeg_data_dir "$EEG_DATA_DIR" \
        --device "$DEVICE"  \
        --output_dir "$OUTPUT_DIR" \
        --selected_channels "${SELECTED_CHANNELS[@]}" \
        --eeg_aug \
        --eeg_aug_type "smooth" \
        --frozen_eeg_prior \
        --img_l2norm \
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
        --seed 2025;
done

python compute_avg_results.py --result_dir "$OUTPUT_DIR";