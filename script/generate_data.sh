#!/bin/bash

WORK_DIR=$EXPERIMENT_DIR
DATASET_PATH=$DATA_DIR
WINDOW_SIZE=5
HORIZON_SIZE=10

echo "Processed data cache dir: $WORK_DIR"
echo "Origin b2d data dir: $DATASET_PATH"
echo "History size: $WINDOW_SIZE"
echo "Horizon size: $HORIZON_SIZE"

mkdir -p "$WORK_DIR" || { echo "Error: Failed to create directory $WORK_DIR"; exit 1; }

python src/data/generate_data/generate_action.py --dataset_path "$DATASET_PATH" --work_dir "$WORK_DIR"
python src/data/generate_data/window.py --work_dir "$WORK_DIR" --window_size "$WINDOW_SIZE" --horizon "$HORIZON_SIZE" --num_cpus 32
python src/data/generate_data/get_statistics.py --data_path "${WORK_DIR}/b2d_action/train"

TARGET_JSON="${PWD}/config/statistics/b2d_statistics.json"
STATISTICS_DIR="${PWD}/config/statistics"
GENERATED_JSON="${PWD}/b2d_statistics.json"

if [ -f "$GENERATED_JSON" ]; then
    if [ ! -f "$TARGET_JSON" ]; then
        echo "Moving JSON file to target directory..."
        mv "$GENERATED_JSON" "$STATISTICS_DIR" || { echo "Error: Failed to move file"; exit 1; }
    else
        echo "Warning: Target file already exists at $TARGET_JSON (skipped)"
        echo "$(date) - WARNING: File $TARGET_JSON already exists" >> "$WORK_DIR/process.log"
    fi
else
    echo "Error: Generated JSON file not found at $GENERATED_JSON"
    exit 1
fi