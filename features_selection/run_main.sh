#!/bin/bash

OUTPUT_DIR="./../results/features_selection"
INPUT_CSV="./../results/features_engineering/transfer_features_encoded.csv"
TARGET_COLUMN="transfer_fee"
OUTPUT_CSV="$OUTPUT_DIR/transfer_features_selected.csv"

export PYTHONPATH=./../

python3 main.py \
  --input-csv "$INPUT_CSV" \
  --target-column "$TARGET_COLUMN" \
  --output-csv "$OUTPUT_CSV"