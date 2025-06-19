#!/bin/bash

ALGORITHM="decision_tree"  # Specify the algorithm to use
OUTPUT_DIR="./../results/training/$ALGORITHM"
TARGET_COLUMN="transfer_fee"

export PYTHONPATH=./../

python3 $ALGORITHM.py \
  --target-column "$TARGET_COLUMN" \
  --output-dir "$OUTPUT_DIR"