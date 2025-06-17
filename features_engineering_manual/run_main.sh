#!/bin/bash

OUTPUT_DIR="./../results/features_engineering_manual"
CURRENT_FILE="./../results/features_engineering_manual/feature_engineered_transfers.csv"

export PYTHONPATH=./../

# Run the Python script with default parameters
python3 main.py --output-dir $OUTPUT_DIR --current-file $CURRENT_FILE