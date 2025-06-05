#!/bin/bash

OUTPUT_DIR="./../results/features_engineering"

export PYTHONPATH=./../

# Run the Python script with default parameters
python3 main.py --output-dir $OUTPUT_DIR