#!/bin/bash

OUTPUT_DIR="./../results/data_analysis_training_table"

export PYTHONPATH=./../

# Run the Python script with default parameters
python3 main.py --output-dir $OUTPUT_DIR