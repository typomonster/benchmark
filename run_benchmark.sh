#!/bin/bash

# Set ROOT_DIR - default to current directory if not set
ROOT_DIR=${ROOT_DIR:-$(pwd)}

echo "Using ROOT_DIR: $ROOT_DIR"

DATASET=$ROOT_DIR/datasets/VisualWebBench

export MODEL_NAME=7b

# Default task type
TASK_TYPE=element_ocr,heading_ocr,action_ground,action_prediction,element_ground
SEED=""

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --task)
            if [[ "$2" == "all" ]]; then
                TASK_TYPE=element_ocr,heading_ocr,action_ground,action_prediction,element_ground
            else
                TASK_TYPE="$2"
            fi
            shift 2
            ;;
        --seed)
            SEED="$2"
            shift 2
            ;;
        *)
            echo "Unknown option: $1"
            echo "Usage: $0 [--task all|<task_list>] [--seed <seed_value>]"
            exit 1
            ;;
    esac
done

export TASK_TYPE

# Build the python command
PYTHON_CMD="python $DEBUG_MODE benchmark.py --model_name $MODEL_NAME --dataset_name_or_path $DATASET --task_type $TASK_TYPE --gpus 0"

# Add seed if provided
if [[ -n "$SEED" ]]; then
    PYTHON_CMD="$PYTHON_CMD --seed $SEED"
fi

# Execute the command
$PYTHON_CMD