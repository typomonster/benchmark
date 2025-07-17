#!/bin/bash

# Set ROOT_DIR - default to current directory if not set
ROOT_DIR=${ROOT_DIR:-$(pwd)}

echo "Using ROOT_DIR: $ROOT_DIR"

DATASET=$ROOT_DIR/datasets/VisualWebBench

export MODEL_NAME=7b

# Default task type
TASK_TYPE=element_ocr,heading_ocr,webqa,action_ground,action_prediction,element_ground
SEED=""
MAX_EXAMPLES=""
ENGINE="pytorch"
BATCH_SIZE="1"
REPEAT="1"

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --dataset)
            DATASET="$2"
            shift 2
            ;;
        --task)
            if [[ "$2" == "all" ]]; then
                TASK_TYPE=element_ocr,heading_ocr,webqa,action_ground,action_prediction,element_ground
            else
                TASK_TYPE="$2"
            fi
            shift 2
            ;;
        --seed)
            SEED="$2"
            shift 2
            ;;
        --max-examples)
            MAX_EXAMPLES="$2"
            shift 2
            ;;
        --engine)
            ENGINE="$2"
            shift 2
            ;;
        --batch-size)
            BATCH_SIZE="$2"
            shift 2
            ;;
        --repeat)
            REPEAT="$2"
            shift 2
            ;;
        *)
            echo "Unknown option: $1"
            echo "Usage: $0 [--dataset <dataset_path>] [--task all|<task_list>] [--seed <seed_value>] [--max-examples <max_examples>] [--engine pytorch|vllm] [--batch-size <batch_size>] [--repeat <repeat_count>]"
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

# Add max_examples if provided
if [[ -n "$MAX_EXAMPLES" ]]; then
    PYTHON_CMD="$PYTHON_CMD --max_examples $MAX_EXAMPLES"
fi

# Add engine if different from default
if [[ "$ENGINE" != "pytorch" ]]; then
    PYTHON_CMD="$PYTHON_CMD --engine $ENGINE"
fi

# Add batch_size if different from default
if [[ "$BATCH_SIZE" != "1" ]]; then
    PYTHON_CMD="$PYTHON_CMD --batch_size $BATCH_SIZE"
fi

# Add repeat if different from default
if [[ "$REPEAT" != "1" ]]; then
    PYTHON_CMD="$PYTHON_CMD --repeat $REPEAT"
fi

# Execute the command
$PYTHON_CMD