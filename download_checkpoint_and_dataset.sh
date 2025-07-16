#!/bin/bash

# Set ROOT_DIR - default to current directory if not set
ROOT_DIR=${ROOT_DIR:-$(pwd)}

echo "Using ROOT_DIR: $ROOT_DIR"

# Create directories
mkdir -p "$ROOT_DIR/models"
mkdir -p "$ROOT_DIR/datasets"

# Download Qwen2.5-VL-7B-Instruct model
echo "Downloading Qwen/Qwen2.5-VL-7B-Instruct model..."
python -c "
import os
from huggingface_hub import snapshot_download

model_path = os.path.join('$ROOT_DIR', 'models', 'Qwen2.5-VL-7B-Instruct')
snapshot_download(
    repo_id='Qwen/Qwen2.5-VL-7B-Instruct',
    local_dir=model_path,
    local_dir_use_symlinks=False
)
print(f'Model downloaded to: {model_path}')
"

# Download VisualWebBench dataset
echo "Downloading visualwebbench/VisualWebBench dataset..."
python -c "
import os
from datasets import load_dataset

dataset_path = os.path.join('$ROOT_DIR', 'datasets', 'VisualWebBench')
dataset = load_dataset('visualwebbench/VisualWebBench')
dataset.save_to_disk(dataset_path)
print(f'Dataset downloaded to: {dataset_path}')
"

echo "Download completed!"
echo "Model location: $ROOT_DIR/models/Qwen2.5-VL-7B-Instruct"
echo "Dataset location: $ROOT_DIR/datasets/VisualWebBench"

