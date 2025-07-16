#!/bin/bash

# Set ROOT_DIR - default to current directory if not set
ROOT_DIR=${ROOT_DIR:-$(pwd)}

echo "Using ROOT_DIR: $ROOT_DIR"

# Create directories
mkdir -p "$ROOT_DIR/models"
mkdir -p "$ROOT_DIR/datasets"

# Download Workflow-UI-7B-Instruct model
echo "Downloading leo5072/Workflow-UI-7B-Instruct model..."
python -c "
import os
from huggingface_hub import snapshot_download

model_path = os.path.join('$ROOT_DIR', 'models', 'Workflow-UI-7B-Instruct')
snapshot_download(
    repo_id='leo5072/Workflow-UI-7B-Instruct',
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
os.makedirs(dataset_path, exist_ok=True)

# Available configs for VisualWebBench
configs = ['action_ground', 'action_prediction', 'element_ground', 'element_ocr', 'heading_ocr', 'web_caption', 'webqa']

for config in configs:
    print(f'Downloading config: {config}')
    dataset = load_dataset('visualwebbench/VisualWebBench', config)
    config_path = os.path.join(dataset_path, config)
    dataset.save_to_disk(config_path)
    print(f'Config {config} downloaded to: {config_path}')

print(f'All dataset configs downloaded to: {dataset_path}')
"

echo "Download completed!"
echo "Model location: $ROOT_DIR/models/Workflow-UI-7B-Instruct"
echo "Dataset location: $ROOT_DIR/datasets/VisualWebBench"

