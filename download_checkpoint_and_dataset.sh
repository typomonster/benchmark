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
echo "Downloading VisualWebBench dataset..."
# python -c "
# import os
# from datasets import load_dataset

# dataset_path = os.path.join('$ROOT_DIR', 'datasets', 'VisualWebBench')
# os.makedirs(dataset_path, exist_ok=True)

# # Available configs for VisualWebBench
# configs = ['action_ground', 'action_prediction', 'element_ground', 'element_ocr', 'heading_ocr', 'web_caption', 'webqa']

# for config in configs:
#     print(f'Downloading config: {config}')
#     dataset = load_dataset('leo5072/VisualWebBench', config)
    
#     # Create config directory
#     config_path = os.path.join(dataset_path, config)
#     os.makedirs(config_path, exist_ok=True)
    
#     # Save test split as parquet file
#     if 'test' in dataset:
#         parquet_file = os.path.join(config_path, 'test-00000-of-00001.parquet')
#         dataset.to_parquet(parquet_file)
#         print(f'Config {config} test split saved to: {parquet_file}')
#     else:
#         print(f'Warning: No test split found for config {config}')

# print(f'All dataset configs downloaded to: {dataset_path}')
# "

huggingface-cli download leo5072/VisualWebBench --repo-type dataset --local-dir $ROOT_DIR/datasets/VisualWebBench --local-dir-use-symlinks=False

echo "Download completed!"
echo "Model location: $ROOT_DIR/models/Workflow-UI-7B-Instruct"
echo "Dataset location: $ROOT_DIR/datasets/VisualWebBench"

