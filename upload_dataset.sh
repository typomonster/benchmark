#!/bin/bash

# VisualWebBench Dataset Upload Script
# This script uploads the replicated dataset to Hugging Face Hub

set -e  # Exit on any error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration
REPLICATED_DIR="./replicated_dataset_5x"
PYTHON_UPLOAD_SCRIPT="upload_dataset.py"
DEFAULT_REPO_NAME="visualwebbench-replicated-5x"
DEFAULT_ORGANIZATION=""

# Function to print colored output
print_status() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Function to check if command exists
command_exists() {
    command -v "$1" >/dev/null 2>&1
}

# Function to check and install dependencies
check_dependencies() {
    print_status "Checking dependencies..."
    
    # Check if huggingface-cli is installed
    if ! command_exists huggingface-cli; then
        print_error "huggingface-cli not found. Please install it with: pip install huggingface_hub"
        exit 1
    fi
    
    # Check if python is installed
    if ! command_exists python; then
        print_error "Python not found. Please install Python."
        exit 1
    fi
    
    # Check if required Python packages are installed
    python -c "import pandas, pyarrow, huggingface_hub" 2>/dev/null || {
        print_warning "Required Python packages not found. Installing..."
        pip install pandas pyarrow huggingface_hub
    }
    
    print_success "All dependencies are available."
}

# Function to check if user is logged in to Hugging Face
check_hf_auth() {
    print_status "Checking Hugging Face authentication..."
    
    if ! huggingface-cli whoami >/dev/null 2>&1; then
        print_error "Not logged in to Hugging Face Hub."
        print_error "Please run: huggingface-cli login"
        exit 1
    fi
    
    local username=$(huggingface-cli whoami)
    print_success "Logged in as: $username"
}

# Function to create Python upload script
create_upload_script() {
    print_status "Creating Python upload script..."
    
    cat > "$PYTHON_UPLOAD_SCRIPT" << 'EOF'
import os
import sys
from pathlib import Path
from huggingface_hub import HfApi, create_repo, Repository
import pandas as pd

def upload_dataset(dataset_path, repo_name, organization=None, private=False):
    """
    Upload the replicated dataset to Hugging Face Hub.
    
    Args:
        dataset_path: Path to the dataset directory
        repo_name: Name of the repository
        organization: Organization name (optional)
        private: Whether the repository should be private
    """
    api = HfApi()
    
    # Create full repo name
    if organization:
        full_repo_name = f"{organization}/{repo_name}"
    else:
        full_repo_name = repo_name
    
    print(f"Uploading dataset to: {full_repo_name}")
    
    # Create repository
    try:
        create_repo(
            repo_id=full_repo_name,
            repo_type="dataset",
            private=private,
            exist_ok=True
        )
        print(f"Repository created/verified: {full_repo_name}")
    except Exception as e:
        print(f"Error creating repository: {e}")
        return False
    
    # Upload files
    try:
        print("Uploading files...")
        
        # Upload all files in the dataset directory
        for root, dirs, files in os.walk(dataset_path):
            for file in files:
                file_path = os.path.join(root, file)
                relative_path = os.path.relpath(file_path, dataset_path)
                
                print(f"  Uploading: {relative_path}")
                api.upload_file(
                    path_or_fileobj=file_path,
                    path_in_repo=relative_path,
                    repo_id=full_repo_name,
                    repo_type="dataset"
                )
        
        print("Upload completed successfully!")
        return True
        
    except Exception as e:
        print(f"Error uploading files: {e}")
        return False

def generate_dataset_card(dataset_path, repo_name):
    """
    Generate a README.md file for the dataset.
    """
    readme_content = f"""# Augmented {repo_name}

Dataset for the paper: [VisualWebBench: How Far Have Multimodal LLMs Evolved in Web Page Understanding and Grounding?](https://arxiv.org/abs/2404.05955)

[**🌐 Homepage**](https://visualwebbench.github.io/) | [**🐍 GitHub**](https://github.com/VisualWebBench/VisualWebBench) | [**📖 arXiv**](https://arxiv.org/abs/2404.05955)


## Introduction

We introduce **VisualWebBench**, a multimodal benchmark designed to assess the **understanding and grounding capabilities of MLLMs in web scenarios**. VisualWebBench consists of **seven tasks**, and comprises **1.5K** human-curated instances from **139** real websites, covering 87 sub-domains. We evaluate 14 open-source MLLMs, Gemini Pro, Claude 3, and GPT-4V(ision) on WebBench, revealing significant challenges and performance gaps. Further analysis highlights the limitations of current MLLMs, including inadequate grounding in text-rich environments and subpar performance with low-resolution image inputs. We believe VisualWebBench will serve as a valuable resource for the research community and contribute to the creation of more powerful and versatile MLLMs for web-related applications.

![Alt text](https://raw.githubusercontent.com/VisualWebBench/VisualWebBench/main/assets/main.png)

## Benchmark Construction
We introduce VisualWebBench, a comprehensive multimodal benchmark designed to assess the capabilities of MLLMs in the web domain. Inspired by the human interaction process with web browsers, VisualWebBench consists of seven tasks that map to core abilities required for web tasks: captioning, webpage QA, heading OCR, element OCR, element grounding, action prediction, and action grounding, as detailed in the figure. The benchmark comprises 1.5K instances, all uniformly formulated in the QA style, making it easy to evaluate and compare the performance of different MLLMs.
![Alt text](https://raw.githubusercontent.com/VisualWebBench/VisualWebBench/main/assets/compare.png)
The proposed VisualWebBench possesses the following features:
- **Comprehensiveness**: VisualWebBench spans 139 websites with 1.5K samples, encompassing 12 different domains (e.g., travel, sports, hobby, lifestyle, animals, science, etc.) and 87 sub-domains.
- **Multi-granularity**: VisualWebBench assesses MLLMs at three levels: website-level, element-level, and action-level.
- **Multi-tasks**: WebBench encompasses seven tasks designed to evaluate the understanding, OCR, grounding, and reasoning capabilities of MLLMs.
- **High quality**: Quality is ensured through careful human verification and curation efforts.
![Alt text](https://raw.githubusercontent.com/VisualWebBench/VisualWebBench/main/assets/detail.png)

## Evaluation

We provide [evaluation code](https://github.com/VisualWebBench/VisualWebBench) for GPT-4V, Claude, Gemini, and LLaVA 1.6 series.

## Contact
- Junpeng Liu: [jpliu@link.cuhk.edu.hk](jpliu@link.cuhk.edu.hk)
- Yifan Song: [yfsong@pku.edu.cn](yfsong@pku.edu.cn)
- Xiang Yue: [xyue2@andrew.cmu.edu](xyue2@andrew.cmu.edu)

## Citation
If you find this work helpful, please cite out paper:
```
@misc{liu2024visualwebbench,
      title={VisualWebBench: How Far Have Multimodal LLMs Evolved in Web Page Understanding and Grounding?}, 
      author={Junpeng Liu and Yifan Song and Bill Yuchen Lin and Wai Lam and Graham Neubig and Yuanzhi Li and Xiang Yue},
      year={2024},
      eprint={2404.05955},
      archivePrefix={arXiv},
      primaryClass={cs.CL}
}
```
EOF
    
    print_success "Python upload script created."
}

# Function to upload dataset
upload_dataset() {
    local repo_name="${1:-$DEFAULT_REPO_NAME}"
    local organization="${2:-$DEFAULT_ORGANIZATION}"
    local private="${3:-false}"
    
    print_status "Starting dataset upload..."
    
    if [ ! -d "$REPLICATED_DIR" ]; then
        print_error "Replicated dataset not found at $REPLICATED_DIR"
        print_error "Please run the replication script first."
        exit 1
    fi
    
    # Create upload script
    create_upload_script
    
    # Build Python command
    local python_cmd="python $PYTHON_UPLOAD_SCRIPT $REPLICATED_DIR $repo_name --generate-card"
    
    if [ -n "$organization" ]; then
        python_cmd="$python_cmd --organization $organization"
    fi
    
    if [ "$private" = "true" ]; then
        python_cmd="$python_cmd --private"
    fi
    
    # Run the upload
    print_status "Running upload command: $python_cmd"
    eval "$python_cmd"
    
    print_success "Dataset upload completed!"
}

# Function to show upload status
show_upload_info() {
    print_status "Upload Information:"
    
    if [ -d "$REPLICATED_DIR" ]; then
        echo "Dataset ready for upload: $REPLICATED_DIR"
        du -sh "$REPLICATED_DIR"
    else
        echo "Dataset not found. Please run replication first."
    fi
    
    echo ""
    echo "To upload the dataset:"
    echo "  $0 upload [repo_name] [organization] [private]"
    echo ""
    echo "Examples:"
    echo "  $0 upload my-dataset"
    echo "  $0 upload my-dataset my-org"
    echo "  $0 upload my-dataset my-org true"
}

# Function to clean up
cleanup() {
    print_status "Cleaning up upload files..."
    rm -f "$PYTHON_UPLOAD_SCRIPT"
    print_success "Upload files cleaned up."
}

# Main function
main() {
    echo "=== VisualWebBench Dataset Upload Script ==="
    echo "This script uploads the replicated dataset to Hugging Face Hub"
    echo
    
    case "${1:-info}" in
        "upload")
            check_dependencies
            check_hf_auth
            upload_dataset "$2" "$3" "$4"
            ;;
        "info")
            show_upload_info
            ;;
        "clean")
            cleanup
            ;;
        "help"|"-h"|"--help")
            echo "Usage: $0 [command] [options]"
            echo
            echo "Commands:"
            echo "  upload [repo_name] [organization] [private] - Upload dataset to Hugging Face Hub"
            echo "  info                                        - Show upload information (default)"
            echo "  clean                                       - Clean up upload files"
            echo "  help                                        - Show this help message"
            echo
            echo "Arguments:"
            echo "  repo_name     - Repository name (default: $DEFAULT_REPO_NAME)"
            echo "  organization  - Organization name (optional)"
            echo "  private       - Make repository private (true/false, default: false)"
            echo
            echo "Examples:"
            echo "  $0 upload                                   # Upload to default repo"
            echo "  $0 upload my-dataset                        # Upload to specific repo"
            echo "  $0 upload my-dataset my-org                 # Upload to organization"
            echo "  $0 upload my-dataset my-org true            # Upload as private repo"
            ;;
        *)
            print_error "Unknown command: $1"
            echo "Use '$0 help' for usage information."
            exit 1
            ;;
    esac
}

# Run main function with all arguments
main "$@" 