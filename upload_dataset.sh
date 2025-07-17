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
REPLICATED_DIR="./replicated_dataset_10x"
PYTHON_UPLOAD_SCRIPT="upload_dataset.py"
DEFAULT_REPO_NAME="visualwebbench-replicated-10x"
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
    Use existing README.md from the source directory if available, otherwise skip.
    """
    # First, check the current directory (project root) for README.md
    current_dir_readme = "README.md"
    dataset_readme = os.path.join(dataset_path, "README.md")
    
    if os.path.exists(current_dir_readme):
        print(f"Using existing README.md from project root: {current_dir_readme}")
        # Copy the README.md to the dataset directory so it gets uploaded
        import shutil
        shutil.copy2(current_dir_readme, dataset_readme)
        return True
    elif os.path.exists(dataset_readme):
        print(f"Using existing README.md from dataset directory: {dataset_readme}")
        return True
    else:
        print(f"No README.md found in project root or {dataset_path}, skipping dataset card generation")
        return False

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Upload dataset to Hugging Face Hub")
    parser.add_argument("dataset_path", help="Path to the dataset directory")
    parser.add_argument("repo_name", help="Repository name")
    parser.add_argument("--organization", help="Organization name")
    parser.add_argument("--private", action="store_true", help="Make repository private")
    parser.add_argument("--generate-card", action="store_true", help="Generate dataset card")
    
    args = parser.parse_args()
    
    # Generate dataset card if requested
    if args.generate_card:
        generate_dataset_card(args.dataset_path, args.repo_name)
    
    # Upload the dataset
    success = upload_dataset(
        dataset_path=args.dataset_path,
        repo_name=args.repo_name,
        organization=args.organization,
        private=args.private
    )
    
    if success:
        print("Dataset upload completed successfully!")
        sys.exit(0)
    else:
        print("Dataset upload failed!")
        sys.exit(1)
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
    echo "It will use existing README.md from the project root if available"
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