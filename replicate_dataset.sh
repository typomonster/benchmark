#!/bin/bash

# VisualWebBench Dataset Replication Script
# This script downloads and replicates the VisualWebBench dataset by 5x

set -e  # Exit on any error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration
DATASET_NAME="visualwebbench/VisualWebBench"
ORIGINAL_DIR="./original_dataset"
REPLICATED_DIR="./replicated_dataset_10x"
REPLICATION_FACTOR=10
PYTHON_SCRIPT="replicate_dataset.py"

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
    python -c "import pandas, pyarrow" 2>/dev/null || {
        print_warning "Required Python packages not found. Installing pandas and pyarrow..."
        pip install pandas pyarrow
    }
    
    print_success "All dependencies are available."
}

# Function to download dataset
download_dataset() {
    print_status "Downloading VisualWebBench dataset..."
    
    # Download dataset using huggingface-cli
    huggingface-cli download $DATASET_NAME --repo-type=dataset
    
    # Find the downloaded dataset path
    local cache_path="$HOME/.cache/huggingface/hub/datasets--visualwebbench--VisualWebBench"
    local snapshot_path=$(find "$cache_path/snapshots" -maxdepth 1 -type d | tail -1)
    
    if [ ! -d "$snapshot_path" ]; then
        print_error "Downloaded dataset not found in cache"
        exit 1
    fi
    
    # Copy to local directory
    print_status "Copying dataset to local directory..."
    cp -r "$snapshot_path" "$ORIGINAL_DIR"
    
    print_success "Dataset downloaded and copied to $ORIGINAL_DIR"
}

# Function to check if Python replication script exists
check_python_script() {
    if [ ! -f "$PYTHON_SCRIPT" ]; then
        print_error "Python replication script not found at $PYTHON_SCRIPT"
        print_error "Please ensure replicate_dataset.py exists in the current directory"
        exit 1
    fi
    print_success "Python replication script found."
}

# Function to replicate dataset
replicate_dataset() {
    print_status "Starting dataset replication (${REPLICATION_FACTOR}x)..."
    
    if [ ! -d "$ORIGINAL_DIR" ]; then
        print_error "Original dataset not found at $ORIGINAL_DIR"
        exit 1
    fi
    
    # Run the Python replication script
    python "$PYTHON_SCRIPT" "$ORIGINAL_DIR" "$REPLICATED_DIR" "$REPLICATION_FACTOR"
    
    print_success "Dataset replication completed!"
}

# Function to show dataset statistics
show_statistics() {
    print_status "Dataset Statistics:"
    
    echo "Original dataset:"
    if [ -d "$ORIGINAL_DIR" ]; then
        du -sh "$ORIGINAL_DIR"
        echo "  Subsets: $(ls -1 "$ORIGINAL_DIR" | grep -v -E '\.(md|gitattributes)$' | wc -l)"
    else
        echo "  Not found"
    fi
    
    echo "Replicated dataset:"
    if [ -d "$REPLICATED_DIR" ]; then
        du -sh "$REPLICATED_DIR"
        echo "  Subsets: $(ls -1 "$REPLICATED_DIR" | grep -v -E '\.(md|gitattributes)$' | wc -l)"
    else
        echo "  Not found"
    fi
}

# Function to clean up
cleanup() {
    if [ "$1" = "all" ]; then
        print_status "Cleaning up all generated files..."
        rm -rf "$ORIGINAL_DIR" "$REPLICATED_DIR"
        print_success "Cleanup completed."
    elif [ "$1" = "replicated" ]; then
        print_status "Cleaning up replicated dataset..."
        rm -rf "$REPLICATED_DIR"
        print_success "Replicated dataset removed."
    fi
}

# Main function
main() {
    echo "=== VisualWebBench Dataset Replication Script ==="
    echo "This script will download and replicate the VisualWebBench dataset:"
    echo "  - Default replication factor: ${REPLICATION_FACTOR}x"
    echo "  - heading_ocr and webqa: 5x"
    echo
    
    case "${1:-replicate}" in
        "download")
            check_dependencies
            download_dataset
            ;;
        "replicate")
            check_dependencies
            check_python_script
            
            # Download dataset if not exists
            if [ ! -d "$ORIGINAL_DIR" ]; then
                download_dataset
            fi
            
            replicate_dataset
            show_statistics
            ;;
        "stats")
            show_statistics
            ;;
        "clean")
            cleanup "${2:-replicated}"
            ;;
        "help"|"-h"|"--help")
            echo "Usage: $0 [command]"
            echo
            echo "Commands:"
            echo "  download    - Download the original dataset only"
            echo "  replicate   - Download (if needed) and replicate the dataset (default)"
            echo "  stats       - Show dataset statistics"
            echo "  clean       - Clean up replicated dataset"
            echo "  clean all   - Clean up all generated files"
            echo "  help        - Show this help message"
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