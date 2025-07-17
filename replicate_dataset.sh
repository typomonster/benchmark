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
    print_status "Checking dependencies for dataset download..."
    
    local missing_deps=()
    
    # Check if huggingface-cli is installed
    if ! command_exists huggingface-cli; then
        missing_deps+=("huggingface-cli")
        print_warning "huggingface-cli not found"
    else
        print_success "huggingface-cli is installed"
        # Check version
        local hf_version=$(huggingface-cli --version 2>/dev/null || echo "unknown")
        print_status "huggingface-cli version: $hf_version"
    fi
    
    # Check if python is installed
    if ! command_exists python; then
        if command_exists python3; then
            print_status "Using python3 instead of python"
            # Create an alias for the script
            alias python=python3
        else
            missing_deps+=("python")
            print_warning "Python not found"
        fi
    else
        print_success "Python is installed"
        local python_version=$(python --version 2>/dev/null || echo "unknown")
        print_status "Python version: $python_version"
    fi
    
    # Check if pip is available
    if ! command_exists pip; then
        if command_exists pip3; then
            print_status "Using pip3 instead of pip"
            alias pip=pip3
        else
            print_warning "pip not found - will try to install packages with python -m pip"
        fi
    fi
    
    # Check if required Python packages are installed
    print_status "Checking Python packages..."
    
    # Check for huggingface_hub
    if ! python -c "import huggingface_hub" 2>/dev/null; then
        print_warning "huggingface_hub not found"
        missing_deps+=("huggingface_hub")
    else
        print_success "huggingface_hub is installed"
    fi
    
    # Check for pandas and pyarrow (for dataset processing)
    if ! python -c "import pandas" 2>/dev/null; then
        print_warning "pandas not found"
        missing_deps+=("pandas")
    else
        print_success "pandas is installed"
    fi
    
    if ! python -c "import pyarrow" 2>/dev/null; then
        print_warning "pyarrow not found"
        missing_deps+=("pyarrow")
    else
        print_success "pyarrow is installed"
    fi
    
    # Install missing dependencies
    if [ ${#missing_deps[@]} -gt 0 ]; then
        print_status "Installing missing dependencies: ${missing_deps[*]}"
        
        # Install huggingface-cli and huggingface_hub
        if [[ " ${missing_deps[*]} " =~ " huggingface-cli " ]] || [[ " ${missing_deps[*]} " =~ " huggingface_hub " ]]; then
            print_status "Installing huggingface_hub..."
            if command_exists pip; then
                pip install huggingface_hub[cli]
            else
                python -m pip install huggingface_hub[cli]
            fi
        fi
        
        # Install pandas
        if [[ " ${missing_deps[*]} " =~ " pandas " ]]; then
            print_status "Installing pandas..."
            if command_exists pip; then
                pip install pandas
            else
                python -m pip install pandas
            fi
        fi
        
        # Install pyarrow
        if [[ " ${missing_deps[*]} " =~ " pyarrow " ]]; then
            print_status "Installing pyarrow..."
            if command_exists pip; then
                pip install pyarrow
            else
                python -m pip install pyarrow
            fi
        fi
        
        print_success "Dependencies installed successfully"
    else
        print_success "All dependencies are already installed"
    fi
    
    # Verify huggingface-cli is working
    print_status "Verifying huggingface-cli functionality..."
    if huggingface-cli --help >/dev/null 2>&1; then
        print_success "huggingface-cli is working correctly"
    else
        print_error "huggingface-cli is not working properly"
        return 1
    fi
    
    return 0
}

# Function to check Hugging Face authentication
check_hf_auth() {
    print_status "Checking Hugging Face authentication..."
    
    # Check if user is logged in
    if huggingface-cli whoami >/dev/null 2>&1; then
        local username=$(huggingface-cli whoami 2>/dev/null)
        print_success "Logged in as: $username"
        return 0
    else
        print_warning "Not logged in to Hugging Face"
        print_status "Some datasets may require authentication."
        print_status "To login, run: huggingface-cli login"
        return 1
    fi
}

# Generic function to download any Hugging Face dataset
download_hf_dataset() {
    local dataset_name="$1"
    local output_dir="$2"
    local include_pattern="$3"  # Optional: specific files to include
    local exclude_pattern="$4"  # Optional: specific files to exclude
    
    if [ -z "$dataset_name" ] || [ -z "$output_dir" ]; then
        print_error "Usage: download_hf_dataset <dataset_name> <output_directory> [include_pattern] [exclude_pattern]"
        return 1
    fi
    
    print_status "Downloading dataset: $dataset_name"
    print_status "Output directory: $output_dir"
    
    # Clean up existing directory if it exists
    if [ -d "$output_dir" ]; then
        print_warning "Removing existing directory: $output_dir"
        rm -rf "$output_dir"
    fi
    
    # Create temp directory for download
    local temp_dir=$(mktemp -d)
    print_status "Using temporary directory: $temp_dir"
    
    # Build huggingface-cli command
    local hf_cmd="huggingface-cli download \"$dataset_name\" --repo-type=dataset --local-dir \"$temp_dir\" --local-dir-use-symlinks=False"
    
    # Add include pattern if specified
    if [ -n "$include_pattern" ]; then
        hf_cmd="$hf_cmd --include=\"$include_pattern\""
        print_status "Including pattern: $include_pattern"
    fi
    
    # Add exclude pattern if specified
    if [ -n "$exclude_pattern" ]; then
        hf_cmd="$hf_cmd --exclude=\"$exclude_pattern\""
        print_status "Excluding pattern: $exclude_pattern"
    fi
    
    print_status "Executing: $hf_cmd"
    
    # Execute the download command
    if ! eval "$hf_cmd"; then
        print_error "Failed to download dataset: $dataset_name"
        rm -rf "$temp_dir"
        return 1
    fi
    
    # Verify download was successful
    if [ ! -d "$temp_dir" ] || [ -z "$(ls -A "$temp_dir")" ]; then
        print_error "Downloaded dataset directory is empty or doesn't exist"
        rm -rf "$temp_dir"
        return 1
    fi
    
    # Move to final destination
    print_status "Moving dataset to final destination..."
    mv "$temp_dir" "$output_dir"
    
    # Verify the move was successful
    if [ ! -d "$output_dir" ]; then
        print_error "Failed to move dataset to final destination"
        return 1
    fi
    
    # Show download results
    print_status "Dataset contents:"
    ls -la "$output_dir"
    
    local dataset_size=$(du -sh "$output_dir" | cut -f1)
    local file_count=$(find "$output_dir" -type f | wc -l)
    
    print_success "Dataset downloaded successfully!"
    print_success "Dataset: $dataset_name"
    print_success "Size: $dataset_size"
    print_success "Files: $file_count"
    print_success "Location: $output_dir"
    
    return 0
}

# Function to download dataset
download_dataset() {
    print_status "Starting VisualWebBench dataset download from Hugging Face..."
    
    # Check authentication (optional but recommended)
    check_hf_auth
    
    # Use the generic download function
    if download_hf_dataset "$DATASET_NAME" "$ORIGINAL_DIR"; then
        print_success "VisualWebBench dataset download completed successfully!"
        return 0
    else
        print_error "Failed to download VisualWebBench dataset"
        return 1
    fi
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
            check_hf_auth
            download_dataset
            ;;
        "download-any")
            if [ -z "$2" ] || [ -z "$3" ]; then
                print_error "Usage: $0 download-any <dataset_name> <output_directory> [include_pattern] [exclude_pattern]"
                print_error "Example: $0 download-any microsoft/DialoGPT-medium ./my_dataset"
                print_error "Example: $0 download-any microsoft/DialoGPT-medium ./my_dataset '*.json' '*.txt'"
                exit 1
            fi
            check_dependencies
            check_hf_auth
            download_hf_dataset "$2" "$3" "$4" "$5"
            ;;
        "replicate")
            check_dependencies
            check_hf_auth
            check_python_script
            
            # Download dataset if not exists
            if [ ! -d "$ORIGINAL_DIR" ]; then
                download_dataset
            fi
            
            replicate_dataset
            show_statistics
            ;;
        "auth")
            check_dependencies
            check_hf_auth
            ;;
        "login")
            check_dependencies
            print_status "Opening Hugging Face login..."
            huggingface-cli login
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
            echo "  download          - Download the VisualWebBench dataset only"
            echo "  download-any      - Download any Hugging Face dataset"
            echo "                      Usage: $0 download-any <dataset_name> <output_dir> [include] [exclude]"
            echo "  replicate         - Download (if needed) and replicate the dataset (default)"
            echo "  auth              - Check Hugging Face authentication status"
            echo "  login             - Login to Hugging Face"
            echo "  stats             - Show dataset statistics"
            echo "  clean             - Clean up replicated dataset"
            echo "  clean all         - Clean up all generated files"
            echo "  help              - Show this help message"
            echo
            echo "Examples:"
            echo "  $0 download                              # Download VisualWebBench dataset"
            echo "  $0 download-any microsoft/DialoGPT-medium ./my_dataset"
            echo "  $0 download-any squad ./squad_data '*.json'"
            echo "  $0 replicate                             # Full replication workflow"
            echo "  $0 auth                                  # Check login status"
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