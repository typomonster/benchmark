#!/bin/bash

# Generate Dataset Metadata Script
# This script generates the YAML front matter for Hugging Face Hub dataset README

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to print colored output
print_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Default values
DATASET_NAME="visualwebbench/VisualWebBench"
OUTPUT_FORMAT="frontmatter"
README_FILE="README.md"
METADATA_FILE="dataset_metadata.yaml"

# Function to show usage
show_usage() {
    echo "Usage: $0 [OPTIONS]"
    echo ""
    echo "Generate dataset metadata for Hugging Face Hub README.md"
    echo ""
    echo "Options:"
    echo "  -d, --dataset NAME        Dataset name or path (default: $DATASET_NAME)"
    echo "  -f, --format FORMAT       Output format: yaml or frontmatter (default: $OUTPUT_FORMAT)"
    echo "  -r, --readme FILE         README file to update (default: $README_FILE)"
    echo "  -m, --metadata FILE       Metadata output file (default: $METADATA_FILE)"
    echo "  -h, --help                Show this help message"
    echo ""
    echo "Examples:"
    echo "  $0                                    # Generate frontmatter for README.md"
    echo "  $0 --format yaml                     # Generate YAML file only"
    echo "  $0 --dataset ./my_dataset            # Use local dataset"
    echo "  $0 --readme custom_readme.md         # Update custom README file"
}

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        -d|--dataset)
            DATASET_NAME="$2"
            shift 2
            ;;
        -f|--format)
            OUTPUT_FORMAT="$2"
            shift 2
            ;;
        -r|--readme)
            README_FILE="$2"
            shift 2
            ;;
        -m|--metadata)
            METADATA_FILE="$2"
            shift 2
            ;;
        -h|--help)
            show_usage
            exit 0
            ;;
        *)
            print_error "Unknown option: $1"
            show_usage
            exit 1
            ;;
    esac
done

# Check if Python script exists
if [ ! -f "generate_dataset_metadata.py" ]; then
    print_error "generate_dataset_metadata.py not found in current directory"
    exit 1
fi

# Check if Python is available
if ! command -v python >/dev/null 2>&1; then
    print_error "Python not found. Please install Python."
    exit 1
fi

# Check if required packages are installed
python -c "import datasets, yaml, huggingface_hub" 2>/dev/null || {
    print_error "Required Python packages not found."
    print_error "Please install: pip install datasets PyYAML huggingface_hub"
    exit 1
}

print_info "Generating dataset metadata..."
print_info "Dataset: $DATASET_NAME"
print_info "Format: $OUTPUT_FORMAT"

# Build Python command
PYTHON_CMD="python generate_dataset_metadata.py"
PYTHON_CMD="$PYTHON_CMD --dataset_name_or_path $DATASET_NAME"
PYTHON_CMD="$PYTHON_CMD --format $OUTPUT_FORMAT"
PYTHON_CMD="$PYTHON_CMD --output_file $METADATA_FILE"

if [ "$OUTPUT_FORMAT" = "frontmatter" ]; then
    PYTHON_CMD="$PYTHON_CMD --readme_file $README_FILE"
fi

# Run the Python script
print_info "Running: $PYTHON_CMD"
eval "$PYTHON_CMD"

if [ $? -eq 0 ]; then
    print_success "Dataset metadata generation completed!"
    
    if [ "$OUTPUT_FORMAT" = "frontmatter" ]; then
        print_success "README.md updated with metadata frontmatter"
    else
        print_success "Metadata saved to $METADATA_FILE"
    fi
else
    print_error "Dataset metadata generation failed!"
    exit 1
fi 