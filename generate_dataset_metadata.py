#!/usr/bin/env python3
"""
Generate dataset metadata for Hugging Face Hub README.md

This script analyzes the VisualWebBench dataset structure and generates
the YAML front matter that should be included in the README.md file for
Hugging Face Hub datasets.
"""

import os
import yaml
import json
import argparse
from pathlib import Path
from typing import Dict, List, Any
import datasets
from datasets import Dataset


def get_feature_info(dataset: Dataset) -> List[Dict[str, Any]]:
    """
    Extract feature information from a dataset.

    Args:
        dataset: HuggingFace dataset

    Returns:
        List of feature dictionaries with name and dtype
    """
    features = []

    for name, feature in dataset.features.items():
        feature_dict = {"name": name}

        # Handle different feature types
        if hasattr(feature, "dtype"):
            if feature.dtype == "string":
                feature_dict["dtype"] = "string"
            elif "int" in str(feature.dtype):
                feature_dict["dtype"] = "int64"
            elif "float" in str(feature.dtype):
                feature_dict["dtype"] = "float64"
            else:
                feature_dict["dtype"] = str(feature.dtype)
        elif hasattr(feature, "_type"):
            if feature._type == "Image":
                feature_dict["dtype"] = "image"
            elif feature._type == "Sequence":
                if hasattr(feature.feature, "dtype"):
                    if "int" in str(feature.feature.dtype):
                        feature_dict["sequence"] = "int64"
                    elif "float" in str(feature.feature.dtype):
                        feature_dict["sequence"] = "float64"
                    elif feature.feature.dtype == "string":
                        feature_dict["sequence"] = "string"
                    else:
                        feature_dict["sequence"] = str(feature.feature.dtype)
                elif hasattr(feature.feature, "_type"):
                    if feature.feature._type == "Sequence":
                        feature_dict["sequence"] = {"sequence": "float64"}
                    else:
                        feature_dict["sequence"] = str(feature.feature._type)
                else:
                    feature_dict["sequence"] = "string"
            else:
                feature_dict["dtype"] = str(feature._type).lower()
        else:
            feature_dict["dtype"] = "string"

        features.append(feature_dict)

    return features


def calculate_dataset_size(dataset: Dataset) -> int:
    """
    Calculate the approximate dataset size in bytes.

    Args:
        dataset: HuggingFace dataset

    Returns:
        Size in bytes
    """
    try:
        # Try to get the actual size from dataset info
        if hasattr(dataset, "info") and hasattr(dataset.info, "dataset_size"):
            return dataset.info.dataset_size

        # Fallback: estimate based on number of examples
        num_examples = len(dataset)

        # Rough estimates per task type (based on your reference data)
        size_estimates = {
            "action_ground": 1126000,  # ~1.1MB per example
            "action_prediction": 755000,  # ~755KB per example
            "element_ground": 1310000,  # ~1.3MB per example
            "element_ocr": 722000,  # ~722KB per example
            "heading_ocr": 791000,  # ~791KB per example
            "web_caption": 842000,  # ~842KB per example
            "webqa": 865000,  # ~865KB per example
        }

        # Try to determine task type from dataset
        if num_examples > 0:
            sample = dataset[0]
            task_type = sample.get("task_type", "unknown")
            if task_type in size_estimates:
                return size_estimates[task_type] * num_examples

            # If task_type is not in the sample, try to infer from column names
            # This helps with local datasets that might have different structure
            if "instruction" in sample:
                return size_estimates["action_ground"] * num_examples
            elif "elem_desc" in sample and "options" in sample:
                return size_estimates["element_ground"] * num_examples
            elif "bbox" in sample and "elem_desc" in sample:
                return size_estimates["element_ocr"] * num_examples
            elif "question" in sample:
                return size_estimates["webqa"] * num_examples

        # Default estimate: 800KB per example
        return 800000 * num_examples

    except Exception as e:
        print(f"Warning: Could not calculate dataset size: {e}")
        return 0


def generate_dataset_metadata(
    dataset_name_or_path: str, output_file: str = None
) -> Dict[str, Any]:
    """
    Generate complete dataset metadata for Hugging Face Hub.

    Args:
        dataset_name_or_path: Path to dataset or HuggingFace dataset name
        output_file: Optional file to save the metadata to

    Returns:
        Dictionary containing the complete metadata
    """

    # Task configurations
    task_configs = [
        "action_ground",
        "action_prediction",
        "element_ground",
        "element_ocr",
        "heading_ocr",
        "web_caption",
        "webqa",
    ]

    dataset_info = []
    configs = []

    total_examples = 0

    print("Analyzing dataset structure...")

    # Check if this is a local path or HuggingFace dataset
    is_local_path = os.path.exists(dataset_name_or_path)

    for config_name in task_configs:
        try:
            print(f"Processing config: {config_name}")

            dataset = None
            test_split = None

            if is_local_path:
                # Handle local dataset directory structure
                config_path = os.path.join(dataset_name_or_path, config_name)
                if os.path.exists(config_path):
                    # Try to load from subdirectory
                    try:
                        dataset = datasets.load_from_disk(config_path)
                        if isinstance(dataset, dict):
                            # If it's a DatasetDict, get the test split
                            test_split = dataset.get(
                                "test", dataset.get("train", list(dataset.values())[0])
                            )
                        else:
                            # If it's a single dataset, use it directly
                            test_split = dataset
                    except Exception:
                        # Fallback: try to load parquet files directly
                        parquet_files = []
                        for file in os.listdir(config_path):
                            if file.endswith(".parquet"):
                                parquet_files.append(os.path.join(config_path, file))

                        if parquet_files:
                            test_split = datasets.load_dataset(
                                "parquet", data_files=parquet_files
                            )["train"]
                        else:
                            print(f"Warning: No parquet files found in {config_path}")
                            continue
                else:
                    print(f"Warning: Config directory {config_path} not found")
                    continue
            else:
                # Handle HuggingFace dataset
                dataset = datasets.load_dataset(dataset_name_or_path, config_name)

                # Process test split (assuming that's what we have)
                test_split = dataset.get("test")
                if test_split is None:
                    # Try other possible split names
                    for split_name in ["train", "validation", "dev"]:
                        if split_name in dataset:
                            test_split = dataset[split_name]
                            break

            if test_split is None:
                print(f"Warning: No test split found for {config_name}")
                continue

            # Get features
            features = get_feature_info(test_split)

            # Calculate sizes
            num_examples = len(test_split)
            dataset_size = calculate_dataset_size(test_split)
            download_size = int(dataset_size * 0.98)  # Approximate download size

            total_examples += num_examples

            # Create dataset info entry
            config_info = {
                "config_name": config_name,
                "features": features,
                "splits": [
                    {
                        "name": "test",
                        "num_bytes": dataset_size,
                        "num_examples": num_examples,
                    }
                ],
                "download_size": download_size,
                "dataset_size": dataset_size,
            }

            dataset_info.append(config_info)

            # Create config entry
            config_entry = {
                "config_name": config_name,
                "data_files": [{"split": "test", "path": f"{config_name}/test-*"}],
            }

            configs.append(config_entry)

        except Exception as e:
            print(f"Error processing {config_name}: {e}")
            continue

    # Determine size category
    if total_examples < 1000:
        size_category = "n<1K"
    elif total_examples < 10000:
        size_category = "1K<n<10K"
    elif total_examples < 100000:
        size_category = "10K<n<100K"
    else:
        size_category = "n>100K"

    # Complete metadata structure
    metadata = {
        "dataset_info": dataset_info,
        "configs": configs,
        "license": "apache-2.0",
        "task_categories": ["image-to-text", "visual-question-answering"],
        "language": ["en"],
        "pretty_name": "VisualWebBench",
        "size_categories": [size_category],
    }

    # Save to file if specified
    if output_file:
        with open(output_file, "w") as f:
            yaml.dump(metadata, f, default_flow_style=False, sort_keys=False)
        print(f"Metadata saved to {output_file}")

    return metadata


def format_yaml_frontmatter(metadata: Dict[str, Any]) -> str:
    """
    Format metadata as YAML front matter for README.md

    Args:
        metadata: Dictionary containing metadata

    Returns:
        Formatted YAML string with front matter delimiters
    """
    yaml_str = yaml.dump(metadata, default_flow_style=False, sort_keys=False)
    return f"---\n{yaml_str}---\n"


def main():
    parser = argparse.ArgumentParser(
        description="Generate dataset metadata for Hugging Face Hub"
    )
    parser.add_argument(
        "--dataset_name_or_path",
        default="visualwebbench/VisualWebBench",
        help="Dataset name or path",
    )
    parser.add_argument(
        "--output_file",
        default="dataset_metadata.yaml",
        help="Output file for metadata",
    )
    parser.add_argument(
        "--format",
        choices=["yaml", "frontmatter"],
        default="frontmatter",
        help="Output format: yaml file or frontmatter for README",
    )
    parser.add_argument(
        "--readme_file",
        default="README.md",
        help="README file to prepend frontmatter to",
    )

    args = parser.parse_args()

    # Generate metadata
    metadata = generate_dataset_metadata(args.dataset_name_or_path, args.output_file)

    if args.format == "frontmatter":
        # Generate frontmatter
        frontmatter = format_yaml_frontmatter(metadata)

        # Check if README exists
        readme_path = Path(args.readme_file)
        if readme_path.exists():
            # Read existing README
            with open(readme_path, "r", encoding="utf-8") as f:
                content = f.read()

            # Remove existing frontmatter if present
            if content.startswith("---\n"):
                parts = content.split("---\n", 2)
                if len(parts) >= 3:
                    content = parts[2]

            # Prepend new frontmatter
            new_content = frontmatter + "\n" + content

            # Write back to README
            with open(readme_path, "w", encoding="utf-8") as f:
                f.write(new_content)

            print(f"README.md updated with metadata frontmatter")
        else:
            # Create new README with just frontmatter
            with open(readme_path, "w", encoding="utf-8") as f:
                f.write(frontmatter)

            print(f"New README.md created with metadata frontmatter")

    print("Dataset metadata generation completed!")
    print(f"Total configurations: {len(metadata['dataset_info'])}")
    print(f"Size category: {metadata['size_categories'][0]}")


if __name__ == "__main__":
    main()
