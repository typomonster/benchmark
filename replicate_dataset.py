import pandas as pd
import os
import shutil
from pathlib import Path


def replicate_dataset(
    original_path, output_path, default_replication_factor=10, custom_factors=None
):
    """
    Replicate the VisualWebBench dataset by the specified factor.

    Args:
        original_path: Path to the original dataset
        output_path: Path where the replicated dataset will be saved
        default_replication_factor: Default number of times to replicate the data
        custom_factors: Dict mapping dataset names to custom replication factors
    """
    if custom_factors is None:
        custom_factors = {}

    # Create output directory
    os.makedirs(output_path, exist_ok=True)

    # Copy non-parquet files (README.md, .gitattributes)
    for item in os.listdir(original_path):
        item_path = os.path.join(original_path, item)
        if os.path.isfile(item_path) and not item.endswith(".parquet"):
            shutil.copy2(item_path, output_path)

    # Define the subsets
    subsets = [
        "action_ground",
        "action_prediction",
        "element_ground",
        "element_ocr",
        "heading_ocr",
        "web_caption",
        "webqa",
    ]

    for subset in subsets:
        subset_path = os.path.join(original_path, subset)
        output_subset_path = os.path.join(output_path, subset)

        if os.path.exists(subset_path):
            # Get replication factor for this subset
            replication_factor = custom_factors.get(subset, default_replication_factor)
            print(f"Processing {subset} with {replication_factor}x replication...")

            # Create subset directory
            os.makedirs(output_subset_path, exist_ok=True)

            # Find the parquet file
            parquet_files = [
                f for f in os.listdir(subset_path) if f.endswith(".parquet")
            ]

            for parquet_file in parquet_files:
                parquet_path = os.path.join(subset_path, parquet_file)

                # Read the parquet file
                df = pd.read_parquet(parquet_path)
                print(f"  Original {subset} size: {len(df)} rows")

                # Replicate the data
                replicated_data = []
                for i in range(replication_factor):
                    replicated_data.append(df.copy())

                # Concatenate all replicated data
                replicated_df = pd.concat(replicated_data, ignore_index=True)
                print(f"  Replicated {subset} size: {len(replicated_df)} rows")

                # Save the replicated data
                output_parquet_path = os.path.join(output_subset_path, parquet_file)
                replicated_df.to_parquet(output_parquet_path, index=False)
                print(f"  Saved to {output_parquet_path}")
        else:
            print(f"Warning: {subset} not found in original dataset")

    print(f"\nDataset replication complete!")
    print(f"Original dataset: {original_path}")
    print(f"Replicated dataset: {output_path}")


if __name__ == "__main__":
    import sys

    original_path = sys.argv[1] if len(sys.argv) > 1 else "./original_dataset"
    output_path = sys.argv[2] if len(sys.argv) > 2 else "./replicated_dataset_10x"
    default_replication_factor = int(sys.argv[3]) if len(sys.argv) > 3 else 10

    # Set custom factors for specific datasets
    custom_factors = {"heading_ocr": 5, "webqa": 5, "element_ground": 8}

    replicate_dataset(
        original_path, output_path, default_replication_factor, custom_factors
    )
