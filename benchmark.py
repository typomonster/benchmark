"""
Benchmark evaluation script for visual web understanding models.

This module provides functionality to evaluate different visual language models
on the VisualWebBench dataset across multiple tasks including web captioning,
OCR tasks, element grounding, action prediction, and visual question answering.

The evaluation pipeline supports multiple model adapters and can evaluate on
single or multiple tasks simultaneously, computing task-specific metrics and
saving results to JSON files.
"""

import os
import json
import yaml
import argparse
import random
import numpy as np
import time
from tqdm import tqdm

import datasets
import torch

import model_adapters
from utils import DEFAULT_PROMPTS
from utils import (
    eval_web_caption,
    eval_heading_ocr,
    eval_element_ocr,
    eval_action_prediction,
    eval_element_ground,
    eval_action_ground,
    eval_webqa,
)
from utils.constants import *

# Mapping from task types to their corresponding evaluation functions
eval_metric = {
    CAPTION_TASK: eval_web_caption,
    HEADING_OCR_TASK: eval_heading_ocr,
    WEBQA_TASK: eval_webqa,
    ELEMENT_OCR_TASK: eval_element_ocr,
    ELEMENT_GROUND_TASK: eval_element_ground,
    ACTION_PREDICTION_TASK: eval_action_prediction,
    ACTION_GROUND_TASK: eval_action_ground,
}


def set_random_seed(seed):
    """
    Set random seed for reproducible results across all libraries.

    This function ensures deterministic behavior by setting seeds for:
    - Python's built-in random module
    - NumPy's random number generator
    - PyTorch's CPU and CUDA random number generators
    - Transformers library (if available)
    - CUDA backend operations (disables non-deterministic algorithms)

    Args:
        seed (int): Random seed value to use for all random number generators.
                   Common values are 42, 0, or any positive integer.

    Note:
        Setting deterministic behavior may impact performance on CUDA devices
        due to disabled optimizations.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # Set deterministic behavior for CUDA operations
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # Set transformers seed
    try:
        import transformers

        transformers.set_seed(seed)
    except ImportError:
        pass


def evaluate(
    model_adapter: model_adapters.BaseAdapter,
    prompt: str,
    dataset: datasets.Dataset,
    task_type: str,
    max_examples: int = 0,
    batch_size: int = 1,
    repeat: int = 1,
    **kwargs,
):
    """
    Evaluate a model adapter on a specific task using the provided dataset.

    This function orchestrates the evaluation pipeline by:
    1. Iterating through dataset samples (in batches if supported)
    2. Formatting prompts based on task-specific requirements
    3. Generating predictions using the model adapter
    4. Computing task-specific evaluation metrics
    5. Tracking token usage statistics

    The function supports various visual web understanding tasks including:
    - Web page captioning (CAPTION_TASK)
    - Heading OCR extraction (HEADING_OCR_TASK)
    - Web-based question answering (WEBQA_TASK)
    - Element text extraction (ELEMENT_OCR_TASK)
    - UI element grounding (ELEMENT_GROUND_TASK)
    - Action prediction (ACTION_PREDICTION_TASK)
    - Action grounding (ACTION_GROUND_TASK)

    Args:
        model_adapter (BaseAdapter): The model adapter instance implementing the
                                    inference interface for the specific model.
        prompt (str): The prompt template with placeholders for task-specific
                     information (e.g., {question}, {bbox_ratio}, {element_desc}).
        dataset (datasets.Dataset): HuggingFace dataset containing test samples
                                   with 'image' and task-specific fields.
        task_type (str): Task identifier from utils.constants that determines
                        prompt formatting and evaluation metrics.
        max_examples (int, optional): Maximum number of examples to evaluate.
                                     0 means no limit. Defaults to 0.
        batch_size (int, optional): Number of samples to process in each batch.
                                   Only used if the adapter supports batch processing.
                                   Defaults to 1 (single sample processing).
        repeat (int, optional): Number of times to repeat the dataset for evaluation.
                               This augments the dataset size by cycling through the
                               original samples. Defaults to 1 (no repetition).
        **kwargs: Additional arguments passed to task-specific evaluation functions.

    Returns:
        tuple[dict, list, list]: A tuple containing:
            - scores (dict): Evaluation metrics including task-specific scores
                           (e.g., ROUGE, F1, accuracy) and token usage stats:
                           - acc_input: Average input tokens per example
                           - acc_output: Average output tokens per example
                           - acc_total: Average total tokens per example
                           - data_size: Number of examples evaluated
            - preds (list): Model predictions for each example
            - golds (list): Ground truth answers for each example

    Raises:
        NotImplementedError: If the task_type is not supported.
    """
    preds, golds = [], []
    print("=" * 80)
    print("Prompt: ", prompt)

    # Calculate effective dataset size with repetition
    original_size = len(dataset)
    effective_size = original_size * repeat

    # Limit dataset size if max_examples is specified
    data_size = effective_size
    if max_examples > 0:
        data_size = min(data_size, max_examples)
        print(f"Limiting evaluation to first {data_size} examples")

    # if repeat > 1:
    #     print(
    #         f"Dataset will be repeated {repeat} times (original size: {original_size}, effective size: {effective_size})"
    #     )

    # Check if adapter supports batch processing
    supports_batch = hasattr(model_adapter, "generate_batch") and batch_size > 1

    # Iterate through dataset and generate predictions
    acc_tokens = {
        "input": 0,
        "output": 0,
        "total": 0,
    }

    if supports_batch:
        # Batch processing mode
        for start_idx in tqdm(
            range(0, data_size, batch_size), desc=f"{task_type} (batch)"
        ):
            end_idx = min(start_idx + batch_size, data_size)
            batch_samples = [
                dataset[idx_ % original_size] for idx_ in range(start_idx, end_idx)
            ]

            # Prepare batch data
            batch_prompts = []
            batch_images = []
            batch_task_types = []
            batch_golds = []

            for sample in batch_samples:
                # Format prompt based on task type
                if task_type in [CAPTION_TASK, HEADING_OCR_TASK]:
                    cur_prompt = prompt
                elif task_type == WEBQA_TASK:
                    cur_prompt = prompt.format(question=sample["question"])
                elif task_type == ELEMENT_OCR_TASK:
                    cur_prompt = prompt.format(bbox_ratio=sample["bbox"])
                elif task_type == ELEMENT_GROUND_TASK:
                    cur_prompt = prompt.format(element_desc=sample["elem_desc"])
                elif task_type == ACTION_PREDICTION_TASK:
                    choices_text = "\n".join(
                        [
                            f"{chr(ord('A')+i)}. {option}"
                            for i, option in enumerate(sample["options"])
                        ]
                    )
                    cur_prompt = prompt.format(
                        bbox_ratio=sample["bbox"], choices_text=choices_text
                    )
                elif task_type == ACTION_GROUND_TASK:
                    cur_prompt = prompt.format(instruction=sample["instruction"])
                else:
                    raise NotImplementedError(f"Task type {task_type} not implemented.")

                batch_prompts.append(cur_prompt)
                batch_images.append(sample["image"])
                batch_task_types.append(task_type)
                batch_golds.append(sample["answer"])

            # Generate predictions for the batch
            batch_responses, batch_token_stats = model_adapter.generate_batch(
                batch_prompts, batch_images, batch_task_types
            )

            # Collect results
            for response, num_tokens, gold in zip(
                batch_responses, batch_token_stats, batch_golds
            ):
                acc_tokens["input"] += num_tokens["input"]
                acc_tokens["output"] += num_tokens["output"]
                acc_tokens["total"] += num_tokens["total"]

                preds.append(response)
                golds.append(gold)
    else:
        # Single sample processing mode (original behavior)
        for idx_ in tqdm(range(data_size), desc=task_type):
            sample = dataset[idx_ % original_size]

            # Format prompt based on task type
            if task_type in [CAPTION_TASK, HEADING_OCR_TASK]:
                cur_prompt = prompt
            elif task_type == WEBQA_TASK:
                cur_prompt = prompt.format(question=sample["question"])
            elif task_type == ELEMENT_OCR_TASK:
                cur_prompt = prompt.format(bbox_ratio=sample["bbox"])
            elif task_type == ELEMENT_GROUND_TASK:
                cur_prompt = prompt.format(element_desc=sample["elem_desc"])
            elif task_type == ACTION_PREDICTION_TASK:
                choices_text = "\n".join(
                    [
                        f"{chr(ord('A')+i)}. {option}"
                        for i, option in enumerate(sample["options"])
                    ]
                )
                cur_prompt = prompt.format(
                    bbox_ratio=sample["bbox"], choices_text=choices_text
                )
            elif task_type == ACTION_GROUND_TASK:
                cur_prompt = prompt.format(instruction=sample["instruction"])
            else:
                raise NotImplementedError(f"Task type {task_type} not implemented.")

            # Generate prediction using the model adapter
            response = model_adapter(cur_prompt, sample["image"], task_type=task_type)
            if isinstance(response, tuple):
                response, num_tokens = response
            else:
                num_tokens = {
                    "input": 0,
                    "output": 0,
                    "total": 0,
                }
            acc_tokens["input"] += num_tokens["input"]
            acc_tokens["output"] += num_tokens["output"]
            acc_tokens["total"] += num_tokens["total"]

            preds.append(response)
            golds.append(sample["answer"])

    # Compute evaluation metrics
    scores = eval_metric[task_type](preds, golds)
    for k, v in acc_tokens.items():
        scores[f"acc_{k}"] = v / data_size
    scores[f"data_size"] = data_size
    return scores, preds, golds


def main(args):
    """
    Main function to orchestrate the complete evaluation pipeline.

    This function coordinates the entire benchmark evaluation process:
    1. Sets up reproducible environment with random seeds
    2. Loads model configuration from YAML files
    3. Initializes the appropriate model adapter based on engine choice
    4. Loads datasets for specified tasks
    5. Runs evaluation and collects metrics
    6. Saves results to JSON files
    7. Computes and displays aggregate benchmark scores

    The function supports:
    - Multiple inference engines (PyTorch, vLLM)
    - Single or multiple task evaluation
    - Configurable model paths and parameters
    - Comprehensive result logging and scoring

    Args:
        args (argparse.Namespace): Command line arguments containing:
            - model_name (str): Model configuration name (matches configs/*.yaml)
            - task_type (str): Single task or comma-separated list of tasks
            - dataset_name_or_path (str): Path to VisualWebBench dataset
            - output_path (str): Directory for saving evaluation results
            - gpus (str): GPU device specification (e.g., "0" or "0,2")
            - seed (int): Random seed for reproducibility
            - max_examples (int): Maximum examples per task (0 = unlimited)
            - engine (str): Inference engine choice ("pytorch" or "vllm")
            - batch_size (int): Number of samples to process in each batch
            - repeat (int): Number of times to repeat the dataset for evaluation

    Side Effects:
        - Creates output directories if they don't exist
        - Saves evaluation results to JSON files in output_path
        - Prints evaluation progress and final scores to stdout

    Note:
        The benchmark computes two aggregate scores:
        - Multimodal Score: Average of ROUGE-1/F1 scores for text generation tasks
        - Grounding Score: Average accuracy for visual grounding tasks
    """
    # Record start time for end-to-end timing
    benchmark_start_time = time.time()

    # Set random seed for reproducible results
    set_random_seed(args.seed)
    print(f"Using random seed: {args.seed}")

    # Load model configuration
    model_config = yaml.load(
        open(f"configs/{args.model_name}.yaml"), Loader=yaml.FullLoader
    )
    model_path = model_config.get("model_path")

    device = f"cuda:{args.gpus}"

    # Initialize model_adapter to avoid potential assignment issues
    model_adapter = None

    if model_config["model_adapter"] == "WorkflowUIAdapter":
        # Workflow UI models
        from transformers import AutoProcessor

        processor = AutoProcessor.from_pretrained(
            model_path,
            trust_remote_code=True,
            use_fast=True,
        )

        if args.engine == "pytorch":
            # PyTorch-based inference
            try:
                from transformers import Qwen2_5_VLForConditionalGeneration

                model_class = Qwen2_5_VLForConditionalGeneration
            except ImportError:
                # Fallback for different transformers versions
                from transformers import AutoModelForCausalLM

                model_class = AutoModelForCausalLM

            model = model_class.from_pretrained(
                model_path,
                device_map=device,
                torch_dtype=torch.bfloat16,
                trust_remote_code=True,
            )
            model_adapter = getattr(model_adapters, model_config["model_adapter"])(
                model, processor
            )
        elif args.engine == "vllm":
            # vLLM-based inference
            model_adapter = model_adapters.VLLMWorkflowUIAdapter(
                model_path=model_path,
                processor=processor,
                tensor_parallel_size=1,  # Can be configured based on available GPUs
                gpu_memory_utilization=0.9,
                max_model_len=None,  # Will use model's default
            )
    else:
        raise NotImplementedError(
            f"Model adapter {model_config['model_adapter']} not implemented."
        )

    # Parse task types (support multiple tasks separated by commas)
    if "," in args.task_type:
        task_types = [item.strip() for item in args.task_type.split(",")]
    else:
        task_types = [args.task_type]

    # Store all results for summary
    all_results = {}
    task_timings = {}

    # Evaluate on each task type
    for task_type in task_types:
        print(model_config.keys())

        # Get task-specific prompt (use custom prompt if available, otherwise use default)
        prompt = model_config.get(
            f"{task_type}_prompt", DEFAULT_PROMPTS[f"{task_type}_prompt"]
        )

        # Load dataset for the specific task
        import glob

        arrow_files = glob.glob(
            os.path.join(args.dataset_name_or_path, task_type, "**/*.arrow"),
            recursive=True,
        )
        if arrow_files:
            dataset = datasets.load_dataset("arrow", data_files=arrow_files)["train"]
        else:
            dataset = datasets.load_from_disk(
                os.path.join(args.dataset_name_or_path, task_type)
            )["test"]

        # Run evaluation with timing
        task_start_time = time.time()
        scores, preds, golds = evaluate(
            model_adapter=model_adapter,
            prompt=prompt,
            dataset=dataset,
            task_type=task_type,
            max_examples=args.max_examples,
            batch_size=args.batch_size,
            repeat=args.repeat,
        )
        task_end_time = time.time()
        task_duration = task_end_time - task_start_time
        task_timings[task_type] = task_duration

        # Store results for summary
        all_results[task_type] = scores

        # Format and print results
        score_str = ", ".join([f"{k}: {v:.2f}" for k, v in scores.items()])
        print(f"Model: {args.model_name}, Task: {task_type}, Scores: {score_str}")
        print(f"Task {task_type} completed in {task_duration:.2f} seconds")

        # Save results to file
        output_res = [
            {
                "pred": pred,
                "gold": gold,
            }
            for pred, gold in zip(preds, golds)
        ]
        output_res = [
            {"score": score_str, "task_duration": f"{task_duration:.2f}s"}
        ] + output_res
        with open(os.path.join(args.output_path, f"{task_type}.json"), "w") as f:
            json.dump(output_res, f, indent=2)

    # Display benchmark summary
    # Calculate end-to-end timing
    benchmark_end_time = time.time()
    total_benchmark_time = benchmark_end_time - benchmark_start_time

    print("\n" + "=" * 80)
    print("BENCHMARK SUMMARY")
    print("=" * 80)

    # Calculate multimodal score (weighted average)
    multimodal_tasks = [ELEMENT_OCR_TASK, HEADING_OCR_TASK, WEBQA_TASK]
    multimodal_scores = []
    multimodal_weights = []

    for task in multimodal_tasks:
        if task == WEBQA_TASK:
            # WebQA uses F1 score
            if task in all_results and "f1" in all_results[task]:
                multimodal_scores.append(all_results[task]["f1"])
                multimodal_weights.append(all_results[task]["data_size"])
                print(
                    f"{task}['f1']: {all_results[task]['f1']:.2f} (n={all_results[task]['data_size']})"
                )
        else:
            # Other multimodal tasks use ROUGE-1
            if task in all_results and "rouge_1" in all_results[task]:
                multimodal_scores.append(all_results[task]["rouge_1"])
                multimodal_weights.append(all_results[task]["data_size"])
                print(
                    f"{task}['rouge_1']: {all_results[task]['rouge_1']:.2f} (n={all_results[task]['data_size']})"
                )

    if multimodal_scores:
        multimodal_score = np.average(multimodal_scores, weights=multimodal_weights)
        print(f"\nMultimodal Score (weighted average): {multimodal_score:.2f}")
    else:
        print("\nMultimodal Score: N/A (no multimodal tasks evaluated)")

    # Calculate grounding score (weighted average)
    grounding_tasks = [ACTION_GROUND_TASK, ACTION_PREDICTION_TASK, ELEMENT_GROUND_TASK]
    grounding_scores = []
    grounding_weights = []

    print("\nGrounding Task Results:")
    for task in grounding_tasks:
        if task in all_results and "accuracy" in all_results[task]:
            grounding_scores.append(all_results[task]["accuracy"])
            grounding_weights.append(all_results[task]["data_size"])
            print(
                f"{task}['accuracy']: {all_results[task]['accuracy']:.2f} (n={all_results[task]['data_size']})"
            )

    if grounding_scores:
        grounding_score = np.average(grounding_scores, weights=grounding_weights)
        print(f"\nGrounding Score (weighted average): {grounding_score:.2f}")
    else:
        print("\nGrounding Score: N/A (no grounding tasks evaluated)")

    # Display timing results
    print("\nBenchmark Timing:")
    for task_type, duration in task_timings.items():
        print(f"{task_type}: {duration:.2f} seconds")

    print(f"\nTotal Benchmark Time: {total_benchmark_time:.2f} seconds")
    print(
        f"Average Time per Task: {total_benchmark_time / len(task_types):.2f} seconds"
    )

    print("\n" + "=" * 80)
    print(f"Model: {args.model_name}")
    if multimodal_scores:
        print(f"Final Multimodal Score: {multimodal_score:.2f}")
    if grounding_scores:
        print(f"Final Grounding Score: {grounding_score:.2f}")
    print(f"Total Execution Time: {total_benchmark_time:.2f} seconds")
    print("=" * 80)


if __name__ == "__main__":
    # Set up command line argument parsing
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset_name_or_path",
        default="visualwebbench/VisualWebBench",
        type=str,
    )
    parser.add_argument(
        "--model_name",
        default="7b",
        type=str,
        choices=[
            file.split(".")[0]
            for file in os.listdir("configs")
            if file.endswith(".yaml")
        ],
    )
    parser.add_argument(
        "--task_type",
        default="web_caption",
        type=str,
        help="Task type can be one of web_caption, heading_ocr, element_ocr, action_prediction, element_ground, action_ground, webqa. Or several tasks separated by comma.",
    )
    parser.add_argument("--output_path", default="output", type=str)
    parser.add_argument(
        "--gpus",
        default="0",
        type=str,
        help="A single GPU like 1 or multiple GPUs like 0,2",
    )
    parser.add_argument(
        "--seed",
        default=42,
        type=int,
        help="Random seed for reproducible results",
    )
    parser.add_argument(
        "--max_examples",
        default=0,
        type=int,
        help="Maximum number of examples to evaluate per task (0 means no limit).",
    )
    parser.add_argument(
        "--engine",
        default="pytorch",
        type=str,
        choices=["pytorch", "vllm"],
        help="Inference engine to use (pytorch or vllm).",
    )
    parser.add_argument(
        "--batch_size",
        default=1,
        type=int,
        help="Number of samples to process in each batch (only used with vLLM engine).",
    )
    parser.add_argument(
        "--repeat",
        default=1,
        type=int,
        help="Number of times to repeat the dataset for evaluation (augments dataset size).",
    )
    args = parser.parse_args()

    # Create output directories if they don't exist
    if not os.path.exists(args.output_path):
        os.makedirs(args.output_path)

    args.output_path = os.path.join(args.output_path, args.model_name)
    if not os.path.exists(args.output_path):
        os.makedirs(args.output_path)
    print(args)

    # Run main evaluation pipeline
    main(args)
