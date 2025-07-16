import os
import json
import yaml
import argparse
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


def evaluate(
    model_adapter: model_adapters.BaseAdapter,
    prompt: str,
    dataset: datasets.Dataset,
    task_type: str,
    **kwargs,
):
    """
    Evaluate a model adapter on a specific task using the provided dataset.

    This function iterates through the dataset, generates predictions using the
    model adapter, and computes task-specific evaluation metrics. It handles
    different prompt formats based on the task type.

    Args:
        model_adapter: The model adapter instance to evaluate.
        prompt: The prompt template to use for the task.
        dataset: The dataset containing test samples.
        task_type: The type of task being evaluated (affects prompt formatting).
        **kwargs: Additional keyword arguments passed to the evaluation function.

    Returns:
        tuple: A tuple containing:
            - scores (dict): Dictionary of evaluation metrics for the task
            - preds (list): List of model predictions
            - golds (list): List of ground truth answers
    """
    preds, golds = [], []
    print("=" * 80)
    print("Prompt: ", prompt)
    data_size = len(dataset)

    # Iterate through dataset and generate predictions
    acc_tokens = {
        "input": 0,
        "output": 0,
        "total": 0,
    }
    for idx_ in tqdm(range(data_size), desc=task_type):
        sample = dataset[idx_]

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
    Main function to run the evaluation pipeline.

    This function handles model initialization, dataset loading, and evaluation
    orchestration. It supports different model types and automatically selects
    the appropriate adapter and configuration.

    Args:
        args: Command line arguments containing model name, task type, dataset path,
              output path, and other configuration options.
    """
    # Load model configuration
    model_config = yaml.load(
        open(f"configs/{args.model_name}.yaml"), Loader=yaml.FullLoader
    )
    model_path = model_config.get("model_path")
    tokenizer_path = model_config.get("tokenizer_path", model_path)

    device = f"cuda:{args.gpus}"
    model_name = model_path.split("/")[-1].lower()

    if model_config["model_adapter"] == "Qwen25VLAdapter":
        # Qwen 2.5 VL models
        from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor

        processor = AutoProcessor.from_pretrained(
            model_path,
            trust_remote_code=True,
            use_fast=True,
        )
        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model_path,
            device_map=device,
            torch_dtype=torch.bfloat16,
            trust_remote_code=True,
        )
        model_adapter = getattr(model_adapters, model_config["model_adapter"])(
            model, processor
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

    # Evaluate on each task type
    for task_type in task_types:
        print(model_config.keys())

        # Get task-specific prompt (use custom prompt if available, otherwise use default)
        prompt = model_config.get(
            f"{task_type}_prompt", DEFAULT_PROMPTS[f"{task_type}_prompt"]
        )

        # Load dataset for the specific task
        dataset = datasets.load_dataset(args.dataset_name_or_path, task_type)["test"]

        # Run evaluation
        scores, preds, golds = evaluate(
            model_adapter=model_adapter,
            prompt=prompt,
            dataset=dataset,
            task_type=task_type,
        )

        # Format and print results
        score_str = ", ".join([f"{k}: {v:.2f}" for k, v in scores.items()])
        print(f"Model: {args.model_name}, Task: {task_type}, Scores: {score_str}")

        # Save results to file
        output_res = [
            {
                "pred": pred,
                "gold": gold,
            }
            for pred, gold in zip(preds, golds)
        ]
        output_res = [{"score": score_str}] + output_res
        with open(os.path.join(args.output_path, f"{task_type}.json"), "w") as f:
            json.dump(output_res, f, indent=2)


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
        default="qwen_vl",
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
