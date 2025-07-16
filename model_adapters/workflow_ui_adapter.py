"""
Adapter implementation for Workflow UI vision-language models.

This module provides the WorkflowUIAdapter class which handles model inference
for Workflow UI models using PyTorch backend. It manages image encoding, prompt
formatting, and task-specific response post-processing for various visual web
understanding tasks.

The adapter supports:
- Image-to-text generation for web page understanding
- Multiple task types with specialized prompt formatting
- Token usage tracking for cost monitoring
- Task-specific output post-processing

Typical usage:
    from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
    
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(model_path)
    processor = AutoProcessor.from_pretrained(model_path)
    adapter = WorkflowUIAdapter(model, processor)
    
    response, tokens = adapter.generate(prompt, image, task_type)
"""

import re
import io
import base64

import torch
from PIL import Image
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info

from model_adapters import BaseAdapter
from utils.constants import *


class WorkflowUIAdapter(BaseAdapter):
    """
    PyTorch-based adapter for Workflow UI vision-language models.

    This class provides a unified interface for Workflow UI models, handling the
    complete inference pipeline from image processing to response generation.
    It implements the BaseAdapter interface for compatibility with the benchmark
    evaluation framework.

    The adapter handles:
    - Image preprocessing and base64 encoding
    - Conversation formatting following Workflow UI's message structure
    - Model inference with configurable generation parameters
    - Task-specific output post-processing
    - Token usage tracking for performance monitoring

    Attributes:
        model (Qwen2_5_VLForConditionalGeneration): The Workflow UI model instance
                                                    loaded with PyTorch backend.
        processor (AutoProcessor): Handles tokenization and image preprocessing
                                  for the model.

    Note:
        This adapter uses PyTorch for inference. For high-throughput scenarios,
        consider using VLLMWorkflowUIAdapter instead.
    """

    def __init__(
        self,
        model: Qwen2_5_VLForConditionalGeneration,
        processor: AutoProcessor,
    ):
        """
        Initialize the Workflow UI adapter with PyTorch backend.

        Args:
            model (Qwen2_5_VLForConditionalGeneration): Pre-loaded Workflow UI model
                                                        with weights on target device.
            processor (AutoProcessor): Initialized processor for handling text
                                      tokenization and image preprocessing.

        Note:
            The model should already be loaded on the appropriate device (CPU/GPU)
            with the desired precision (e.g., bfloat16) before passing to adapter.
        """
        super().__init__(model, None)
        self.processor = processor

    def generate(
        self,
        query: str,
        image: Image,
        task_type: str,
    ) -> tuple[str, dict]:
        """
        Generate a response using Workflow UI's vision-language model.

        This method orchestrates the complete inference pipeline:
        1. Converts PIL image to base64-encoded format
        2. Formats input as conversation with image and text
        3. Processes inputs using the model's processor
        4. Generates response with controlled sampling
        5. Applies task-specific post-processing
        6. Tracks token usage for performance monitoring

        Supported task types and their post-processing:
        - CAPTION_TASK: Extracts content from HTML meta description tags
        - ACTION_PREDICTION_TASK: Returns uppercase single character for MCQ
        - WEBQA_TASK: Extracts answer after colon and removes quotes
        - ELEMENT_OCR_TASK: Extracts text after colon and cleans formatting
        - Others: Returns raw generated response

        Args:
            query (str): The text prompt/question to send to the model.
                        May contain placeholders filled by the benchmark.
            image (Image.Image): PIL Image object to analyze. Will be converted
                               to RGB format if necessary.
            task_type (str): Task identifier from utils.constants that determines
                           the post-processing strategy for the response.

        Returns:
            tuple[str, dict]: A tuple containing:
                - response (str): The processed model output tailored for the task
                - num_tokens (dict): Token usage statistics with keys:
                    - 'input': Number of input tokens
                    - 'output': Number of generated tokens
                    - 'total': Sum of input and output tokens

        Note:
            The model uses temperature sampling (do_sample=True) with max_new_tokens=512.
            Consider adjusting these parameters for different use cases.
        """
        # Convert image to RGB format
        image = image.convert("RGB")

        # Convert PIL Image to base64 string
        image_buffer = io.BytesIO()
        image.save(image_buffer, format="PNG")
        image_base64 = base64.b64encode(image_buffer.getvalue()).decode("utf-8")

        # Format input according to Workflow UI's message structure
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": f"data:image/png;base64,{image_base64}"},
                    {"type": "text", "text": query},
                ],
            }
        ]

        # Preparation for inference using the processor
        text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        image_inputs, video_inputs = process_vision_info(messages)
        inputs = self.processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        )
        inputs = inputs.to(self.model.device)

        # Inference: Generation of the output
        with torch.no_grad():
            generated_ids = self.model.generate(
                **inputs,
                max_new_tokens=512,
                do_sample=True,  # FIXME
            )

        # Extract and decode the generated response
        generated_ids_trimmed = [
            out_ids[len(in_ids) :]
            for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        response = self.processor.batch_decode(
            generated_ids_trimmed,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )[0]

        num_input_tokens = inputs.input_ids.shape[1]
        num_output_tokens = generated_ids.shape[1] - num_input_tokens
        num_tokens = {
            "input": num_input_tokens,
            "output": num_output_tokens,
            "total": num_input_tokens + num_output_tokens,
        }

        # Post-process output based on task type
        if task_type == CAPTION_TASK:
            # Extract content from HTML meta tag format
            pattern = re.compile(r"<meta name=\"description\" content=\"(.*)\">")
            cur_meta = re.findall(pattern, response)
            if cur_meta:
                return cur_meta[0], num_tokens
            else:
                return response, num_tokens
        elif task_type == ACTION_PREDICTION_TASK:
            # Return first character in uppercase for multiple choice tasks
            return response[0].upper() if response else "A", num_tokens
        elif task_type in [WEBQA_TASK, ELEMENT_OCR_TASK]:
            # Extract text after colon separator and clean quotes
            if ":" not in response:
                return response, num_tokens
            response = ":".join(response.split(":")[1:])
            response = response.strip().strip('"').strip("'")
            return response, num_tokens
        else:
            return response, num_tokens
