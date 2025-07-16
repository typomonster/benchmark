"""
vLLM-based adapter implementation for Workflow UI vision-language models.

This module provides the VLLMWorkflowUIAdapter class which handles model inference
using vLLM engine for Workflow UI models. It offers high-throughput serving with
continuous batching, PagedAttention, and optimized CUDA kernels.
"""

import re
import io
import base64
from typing import Optional, Dict, Any

import torch
from PIL import Image
from vllm import LLM, SamplingParams
from transformers import AutoProcessor

from model_adapters import BaseAdapter
from utils.constants import *


class VLLMWorkflowUIAdapter(BaseAdapter):
    """
    vLLM-based adapter for Workflow UI vision-language models.

    This class provides an interface to Workflow UI models using vLLM engine
    for high-performance inference. It handles image processing, conversation
    formatting, and model inference with vLLM's optimized architecture.

    Attributes:
        model: The vLLM engine instance for inference.
        processor: The processor for text and image processing.
        sampling_params: Default sampling parameters for generation.
    """

    def __init__(
        self,
        model_path: str,
        processor: AutoProcessor,
        tensor_parallel_size: int = 1,
        gpu_memory_utilization: float = 0.9,
        max_model_len: Optional[int] = None,
    ):
        """
        Initialize the vLLM Workflow UI adapter.

        Args:
            model_path: Path to the Workflow UI model.
            processor: The processor for text and image processing.
            tensor_parallel_size: Number of tensor parallel replicas.
            gpu_memory_utilization: GPU memory utilization for vLLM.
            max_model_len: Maximum sequence length for vLLM engine.
        """
        # Initialize vLLM engine
        self.llm = LLM(
            model=model_path,
            tensor_parallel_size=tensor_parallel_size,
            gpu_memory_utilization=gpu_memory_utilization,
            max_model_len=max_model_len,
            trust_remote_code=True,
            dtype="bfloat16",
        )
        
        # Initialize processor
        self.processor = processor
        
        # Default sampling parameters
        self.sampling_params = SamplingParams(
            temperature=0.7,
            top_p=0.95,
            max_tokens=512,
        )
        
        # Call parent init with None model since we use self.llm
        super().__init__(None, None)

    def generate(
        self,
        query: str,
        image: Image,
        task_type: str,
        sampling_params: Optional[SamplingParams] = None,
    ) -> str:
        """
        Generate a response using vLLM's optimized inference.

        This method formats the input for Workflow UI, processes the image,
        and generates a response using vLLM's high-performance engine.

        Args:
            query: The text query/prompt to send to the model.
            image: The PIL Image object to process.
            task_type: The type of task being performed (affects response post-processing).
            sampling_params: Optional custom sampling parameters.

        Returns:
            The generated text response from the model, post-processed based on task type.
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
        prompt = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

        # Use provided sampling params or default
        params = sampling_params or self.sampling_params

        # Generate using vLLM
        outputs = self.llm.generate(
            prompts=[prompt],
            sampling_params=params,
            use_tqdm=False,
        )

        # Extract response
        response = outputs[0].outputs[0].text

        # Calculate token counts
        num_input_tokens = len(outputs[0].prompt_token_ids)
        num_output_tokens = len(outputs[0].outputs[0].token_ids)
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

    def update_sampling_params(self, **kwargs):
        """
        Update the default sampling parameters.

        Args:
            **kwargs: Sampling parameters to update (temperature, top_p, max_tokens, etc.)
        """
        self.sampling_params = SamplingParams(**kwargs)