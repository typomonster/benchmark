"""
vLLM-based adapter implementation for Workflow UI vision-language models.

This module provides the VLLMWorkflowUIAdapter class which leverages vLLM's
high-performance inference engine for Workflow UI models. vLLM offers significant
performance improvements through:

- Continuous batching for optimal GPU utilization
- PagedAttention for efficient memory management
- Optimized CUDA kernels for faster execution
- Tensor parallelism support for multi-GPU setups
- Quantization support for reduced memory usage

The adapter maintains API compatibility with WorkflowUIAdapter while providing
superior throughput for production deployments.

Typical usage:
    from transformers import AutoProcessor

    processor = AutoProcessor.from_pretrained(model_path)
    adapter = VLLMWorkflowUIAdapter(
        model_path=model_path,
        processor=processor,
        tensor_parallel_size=2,  # Use 2 GPUs
        gpu_memory_utilization=0.9
    )

    response, tokens = adapter.generate(prompt, image, task_type)

Note:
    vLLM requires GPU with compute capability >= 7.0 (Volta or newer).
    For CPU inference, use WorkflowUIAdapter instead.
"""

import re
import io
import base64
from typing import Optional, Dict, Any

import torch
from PIL import Image
from vllm import LLM, SamplingParams
from transformers import AutoProcessor
from qwen_vl_utils import process_vision_info

from model_adapters import BaseAdapter
from utils.constants import *


class VLLMWorkflowUIAdapter(BaseAdapter):
    """
    High-performance vLLM-based adapter for Workflow UI vision-language models.

    This adapter implements the BaseAdapter interface using vLLM's optimized
    inference engine, providing significant performance improvements over
    standard PyTorch inference, especially for high-throughput scenarios.

    Key features:
    - Automatic batching of requests for optimal throughput
    - Efficient memory management with PagedAttention
    - Support for tensor parallelism across multiple GPUs
    - Configurable GPU memory utilization
    - Compatible with various quantization methods

    Attributes:
        llm (vllm.LLM): The vLLM engine instance managing model inference
                       with optimized memory allocation and scheduling.
        processor (AutoProcessor): Handles text tokenization and image
                                  preprocessing for the model.
        sampling_params (SamplingParams): Default generation parameters
                                         including temperature, top_p, etc.

    Performance notes:
        - Best suited for batch processing and server deployments
        - Provides 2-5x throughput improvement over PyTorch inference
        - Memory usage scales efficiently with batch size
        - Supports concurrent request handling
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
        Initialize the vLLM Workflow UI adapter with optimized inference engine.

        This constructor sets up the vLLM engine with specified configuration
        and initializes default sampling parameters for generation.

        Args:
            model_path (str): Path to the Workflow UI model weights. Can be:
                            - Local directory path
                            - Hugging Face model ID
                            - S3/GCS path (with appropriate credentials)
            processor (AutoProcessor): Pre-initialized processor for handling
                                      text tokenization and image preprocessing.
                                      Must be compatible with the model.
            tensor_parallel_size (int, optional): Number of GPUs to use for
                                                 tensor parallelism. Defaults to 1.
                                                 Model layers are sharded across GPUs.
            gpu_memory_utilization (float, optional): Fraction of GPU memory to
                                                     reserve for model and KV cache.
                                                     Defaults to 0.9 (90%).
                                                     Lower values leave headroom.
            max_model_len (Optional[int], optional): Maximum sequence length
                                                    (input + output tokens).
                                                    None uses model's default.
                                                    Reduce to save memory.

        Raises:
            ValueError: If tensor_parallel_size exceeds available GPUs.
            RuntimeError: If GPU memory is insufficient for model loading.

        Note:
            The adapter uses bfloat16 precision by default for optimal
            performance-memory trade-off. Adjust dtype in LLM() call if needed.
        """
        # Initialize vLLM engine with multimodal support
        self.llm = LLM(
            model=model_path,
            tensor_parallel_size=tensor_parallel_size,
            gpu_memory_utilization=gpu_memory_utilization,
            max_model_len=max_model_len,
            trust_remote_code=True,
            dtype="bfloat16",
            limit_mm_per_prompt={"image": 1, "video": 0},  # One image at a time
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
    ) -> tuple[str, dict]:
        """
        Generate a response using vLLM's optimized inference engine.

        This method provides high-performance inference by:
        1. Converting image to proper format for vLLM processing
        2. Formatting conversation following Workflow UI's structure
        3. Leveraging vLLM's continuous batching for efficiency
        4. Applying task-specific post-processing to outputs
        5. Tracking detailed token usage statistics

        The vLLM engine automatically handles:
        - Request batching for optimal throughput
        - KV cache management with PagedAttention
        - Efficient memory allocation and scheduling
        - CUDA graph optimization where applicable

        Args:
            query (str): Text prompt/question for the model. May contain
                        task-specific placeholders (e.g., {bbox_ratio}).
            image (Image.Image): PIL Image to analyze. Automatically
                               converted to RGB if needed.
            task_type (str): Task identifier determining post-processing:
                           - CAPTION_TASK: Extract HTML meta descriptions
                           - ACTION_PREDICTION_TASK: Single character MCQ
                           - WEBQA_TASK: Clean answer extraction
                           - ELEMENT_OCR_TASK: Text extraction and cleaning
            sampling_params (Optional[SamplingParams]): Custom generation
                                                       parameters. If None,
                                                       uses default params.

        Returns:
            tuple[str, dict]: A tuple containing:
                - response (str): Post-processed model output for the task
                - num_tokens (dict): Token usage statistics:
                    - 'input': Prompt tokens consumed
                    - 'output': Tokens generated
                    - 'total': Sum of input and output

        Performance tips:
            - Batch multiple requests together for better throughput
            - Adjust sampling temperature for quality vs diversity
            - Use smaller max_tokens for faster response times
            - Enable tensor parallelism for large models
        """
        # Convert image to RGB format
        image = image.convert("RGB")

        # Format input according to Workflow UI's message structure
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},  # Pass PIL Image directly
                    {"type": "text", "text": query},
                ],
            }
        ]

        # Preparation for inference using the processor
        prompt = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

        # Process vision information for vLLM
        image_inputs, video_inputs, video_kwargs = process_vision_info(
            messages, return_video_kwargs=True
        )

        # Prepare multimodal data
        mm_data = {}
        if image_inputs is not None:
            mm_data["image"] = image_inputs
        if video_inputs is not None:
            mm_data["video"] = video_inputs

        # Prepare input for vLLM
        llm_inputs = {
            "prompt": prompt,
            "multi_modal_data": mm_data,
            "mm_processor_kwargs": video_kwargs,
        }

        # Use provided sampling params or default
        params = sampling_params or self.sampling_params

        # Generate using vLLM
        outputs = self.llm.generate(
            [llm_inputs],
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
        Update the default sampling parameters for generation.

        This method allows dynamic adjustment of generation parameters without
        reinitializing the adapter. Useful for experimenting with different
        sampling strategies or adapting to task requirements.

        Common parameters:
        - temperature: Controls randomness (0.0 = deterministic, 1.0 = more random)
        - top_p: Nucleus sampling threshold (0.0-1.0)
        - top_k: Limits vocabulary to top K tokens
        - max_tokens: Maximum tokens to generate
        - repetition_penalty: Penalizes token repetition (1.0 = no penalty)
        - presence_penalty: Penalizes tokens based on presence (-2.0 to 2.0)
        - frequency_penalty: Penalizes tokens based on frequency (-2.0 to 2.0)
        - stop: List of sequences that stop generation
        - stop_token_ids: List of token IDs that stop generation

        Args:
            **kwargs: Any valid SamplingParams arguments. Common options:
                     - temperature (float): Generation randomness (default: 0.7)
                     - top_p (float): Nucleus sampling cutoff (default: 0.95)
                     - max_tokens (int): Max generation length (default: 512)
                     - repetition_penalty (float): Anti-repetition (default: 1.0)

        Example:
            adapter.update_sampling_params(
                temperature=0.3,  # More deterministic
                max_tokens=256,   # Shorter responses
                top_p=0.9        # Slightly more focused
            )

        Note:
            Changes apply to all subsequent generate() calls unless
            overridden by passing sampling_params to generate().
        """
        self.sampling_params = SamplingParams(**kwargs)
