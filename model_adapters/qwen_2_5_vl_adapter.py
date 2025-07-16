"""
Adapter implementation for Qwen 2.5 VL vision-language models.

This module provides the Qwen25VLAdapter class which handles model inference
for Qwen 2.5 VL models. It manages image encoding, prompt formatting, and
task-specific response post-processing for various visual web understanding tasks.
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


class Qwen25VLAdapter(BaseAdapter):
    """
    Adapter for Qwen 2.5 VL vision-language models.

    This class provides an interface to Qwen 2.5 VL models. It handles image
    processing, conversation formatting, and model inference specific to
    Qwen 2.5 VL's architecture.

    Attributes:
        model: The Qwen 2.5 VL model instance for inference.
        processor: The processor for text and image processing.
    """

    def __init__(
        self,
        model: Qwen2_5_VLForConditionalGeneration,
        processor: AutoProcessor,
    ):
        """
        Initialize the Qwen 2.5 VL adapter.

        Args:
            model: The Qwen 2.5 VL model instance.
            processor: The processor for text and image processing.
        """
        super().__init__(model, None)
        self.processor = processor

    def generate(
        self,
        query: str,
        image: Image,
        task_type: str,
    ) -> str:
        """
        Generate a response using Qwen 2.5 VL's vision-language model.

        This method formats the input for Qwen 2.5 VL, processes the image,
        and generates a response using the model's inference pipeline.

        Args:
            query: The text query/prompt to send to the model.
            image: The PIL Image object to process.
            task_type: The type of task being performed (affects response post-processing).

        Returns:
            The generated text response from the model, post-processed based on task type.
        """
        # Convert image to RGB format
        image = image.convert("RGB")

        # Convert PIL Image to base64 string
        image_buffer = io.BytesIO()
        image.save(image_buffer, format="PNG")
        image_base64 = base64.b64encode(image_buffer.getvalue()).decode("utf-8")

        # Format input according to Qwen 2.5 VL's message structure
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
