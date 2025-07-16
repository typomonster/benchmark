"""
Base adapter module for vision-language models.

This module provides the abstract base class that defines the interface
for all model adapters in the VisualWebBench evaluation framework.
"""
from abc import ABC, abstractmethod

from PIL import Image


class BaseAdapter(ABC):
    """
    Abstract base class for vision-language model adapters.

    This class defines the interface that all model adapters must implement
    to work with the 10xVisualWebBench evaluation framework. It provides a
    standardized way to interact with different vision-language models.

    Attributes:
        model: The underlying model instance (can be None for API-based models).
        tokenizer: The tokenizer instance (can be None for API-based models).
    """

    def __init__(
        self,
        model=None,
        tokenizer=None,
    ):
        """
        Initialize the base adapter.

        Args:
            model: The underlying model instance. Can be None for API-based models.
            tokenizer: The tokenizer instance. Can be None for API-based models.
        """
        self.model = model
        self.tokenizer = tokenizer

    @abstractmethod
    def generate(
        self,
        query: str,
        image: Image,
        task_type: str,
    ) -> str:
        """
        Generate a response based on the input query and image.

        This method must be implemented by all concrete adapter classes.

        Args:
            query: The text query/prompt to send to the model.
            image: The PIL Image object to process.
            task_type: The type of task being performed (e.g., 'web_caption', 'element_ocr').

        Returns:
            The generated text response from the model.
        """
        pass

    def __call__(
        self,
        query: str,
        image: str,
        task_type: str,
    ) -> str:
        """
        Make the adapter callable, delegating to the generate method.

        This method provides a convenient way to call the adapter instance directly.

        Args:
            query: The text query/prompt to send to the model.
            image: The image input (typically a PIL Image or image path).
            task_type: The type of task being performed.

        Returns:
            The generated text response from the model.
        """
        return self.generate(query, image, task_type)
