"""
Base adapter module for vision-language models in VisualWebBench.

This module provides the abstract base class that defines the standardized
interface for all model adapters in the VisualWebBench evaluation framework.
It ensures consistent interaction patterns across different vision-language
models, inference engines, and deployment strategies.

The adapter pattern allows the benchmark to:
- Support diverse model architectures (transformers, API-based, custom)
- Abstract away model-specific implementation details
- Provide consistent evaluation interfaces
- Enable easy addition of new models
- Support multiple inference backends (PyTorch, vLLM, TensorRT, etc.)

Implementation guide for new adapters:
1. Inherit from BaseAdapter
2. Implement the generate() method
3. Handle task-specific formatting in generate()
4. Return both response and token statistics
5. Add adapter to model_adapters/__init__.py
"""
from abc import ABC, abstractmethod

from PIL import Image


class BaseAdapter(ABC):
    """
    Abstract base class for vision-language model adapters.

    This class establishes the contract that all model adapters must fulfill
    to integrate with the VisualWebBench evaluation framework. It provides a
    unified interface for interacting with diverse vision-language models,
    regardless of their underlying implementation or inference backend.

    The adapter pattern enables:
    - Consistent API across different model families
    - Flexible backend implementations (local, API, distributed)
    - Task-agnostic interface with task-specific handling
    - Performance metric collection (token usage, latency)
    - Easy benchmarking of new models

    Attributes:
        model (Optional[Any]): The underlying model instance. Can be:
                              - Transformers model (e.g., AutoModel)
                              - vLLM engine instance
                              - API client object
                              - None for stateless adapters
        tokenizer (Optional[Any]): The tokenizer/processor instance. Can be:
                                  - Transformers tokenizer
                                  - Custom processor
                                  - None for API-based models

    Abstract Methods:
        generate: Must be implemented by all subclasses to handle inference

    Example Implementation:
        class MyModelAdapter(BaseAdapter):
            def generate(self, query, image, task_type):
                # Process inputs
                inputs = self.preprocess(query, image)
                # Run inference
                outputs = self.model(inputs)
                # Post-process based on task_type
                response = self.postprocess(outputs, task_type)
                return response, token_stats
    """

    def __init__(
        self,
        model=None,
        tokenizer=None,
    ):
        """
        Initialize the base adapter with optional model and tokenizer.

        This constructor provides flexibility for different adapter implementations:
        - Local models: Pass both model and tokenizer instances
        - API-based models: Pass None for both and handle in subclass
        - Hybrid approaches: Pass only what's needed

        Args:
            model (Optional[Any]): The underlying model instance. Common types:
                                  - transformers.PreTrainedModel for PyTorch models
                                  - vllm.LLM for vLLM engine
                                  - Custom model wrappers
                                  - None for API-based or lazy-loaded models
            tokenizer (Optional[Any]): The tokenizer/processor instance. Common types:
                                      - transformers.PreTrainedTokenizer
                                      - transformers.ProcessorMixin
                                      - Custom preprocessing classes
                                      - None for models with built-in processing

        Note:
            Subclasses may extend this constructor to accept additional
            configuration parameters specific to their implementation.
        """
        self.model = model
        self.tokenizer = tokenizer

    @abstractmethod
    def generate(
        self,
        query: str,
        image: Image,
        task_type: str,
    ) -> tuple[str, dict]:
        """
        Generate a response based on the input query and image.

        This abstract method defines the core inference interface that all
        adapters must implement. It handles multimodal input (text + image)
        and produces task-appropriate outputs with performance metrics.

        The method should:
        1. Process the text query and image into model-specific format
        2. Run inference using the underlying model/API
        3. Apply task-specific post-processing to the output
        4. Track token usage and other metrics
        5. Return processed response and statistics

        Args:
            query (str): The text prompt/question to process. May contain
                        task-specific placeholders like {bbox_ratio} or {question}
                        that are already filled by the benchmark framework.
            image (Image.Image): PIL Image object to analyze. The image should
                               be processed according to model requirements
                               (resizing, normalization, encoding, etc.).
            task_type (str): Task identifier from utils.constants that determines
                           output formatting. Supported types:
                           - CAPTION_TASK: Web page captioning
                           - HEADING_OCR_TASK: Heading text extraction
                           - WEBQA_TASK: Question answering about web pages
                           - ELEMENT_OCR_TASK: UI element text extraction
                           - ELEMENT_GROUND_TASK: Element localization
                           - ACTION_PREDICTION_TASK: Next action prediction
                           - ACTION_GROUND_TASK: Action grounding

        Returns:
            tuple[str, dict]: A tuple containing:
                - response (str): The model's generated text, post-processed
                                according to task_type requirements
                - stats (dict): Performance metrics with keys:
                               - 'input': Number of input tokens
                               - 'output': Number of generated tokens  
                               - 'total': Sum of input and output tokens
                               Additional keys may include latency, etc.

        Raises:
            NotImplementedError: If the task_type is not supported
            Exception: Model-specific errors (e.g., API failures, OOM)

        Example:
            response, tokens = adapter.generate(
                "What is the main heading?",
                image,
                "heading_ocr"
            )
        """
        pass

    def __call__(
        self,
        query: str,
        image: str,
        task_type: str,
    ) -> tuple[str, dict]:
        """
        Make the adapter callable, providing a convenient interface.

        This method allows treating the adapter instance as a callable function,
        delegating to the generate() method. It maintains the same interface
        and behavior as generate() for consistency.

        This enables usage patterns like:
            adapter = MyModelAdapter(model, tokenizer)
            response, stats = adapter(query, image, task_type)

        Args:
            query (str): The text query/prompt to send to the model.
                        Same format as generate() method.
            image (Image.Image): The PIL Image object to process.
                               Same requirements as generate() method.
            task_type (str): The type of task being performed.
                           Must be a valid task constant.

        Returns:
            tuple[str, dict]: Same return format as generate():
                            - response string and token statistics

        Note:
            This is a convenience method. For direct control or custom
            parameters, use the generate() method directly.
        """
        return self.generate(query, image, task_type)
