"""
Base LLM Backend Interface

Defines the abstract interface that all LLM backends must implement.
"""

from abc import ABC, abstractmethod
from typing import Tuple, Dict, Any, List, Optional


class BaseLLMBackend(ABC):
    """Abstract base class for LLM backends."""

    def __init__(self, settings):
        """Initialize the backend with settings."""
        self.settings = settings
        self._initialize()

    @abstractmethod
    def _initialize(self) -> None:
        """Initialize the backend (load models, connect to servers, etc.)."""
        pass

    @abstractmethod
    def generate(
        self,
        prompt: str,
        max_tokens: int = 512,
        temperature: float = 0.7,
        top_p: float = 0.9,
        top_k: int = 40,
        repeat_penalty: float = 1.1,
        stop_sequences: Optional[List[str]] = None,
    ) -> Tuple[str, Dict[str, Any]]:
        """
        Generate a response from the LLM.

        Args:
            prompt: The input prompt
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            top_p: Nucleus sampling parameter
            top_k: Top-k sampling parameter
            repeat_penalty: Repetition penalty
            stop_sequences: Optional sequences that stop generation

        Returns:
            Tuple of (generated_text, metadata_dict)
            metadata_dict includes: model, tokens_used, etc.
        """
        pass

    @abstractmethod
    def list_models(self) -> List[str]:
        """List available models for this backend."""
        pass

    @abstractmethod
    def health_check(self) -> bool:
        """Check if the backend is healthy and ready."""
        pass

    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the current model."""
        return {
            "backend": self.__class__.__name__,
            "healthy": self.health_check(),
        }
