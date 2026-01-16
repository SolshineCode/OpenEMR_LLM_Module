"""
LLM Backends Module

Provides unified interface for multiple LLM inference backends:
- llama.cpp server
- Ollama
- OpenAI-compatible APIs
- HuggingFace Transformers
"""

from .base import BaseLLMBackend
from .llamacpp import LlamaCppBackend
from .ollama import OllamaBackend
from .openai_compat import OpenAICompatBackend
from .huggingface import HuggingFaceBackend

from config import LLMBackend, settings


def get_llm_backend(backend_type: LLMBackend) -> BaseLLMBackend:
    """Factory function to get the appropriate LLM backend."""
    backends = {
        LLMBackend.LLAMACPP: LlamaCppBackend,
        LLMBackend.OLLAMA: OllamaBackend,
        LLMBackend.OPENAI: OpenAICompatBackend,
        LLMBackend.HUGGINGFACE: HuggingFaceBackend,
    }

    backend_class = backends.get(backend_type)
    if not backend_class:
        raise ValueError(f"Unknown backend type: {backend_type}")

    return backend_class(settings)


__all__ = [
    "BaseLLMBackend",
    "LlamaCppBackend",
    "OllamaBackend",
    "OpenAICompatBackend",
    "HuggingFaceBackend",
    "get_llm_backend",
]
