"""
Ollama Backend

Connects to Ollama for easy local LLM management and inference.

Ollama provides:
- Simple model management (pull, list, delete)
- Easy API for inference
- Support for many models (Llama, Mistral, Phi, CodeLlama, etc.)
- Built-in quantization and optimization

To install Ollama:
    curl -fsSL https://ollama.com/install.sh | sh

To pull a model:
    ollama pull llama3.2
    ollama pull mistral
    ollama pull medllama2  # Medical-focused model
"""

import structlog
from typing import Tuple, Dict, Any, List, Optional

try:
    import ollama
    OLLAMA_AVAILABLE = True
except ImportError:
    OLLAMA_AVAILABLE = False

import requests

from .base import BaseLLMBackend

logger = structlog.get_logger(__name__)


class OllamaBackend(BaseLLMBackend):
    """Backend for Ollama."""

    def _initialize(self) -> None:
        """Initialize Ollama client."""
        self.host = self.settings.ollama_host
        self.model = self.settings.ollama_model
        self.timeout = 120

        if OLLAMA_AVAILABLE:
            self.client = ollama.Client(host=self.host)
        else:
            self.client = None
            logger.warning("ollama_package_not_installed", using_http_fallback=True)

        logger.info(
            "ollama_backend_initialized",
            host=self.host,
            model=self.model,
        )

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
        """Generate text using Ollama."""

        options = {
            "num_predict": max_tokens,
            "temperature": temperature,
            "top_p": top_p,
            "top_k": top_k,
            "repeat_penalty": repeat_penalty,
        }

        if stop_sequences:
            options["stop"] = stop_sequences

        try:
            if self.client and OLLAMA_AVAILABLE:
                # Use official Ollama client
                response = self.client.generate(
                    model=self.model,
                    prompt=prompt,
                    options=options,
                    stream=False,
                )

                generated_text = response.get("response", "")
                metadata = {
                    "model": response.get("model", self.model),
                    "tokens_used": response.get("eval_count", 0),
                    "tokens_evaluated": response.get("prompt_eval_count", 0),
                    "generation_time_ms": response.get("eval_duration", 0) / 1_000_000,
                }
            else:
                # HTTP fallback
                generated_text, metadata = self._generate_http(prompt, options)

            logger.debug(
                "ollama_generation_complete",
                model=self.model,
                tokens_generated=metadata.get("tokens_used", 0),
            )

            return generated_text.strip(), metadata

        except Exception as e:
            logger.error("ollama_generation_failed", error=str(e))
            raise RuntimeError(f"Ollama generation failed: {e}")

    def _generate_http(
        self, prompt: str, options: Dict
    ) -> Tuple[str, Dict[str, Any]]:
        """HTTP fallback for generation."""
        url = f"{self.host}/api/generate"

        payload = {
            "model": self.model,
            "prompt": prompt,
            "options": options,
            "stream": False,
        }

        response = requests.post(url, json=payload, timeout=self.timeout)
        response.raise_for_status()
        result = response.json()

        return result.get("response", ""), {
            "model": result.get("model", self.model),
            "tokens_used": result.get("eval_count", 0),
        }

    def list_models(self) -> List[str]:
        """List available Ollama models."""
        try:
            if self.client and OLLAMA_AVAILABLE:
                models = self.client.list()
                return [m["name"] for m in models.get("models", [])]
            else:
                # HTTP fallback
                response = requests.get(f"{self.host}/api/tags", timeout=10)
                response.raise_for_status()
                return [m["name"] for m in response.json().get("models", [])]
        except Exception as e:
            logger.error("ollama_list_models_failed", error=str(e))
            return []

    def health_check(self) -> bool:
        """Check if Ollama is healthy."""
        try:
            response = requests.get(f"{self.host}/api/tags", timeout=5)
            return response.ok
        except Exception:
            return False

    def pull_model(self, model_name: str) -> bool:
        """Pull a model from Ollama registry."""
        try:
            if self.client and OLLAMA_AVAILABLE:
                self.client.pull(model_name)
            else:
                response = requests.post(
                    f"{self.host}/api/pull",
                    json={"name": model_name},
                    timeout=600,  # Model downloads can be slow
                )
                response.raise_for_status()

            logger.info("ollama_model_pulled", model=model_name)
            return True
        except Exception as e:
            logger.error("ollama_pull_failed", model=model_name, error=str(e))
            return False

    def chat(
        self,
        messages: List[Dict[str, str]],
        max_tokens: int = 512,
        temperature: float = 0.7,
    ) -> Tuple[str, Dict[str, Any]]:
        """
        Chat-style generation with message history.

        Args:
            messages: List of {"role": "user/assistant/system", "content": "..."}
        """
        options = {
            "num_predict": max_tokens,
            "temperature": temperature,
        }

        try:
            if self.client and OLLAMA_AVAILABLE:
                response = self.client.chat(
                    model=self.model,
                    messages=messages,
                    options=options,
                    stream=False,
                )
                return response["message"]["content"], {
                    "model": response.get("model", self.model),
                }
            else:
                response = requests.post(
                    f"{self.host}/api/chat",
                    json={
                        "model": self.model,
                        "messages": messages,
                        "options": options,
                        "stream": False,
                    },
                    timeout=self.timeout,
                )
                response.raise_for_status()
                result = response.json()
                return result["message"]["content"], {
                    "model": result.get("model", self.model),
                }
        except Exception as e:
            logger.error("ollama_chat_failed", error=str(e))
            raise


# Recommended medical models for Ollama
RECOMMENDED_MEDICAL_MODELS = [
    "medllama2",  # Medical fine-tuned Llama 2
    "meditron",   # Medical Llama variant
    "llama3.2",   # General purpose, good for medical with prompting
    "mistral",    # Good general model
    "phi3",       # Microsoft's efficient model
]
