"""
llama.cpp Server Backend

Connects to a running llama.cpp server for efficient local LLM inference.

llama.cpp server provides:
- Efficient CPU/GPU inference with GGUF models
- OpenAI-compatible API endpoints
- Low memory footprint with quantization
- Support for many model architectures (Llama, Mistral, Phi, etc.)

To start llama.cpp server:
    llama-server -m model.gguf --port 8080 --ctx-size 4096

Or with GPU:
    llama-server -m model.gguf --port 8080 --ctx-size 4096 -ngl 35
"""

import requests
import structlog
from typing import Tuple, Dict, Any, List, Optional

from .base import BaseLLMBackend

logger = structlog.get_logger(__name__)


class LlamaCppBackend(BaseLLMBackend):
    """Backend for llama.cpp server."""

    def _initialize(self) -> None:
        """Initialize connection to llama.cpp server."""
        self.base_url = self.settings.llamacpp_server_url.rstrip("/")
        self.api_key = self.settings.llamacpp_api_key
        self.timeout = 120  # seconds

        # Build headers
        self.headers = {"Content-Type": "application/json"}
        if self.api_key:
            self.headers["Authorization"] = f"Bearer {self.api_key}"

        logger.info(
            "llamacpp_backend_initialized",
            server_url=self.base_url,
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
        """Generate text using llama.cpp server."""

        # llama.cpp server uses the /completion endpoint
        url = f"{self.base_url}/completion"

        payload = {
            "prompt": prompt,
            "n_predict": max_tokens,
            "temperature": temperature,
            "top_p": top_p,
            "top_k": top_k,
            "repeat_penalty": repeat_penalty,
            "stop": stop_sequences or ["User:", "\n\nUser:", "</s>"],
            "stream": False,
        }

        try:
            response = requests.post(
                url,
                json=payload,
                headers=self.headers,
                timeout=self.timeout,
            )
            response.raise_for_status()
            result = response.json()

            generated_text = result.get("content", "")

            # Extract metadata
            metadata = {
                "model": result.get("model", "llama.cpp"),
                "tokens_used": result.get("tokens_predicted", 0),
                "tokens_evaluated": result.get("tokens_evaluated", 0),
                "generation_time_ms": result.get("timings", {}).get("predicted_ms", 0),
            }

            logger.debug(
                "llamacpp_generation_complete",
                tokens_generated=metadata["tokens_used"],
            )

            return generated_text.strip(), metadata

        except requests.exceptions.ConnectionError:
            logger.error("llamacpp_connection_failed", url=url)
            raise ConnectionError(
                f"Cannot connect to llama.cpp server at {self.base_url}. "
                "Make sure the server is running with: llama-server -m model.gguf"
            )
        except requests.exceptions.Timeout:
            logger.error("llamacpp_timeout", url=url)
            raise TimeoutError("llama.cpp server request timed out")
        except requests.exceptions.HTTPError as e:
            logger.error("llamacpp_http_error", status=e.response.status_code)
            raise RuntimeError(f"llama.cpp server error: {e.response.text}")

    def list_models(self) -> List[str]:
        """List available models (llama.cpp serves one model at a time)."""
        try:
            # Check props endpoint for model info
            response = requests.get(
                f"{self.base_url}/props",
                headers=self.headers,
                timeout=10,
            )
            if response.ok:
                props = response.json()
                return [props.get("model", "loaded-model")]
        except Exception:
            pass

        # Fallback - just indicate server is available
        if self.health_check():
            return ["llama.cpp-model"]
        return []

    def health_check(self) -> bool:
        """Check if llama.cpp server is healthy."""
        try:
            response = requests.get(
                f"{self.base_url}/health",
                headers=self.headers,
                timeout=5,
            )
            return response.ok
        except Exception:
            return False

    def get_model_info(self) -> Dict[str, Any]:
        """Get detailed model information."""
        info = super().get_model_info()

        try:
            response = requests.get(
                f"{self.base_url}/props",
                headers=self.headers,
                timeout=5,
            )
            if response.ok:
                props = response.json()
                info.update({
                    "model": props.get("model"),
                    "context_length": props.get("n_ctx"),
                    "total_slots": props.get("total_slots"),
                })
        except Exception:
            pass

        return info


class LlamaCppEmbeddingsBackend:
    """
    Embeddings support for llama.cpp server.

    Useful for RAG (Retrieval Augmented Generation) workflows.
    """

    def __init__(self, settings):
        self.base_url = settings.llamacpp_server_url.rstrip("/")
        self.headers = {"Content-Type": "application/json"}
        if settings.llamacpp_api_key:
            self.headers["Authorization"] = f"Bearer {settings.llamacpp_api_key}"

    def embed(self, text: str) -> List[float]:
        """Get embedding for text."""
        response = requests.post(
            f"{self.base_url}/embedding",
            json={"content": text},
            headers=self.headers,
            timeout=30,
        )
        response.raise_for_status()
        return response.json().get("embedding", [])

    def embed_batch(self, texts: List[str]) -> List[List[float]]:
        """Get embeddings for multiple texts."""
        return [self.embed(text) for text in texts]
