"""
OpenAI-Compatible API Backend

Supports any API that implements the OpenAI chat/completions interface:
- llama.cpp server (with --api-key flag)
- vLLM
- LocalAI
- Text Generation Inference (TGI)
- LM Studio
- Any other OpenAI-compatible endpoint

This provides flexibility to switch between different inference servers
without changing code.
"""

import structlog
from typing import Tuple, Dict, Any, List, Optional

try:
    from openai import OpenAI
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False

import requests

from .base import BaseLLMBackend

logger = structlog.get_logger(__name__)


class OpenAICompatBackend(BaseLLMBackend):
    """Backend for OpenAI-compatible APIs."""

    def _initialize(self) -> None:
        """Initialize OpenAI client."""
        self.base_url = self.settings.openai_api_base.rstrip("/")
        self.api_key = self.settings.openai_api_key or "not-needed"
        self.model = self.settings.openai_model
        self.timeout = 120

        if OPENAI_AVAILABLE:
            self.client = OpenAI(
                base_url=self.base_url,
                api_key=self.api_key,
                timeout=self.timeout,
            )
        else:
            self.client = None
            logger.warning("openai_package_not_installed", using_http_fallback=True)

        logger.info(
            "openai_compat_backend_initialized",
            base_url=self.base_url,
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
        """Generate text using OpenAI-compatible API."""

        # Build messages for chat endpoint
        messages = [{"role": "user", "content": prompt}]

        try:
            if self.client and OPENAI_AVAILABLE:
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    top_p=top_p,
                    stop=stop_sequences,
                    frequency_penalty=repeat_penalty - 1.0,  # OpenAI uses different scale
                )

                generated_text = response.choices[0].message.content
                metadata = {
                    "model": response.model,
                    "tokens_used": response.usage.completion_tokens if response.usage else 0,
                    "prompt_tokens": response.usage.prompt_tokens if response.usage else 0,
                    "total_tokens": response.usage.total_tokens if response.usage else 0,
                }
            else:
                # HTTP fallback
                generated_text, metadata = self._generate_http(
                    messages, max_tokens, temperature, top_p, stop_sequences
                )

            logger.debug(
                "openai_compat_generation_complete",
                model=self.model,
                tokens_generated=metadata.get("tokens_used", 0),
            )

            return generated_text.strip() if generated_text else "", metadata

        except Exception as e:
            logger.error("openai_compat_generation_failed", error=str(e))
            raise RuntimeError(f"OpenAI-compatible API generation failed: {e}")

    def _generate_http(
        self,
        messages: List[Dict],
        max_tokens: int,
        temperature: float,
        top_p: float,
        stop_sequences: Optional[List[str]],
    ) -> Tuple[str, Dict[str, Any]]:
        """HTTP fallback for generation."""
        url = f"{self.base_url}/chat/completions"

        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}",
        }

        payload = {
            "model": self.model,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "top_p": top_p,
        }

        if stop_sequences:
            payload["stop"] = stop_sequences

        response = requests.post(
            url, json=payload, headers=headers, timeout=self.timeout
        )
        response.raise_for_status()
        result = response.json()

        choice = result["choices"][0]
        generated_text = choice["message"]["content"]

        usage = result.get("usage", {})
        metadata = {
            "model": result.get("model", self.model),
            "tokens_used": usage.get("completion_tokens", 0),
            "prompt_tokens": usage.get("prompt_tokens", 0),
            "total_tokens": usage.get("total_tokens", 0),
        }

        return generated_text, metadata

    def list_models(self) -> List[str]:
        """List available models."""
        try:
            if self.client and OPENAI_AVAILABLE:
                models = self.client.models.list()
                return [m.id for m in models.data]
            else:
                response = requests.get(
                    f"{self.base_url}/models",
                    headers={"Authorization": f"Bearer {self.api_key}"},
                    timeout=10,
                )
                response.raise_for_status()
                return [m["id"] for m in response.json().get("data", [])]
        except Exception as e:
            logger.error("openai_compat_list_models_failed", error=str(e))
            return [self.model]  # Return configured model as fallback

    def health_check(self) -> bool:
        """Check if the API is healthy."""
        try:
            if self.client and OPENAI_AVAILABLE:
                self.client.models.list()
                return True
            else:
                response = requests.get(
                    f"{self.base_url}/models",
                    headers={"Authorization": f"Bearer {self.api_key}"},
                    timeout=5,
                )
                return response.ok
        except Exception:
            return False

    def chat(
        self,
        messages: List[Dict[str, str]],
        max_tokens: int = 512,
        temperature: float = 0.7,
        system_prompt: Optional[str] = None,
    ) -> Tuple[str, Dict[str, Any]]:
        """
        Chat-style generation with message history.

        Args:
            messages: List of {"role": "user/assistant/system", "content": "..."}
            system_prompt: Optional system prompt to prepend
        """
        full_messages = []

        if system_prompt:
            full_messages.append({"role": "system", "content": system_prompt})

        full_messages.extend(messages)

        try:
            if self.client and OPENAI_AVAILABLE:
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=full_messages,
                    max_tokens=max_tokens,
                    temperature=temperature,
                )
                return response.choices[0].message.content, {
                    "model": response.model,
                    "tokens_used": response.usage.completion_tokens if response.usage else 0,
                }
            else:
                return self._generate_http(
                    full_messages, max_tokens, temperature, 0.9, None
                )
        except Exception as e:
            logger.error("openai_compat_chat_failed", error=str(e))
            raise

    def embeddings(self, text: str) -> List[float]:
        """Get embeddings for text (if supported by the API)."""
        try:
            if self.client and OPENAI_AVAILABLE:
                response = self.client.embeddings.create(
                    model=self.model,
                    input=text,
                )
                return response.data[0].embedding
            else:
                response = requests.post(
                    f"{self.base_url}/embeddings",
                    json={"model": self.model, "input": text},
                    headers={"Authorization": f"Bearer {self.api_key}"},
                    timeout=30,
                )
                response.raise_for_status()
                return response.json()["data"][0]["embedding"]
        except Exception as e:
            logger.error("openai_compat_embeddings_failed", error=str(e))
            raise
