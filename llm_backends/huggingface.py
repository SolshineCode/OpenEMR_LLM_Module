"""
HuggingFace Transformers Backend

Uses the HuggingFace Transformers library for local inference.
This is a fallback option when llama.cpp or Ollama aren't available.

Supports:
- Any causal LM from HuggingFace Hub
- Quantization (8-bit, 4-bit) via bitsandbytes
- GPU acceleration
- Loading fine-tuned adapters (LoRA)

Recommended medical models:
- microsoft/BioGPT
- epfl-llm/meditron-7b
- medalpaca/medalpaca-7b
- johnsnowlabs/JSL-MedLlama-3-8B-v2.0
"""

import os
import structlog
from typing import Tuple, Dict, Any, List, Optional

from .base import BaseLLMBackend

logger = structlog.get_logger(__name__)

# Lazy imports to avoid loading heavy ML libraries if not needed
transformers = None
torch = None


def _lazy_import():
    """Lazily import heavy dependencies."""
    global transformers, torch

    if transformers is None:
        try:
            import transformers as tf
            import torch as th
            transformers = tf
            torch = th
        except ImportError as e:
            raise ImportError(
                "HuggingFace backend requires transformers and torch. "
                "Install with: pip install transformers torch"
            ) from e


class HuggingFaceBackend(BaseLLMBackend):
    """Backend for HuggingFace Transformers."""

    def _initialize(self) -> None:
        """Initialize the HuggingFace model."""
        _lazy_import()

        self.model_name = self.settings.huggingface_model
        self.cache_dir = os.path.expanduser(self.settings.huggingface_cache_dir)
        self.device = self.settings.huggingface_device
        self.load_in_8bit = self.settings.huggingface_load_in_8bit
        self.load_in_4bit = self.settings.huggingface_load_in_4bit

        self.model = None
        self.tokenizer = None
        self._loaded = False

        logger.info(
            "huggingface_backend_initialized",
            model=self.model_name,
            device=self.device,
            quantization="8bit" if self.load_in_8bit else ("4bit" if self.load_in_4bit else "none"),
        )

    def _load_model(self) -> None:
        """Load the model and tokenizer (lazy loading)."""
        if self._loaded:
            return

        logger.info("loading_huggingface_model", model=self.model_name)

        # Determine device
        if self.device == "auto":
            device_map = "auto"
        elif self.device == "cpu":
            device_map = {"": "cpu"}
        else:
            device_map = {"": self.device}

        # Build model loading kwargs
        model_kwargs = {
            "cache_dir": self.cache_dir,
            "device_map": device_map,
            "trust_remote_code": True,
        }

        # Add quantization if requested
        if self.load_in_8bit or self.load_in_4bit:
            try:
                from transformers import BitsAndBytesConfig

                if self.load_in_4bit:
                    model_kwargs["quantization_config"] = BitsAndBytesConfig(
                        load_in_4bit=True,
                        bnb_4bit_compute_dtype=torch.float16,
                        bnb_4bit_use_double_quant=True,
                        bnb_4bit_quant_type="nf4",
                    )
                else:
                    model_kwargs["quantization_config"] = BitsAndBytesConfig(
                        load_in_8bit=True,
                    )
            except ImportError:
                logger.warning(
                    "bitsandbytes_not_available",
                    msg="Quantization disabled - install bitsandbytes",
                )

        # Load tokenizer
        self.tokenizer = transformers.AutoTokenizer.from_pretrained(
            self.model_name,
            cache_dir=self.cache_dir,
            trust_remote_code=True,
        )

        # Set padding token if not set
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # Load model
        self.model = transformers.AutoModelForCausalLM.from_pretrained(
            self.model_name,
            **model_kwargs,
        )

        # Load fine-tuned adapter if configured
        if self.settings.use_finetuned_model:
            adapter_path = self.settings.finetuned_adapter_path
            if os.path.exists(adapter_path):
                try:
                    from peft import PeftModel
                    self.model = PeftModel.from_pretrained(self.model, adapter_path)
                    logger.info("loaded_finetuned_adapter", path=adapter_path)
                except ImportError:
                    logger.warning("peft_not_installed", msg="Cannot load adapter")
                except Exception as e:
                    logger.error("adapter_load_failed", error=str(e))

        self._loaded = True
        logger.info("huggingface_model_loaded", model=self.model_name)

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
        """Generate text using HuggingFace model."""
        self._load_model()

        # Tokenize input
        inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True)

        # Move to model device
        if hasattr(self.model, "device"):
            inputs = {k: v.to(self.model.device) for k, v in inputs.items()}

        # Build generation config
        gen_config = {
            "max_new_tokens": max_tokens,
            "temperature": temperature if temperature > 0 else 1.0,
            "top_p": top_p,
            "top_k": top_k,
            "repetition_penalty": repeat_penalty,
            "do_sample": temperature > 0,
            "pad_token_id": self.tokenizer.pad_token_id,
            "eos_token_id": self.tokenizer.eos_token_id,
        }

        # Add stopping criteria for stop sequences
        if stop_sequences:
            stop_ids = [
                self.tokenizer.encode(seq, add_special_tokens=False)
                for seq in stop_sequences
            ]
            gen_config["stop_sequences"] = stop_ids

        try:
            with torch.no_grad():
                outputs = self.model.generate(**inputs, **gen_config)

            # Decode output
            generated_ids = outputs[0][inputs["input_ids"].shape[1]:]
            generated_text = self.tokenizer.decode(
                generated_ids,
                skip_special_tokens=True,
            )

            # Handle stop sequences manually
            if stop_sequences:
                for seq in stop_sequences:
                    if seq in generated_text:
                        generated_text = generated_text.split(seq)[0]

            metadata = {
                "model": self.model_name,
                "tokens_used": len(generated_ids),
                "input_tokens": inputs["input_ids"].shape[1],
            }

            logger.debug(
                "huggingface_generation_complete",
                tokens_generated=len(generated_ids),
            )

            return generated_text.strip(), metadata

        except Exception as e:
            logger.error("huggingface_generation_failed", error=str(e))
            raise RuntimeError(f"HuggingFace generation failed: {e}")

    def list_models(self) -> List[str]:
        """List recommended medical models."""
        return [
            self.model_name,
            "microsoft/BioGPT",
            "epfl-llm/meditron-7b",
            "medalpaca/medalpaca-7b",
            "johnsnowlabs/JSL-MedLlama-3-8B-v2.0",
        ]

    def health_check(self) -> bool:
        """Check if model can be loaded."""
        try:
            if not self._loaded:
                # Just check if tokenizer can be loaded
                transformers.AutoTokenizer.from_pretrained(
                    self.model_name,
                    cache_dir=self.cache_dir,
                )
            return True
        except Exception:
            return False

    def get_model_info(self) -> Dict[str, Any]:
        """Get model information."""
        info = super().get_model_info()
        info.update({
            "model_name": self.model_name,
            "loaded": self._loaded,
            "quantization": "8bit" if self.load_in_8bit else (
                "4bit" if self.load_in_4bit else "none"
            ),
        })
        return info

    def unload_model(self) -> None:
        """Unload the model to free memory."""
        if self.model is not None:
            del self.model
            self.model = None

        if self.tokenizer is not None:
            del self.tokenizer
            self.tokenizer = None

        self._loaded = False

        # Clear CUDA cache if available
        if torch is not None and torch.cuda.is_available():
            torch.cuda.empty_cache()

        logger.info("huggingface_model_unloaded")


# Recommended medical models from HuggingFace
RECOMMENDED_MODELS = {
    "general": [
        "microsoft/BioGPT",
        "microsoft/BioGPT-Large",
    ],
    "clinical": [
        "epfl-llm/meditron-7b",
        "epfl-llm/meditron-70b",
        "johnsnowlabs/JSL-MedLlama-3-8B-v2.0",
    ],
    "qa": [
        "medalpaca/medalpaca-7b",
        "medalpaca/medalpaca-13b",
    ],
    "multilingual": [
        "bigscience/bloom-7b1",  # Can handle medical text in multiple languages
    ],
}
