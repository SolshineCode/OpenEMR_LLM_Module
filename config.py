"""
OpenEMR LLM Module - Configuration Management

This module provides centralized configuration using Pydantic for validation
and python-dotenv for environment variable loading.
"""

import os
from enum import Enum
from typing import Optional
from functools import lru_cache

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class LLMBackend(str, Enum):
    """Supported LLM inference backends."""
    LLAMACPP = "llamacpp"
    OLLAMA = "ollama"
    OPENAI = "openai"
    HUGGINGFACE = "huggingface"


class Settings(BaseSettings):
    """Application settings with validation."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore"
    )

    # Server Configuration
    flask_host: str = Field(default="127.0.0.1")
    flask_port: int = Field(default=5000, ge=1, le=65535)
    flask_debug: bool = Field(default=False)
    flask_secret_key: str = Field(default="dev-secret-change-in-production")

    # CORS
    cors_origins: str = Field(default="http://localhost,http://127.0.0.1")

    # Rate Limiting
    rate_limit_default: str = Field(default="100/hour")
    rate_limit_generate: str = Field(default="30/minute")

    # LLM Backend Selection
    llm_backend: LLMBackend = Field(default=LLMBackend.LLAMACPP)

    # llama.cpp Configuration
    llamacpp_server_url: str = Field(default="http://localhost:8080")
    llamacpp_api_key: Optional[str] = Field(default=None)

    # Ollama Configuration
    ollama_host: str = Field(default="http://localhost:11434")
    ollama_model: str = Field(default="llama3.2:latest")

    # OpenAI-Compatible Configuration
    openai_api_base: str = Field(default="http://localhost:8080/v1")
    openai_api_key: Optional[str] = Field(default=None)
    openai_model: str = Field(default="local-model")

    # HuggingFace Configuration
    huggingface_model: str = Field(default="microsoft/BioGPT")
    huggingface_cache_dir: str = Field(default="~/.cache/huggingface")
    huggingface_device: str = Field(default="auto")
    huggingface_load_in_8bit: bool = Field(default=False)
    huggingface_load_in_4bit: bool = Field(default=False)

    # Generation Parameters
    max_tokens: int = Field(default=512, ge=1, le=8192)
    temperature: float = Field(default=0.7, ge=0.0, le=2.0)
    top_p: float = Field(default=0.9, ge=0.0, le=1.0)
    top_k: int = Field(default=40, ge=0)
    repeat_penalty: float = Field(default=1.1, ge=0.0, le=2.0)
    context_length: int = Field(default=4096, ge=512, le=131072)

    # System Prompt
    system_prompt: str = Field(
        default="You are a helpful medical assistant integrated with OpenEMR. "
        "Provide accurate, evidence-based information while being clear that you "
        "are an AI assistant and not a replacement for professional medical advice. "
        "Always recommend consulting with healthcare providers for medical decisions."
    )

    # OpenEMR Integration
    openemr_base_url: str = Field(default="https://localhost:9300")
    openemr_api_path: str = Field(default="/apis/default/api")
    openemr_fhir_path: str = Field(default="/apis/default/fhir")
    openemr_client_id: Optional[str] = Field(default=None)
    openemr_client_secret: Optional[str] = Field(default=None)
    openemr_redirect_uri: str = Field(default="http://localhost:5000/oauth/callback")
    openemr_scopes: str = Field(
        default="openid user/Patient.rs user/Encounter.rs user/Observation.rs "
        "user/Condition.rs user/AllergyIntolerance.rs user/MedicationRequest.rs"
    )
    openemr_verify_ssl: bool = Field(default=True)

    # Audit Logging
    audit_log_enabled: bool = Field(default=True)
    audit_log_file: str = Field(default="logs/audit.log")
    audit_log_level: str = Field(default="INFO")

    # Application Logging
    log_level: str = Field(default="INFO")
    log_file: str = Field(default="logs/app.log")
    log_format: str = Field(default="json")

    # Security
    anonymize_patient_data: bool = Field(default=True)
    session_timeout: int = Field(default=3600, ge=60)

    # Fine-tuning
    finetuned_adapter_path: str = Field(default="./models/medical_adapter")
    use_finetuned_model: bool = Field(default=False)

    @field_validator("cors_origins")
    @classmethod
    def parse_cors_origins(cls, v: str) -> str:
        """Validate CORS origins format."""
        origins = [o.strip() for o in v.split(",")]
        for origin in origins:
            if not origin.startswith(("http://", "https://")):
                raise ValueError(f"Invalid CORS origin: {origin}")
        return v

    @property
    def cors_origins_list(self) -> list[str]:
        """Get CORS origins as a list."""
        return [o.strip() for o in self.cors_origins.split(",")]

    @property
    def openemr_scopes_list(self) -> list[str]:
        """Get OpenEMR scopes as a list."""
        return self.openemr_scopes.split()

    @property
    def openemr_api_url(self) -> str:
        """Get full OpenEMR API URL."""
        return f"{self.openemr_base_url}{self.openemr_api_path}"

    @property
    def openemr_fhir_url(self) -> str:
        """Get full OpenEMR FHIR URL."""
        return f"{self.openemr_base_url}{self.openemr_fhir_path}"


@lru_cache
def get_settings() -> Settings:
    """Get cached settings instance."""
    return Settings()


# Export settings instance
settings = get_settings()
