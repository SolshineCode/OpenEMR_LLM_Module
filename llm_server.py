"""
OpenEMR LLM Module - Modern LLM Server

A Flask-based server supporting multiple LLM backends:
- llama.cpp server (recommended for local inference)
- Ollama
- OpenAI-compatible APIs (vLLM, LocalAI, etc.)
- HuggingFace Transformers (fallback)

Features:
- Configurable backends
- Rate limiting
- Audit logging
- OpenEMR integration
- Input validation
- Error handling
"""

import os
import sys
import time
import uuid
import logging
from datetime import datetime
from typing import Optional
from pathlib import Path

import structlog
from flask import Flask, request, jsonify, g
from flask_cors import CORS
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address

from config import settings, LLMBackend
from llm_backends import get_llm_backend
from openemr_client import OpenEMRClient
from utils.logging_config import setup_logging
from utils.security import sanitize_input, anonymize_phi

# Setup logging
setup_logging()
logger = structlog.get_logger(__name__)

# Create Flask app
app = Flask(__name__)
app.secret_key = settings.flask_secret_key

# Configure CORS
CORS(app, origins=settings.cors_origins_list)

# Configure rate limiting
limiter = Limiter(
    key_func=get_remote_address,
    app=app,
    default_limits=[settings.rate_limit_default],
    storage_uri="memory://",
)

# Initialize LLM backend
llm_backend = None
openemr_client = None


def get_llm():
    """Get or initialize the LLM backend."""
    global llm_backend
    if llm_backend is None:
        llm_backend = get_llm_backend(settings.llm_backend)
        logger.info("llm_backend_initialized", backend=settings.llm_backend.value)
    return llm_backend


def get_openemr():
    """Get or initialize the OpenEMR client."""
    global openemr_client
    if openemr_client is None and settings.openemr_client_id:
        openemr_client = OpenEMRClient(settings)
        logger.info("openemr_client_initialized")
    return openemr_client


@app.before_request
def before_request():
    """Add request metadata."""
    g.request_id = str(uuid.uuid4())
    g.start_time = time.time()
    logger.info(
        "request_started",
        request_id=g.request_id,
        method=request.method,
        path=request.path,
        remote_addr=request.remote_addr,
    )


@app.after_request
def after_request(response):
    """Log request completion."""
    duration = time.time() - g.start_time
    logger.info(
        "request_completed",
        request_id=g.request_id,
        status_code=response.status_code,
        duration_ms=round(duration * 1000, 2),
    )
    return response


@app.route("/health", methods=["GET"])
def health_check():
    """Health check endpoint."""
    return jsonify({
        "status": "healthy",
        "timestamp": datetime.utcnow().isoformat(),
        "backend": settings.llm_backend.value,
    })


@app.route("/config", methods=["GET"])
def get_config():
    """Get current configuration (non-sensitive)."""
    return jsonify({
        "backend": settings.llm_backend.value,
        "max_tokens": settings.max_tokens,
        "temperature": settings.temperature,
        "context_length": settings.context_length,
        "openemr_enabled": bool(settings.openemr_client_id),
    })


@app.route("/generate", methods=["POST"])
@limiter.limit(settings.rate_limit_generate)
def generate():
    """
    Generate a response from the LLM.

    Request JSON:
    {
        "prompt": "User's question or prompt",
        "patient_id": "optional OpenEMR patient ID",
        "include_patient_data": false,
        "max_tokens": 512,
        "temperature": 0.7,
        "system_prompt": "optional custom system prompt"
    }

    Response JSON:
    {
        "response": "LLM generated response",
        "request_id": "uuid",
        "model": "model name",
        "tokens_used": 150,
        "duration_ms": 1234
    }
    """
    start_time = time.time()

    try:
        # Parse and validate request
        data = request.get_json()
        if not data:
            return jsonify({"error": "No JSON data provided"}), 400

        prompt = data.get("prompt", "").strip()
        if not prompt:
            return jsonify({"error": "Prompt is required"}), 400

        # Sanitize input
        prompt = sanitize_input(prompt)
        if len(prompt) > 10000:
            return jsonify({"error": "Prompt exceeds maximum length"}), 400

        # Get optional parameters
        patient_id = data.get("patient_id")
        include_patient_data = data.get("include_patient_data", False)
        max_tokens = min(data.get("max_tokens", settings.max_tokens), settings.max_tokens)
        temperature = data.get("temperature", settings.temperature)
        system_prompt = data.get("system_prompt", settings.system_prompt)

        # Build context with patient data if requested
        context = ""
        if patient_id and include_patient_data:
            emr_client = get_openemr()
            if emr_client:
                try:
                    patient_data = emr_client.get_patient_summary(patient_id)
                    if settings.anonymize_patient_data:
                        patient_data = anonymize_phi(patient_data)
                    context = f"\n\nPatient Context:\n{patient_data}\n\n"
                    logger.info(
                        "patient_data_retrieved",
                        request_id=g.request_id,
                        patient_id=patient_id,
                        anonymized=settings.anonymize_patient_data,
                    )
                except Exception as e:
                    logger.warning(
                        "patient_data_retrieval_failed",
                        request_id=g.request_id,
                        error=str(e),
                    )

        # Build full prompt
        full_prompt = f"{system_prompt}{context}User: {prompt}\n\nAssistant:"

        # Generate response
        llm = get_llm()
        response_text, metadata = llm.generate(
            prompt=full_prompt,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=settings.top_p,
            top_k=settings.top_k,
            repeat_penalty=settings.repeat_penalty,
        )

        duration_ms = round((time.time() - start_time) * 1000, 2)

        # Audit log
        if settings.audit_log_enabled:
            logger.info(
                "generation_completed",
                request_id=g.request_id,
                prompt_length=len(prompt),
                response_length=len(response_text),
                patient_id=patient_id if patient_id else None,
                duration_ms=duration_ms,
                model=metadata.get("model", "unknown"),
            )

        return jsonify({
            "response": response_text,
            "request_id": g.request_id,
            "model": metadata.get("model", "unknown"),
            "tokens_used": metadata.get("tokens_used", 0),
            "duration_ms": duration_ms,
        })

    except Exception as e:
        logger.error(
            "generation_failed",
            request_id=g.request_id,
            error=str(e),
            exc_info=True,
        )
        return jsonify({
            "error": "Failed to generate response",
            "request_id": g.request_id,
            "details": str(e) if settings.flask_debug else None,
        }), 500


@app.route("/models", methods=["GET"])
def list_models():
    """List available models for the current backend."""
    try:
        llm = get_llm()
        models = llm.list_models()
        return jsonify({
            "backend": settings.llm_backend.value,
            "models": models,
        })
    except Exception as e:
        logger.error("list_models_failed", error=str(e))
        return jsonify({
            "error": "Failed to list models",
            "details": str(e) if settings.flask_debug else None,
        }), 500


@app.route("/feedback", methods=["POST"])
@limiter.limit("10/minute")
def submit_feedback():
    """
    Submit feedback on a response.

    Request JSON:
    {
        "request_id": "uuid of original request",
        "rating": 1-5,
        "feedback_text": "optional text feedback",
        "helpful": true/false
    }
    """
    try:
        data = request.get_json()
        if not data:
            return jsonify({"error": "No JSON data provided"}), 400

        request_id = data.get("request_id")
        rating = data.get("rating")
        feedback_text = sanitize_input(data.get("feedback_text", ""))
        helpful = data.get("helpful")

        # Log feedback for analysis
        logger.info(
            "feedback_submitted",
            original_request_id=request_id,
            rating=rating,
            helpful=helpful,
            feedback_length=len(feedback_text) if feedback_text else 0,
        )

        return jsonify({
            "status": "received",
            "message": "Thank you for your feedback",
        })

    except Exception as e:
        logger.error("feedback_submission_failed", error=str(e))
        return jsonify({"error": "Failed to submit feedback"}), 500


@app.route("/patient/<patient_id>/summary", methods=["GET"])
@limiter.limit("20/minute")
def get_patient_summary(patient_id: str):
    """
    Get a summary of patient data from OpenEMR.

    Requires OpenEMR integration to be configured.
    """
    try:
        emr_client = get_openemr()
        if not emr_client:
            return jsonify({
                "error": "OpenEMR integration not configured"
            }), 503

        patient_data = emr_client.get_patient_summary(patient_id)

        if settings.anonymize_patient_data:
            patient_data = anonymize_phi(patient_data)

        logger.info(
            "patient_summary_retrieved",
            request_id=g.request_id,
            patient_id=patient_id,
        )

        return jsonify({
            "patient_id": patient_id,
            "summary": patient_data,
        })

    except Exception as e:
        logger.error(
            "patient_summary_failed",
            patient_id=patient_id,
            error=str(e),
        )
        return jsonify({
            "error": "Failed to retrieve patient summary",
            "details": str(e) if settings.flask_debug else None,
        }), 500


# Legacy endpoint for backward compatibility
@app.route("/api/generate", methods=["POST"])
@limiter.limit(settings.rate_limit_generate)
def generate_legacy():
    """Legacy endpoint - redirects to /generate."""
    return generate()


def create_app():
    """Application factory for testing and WSGI servers."""
    return app


if __name__ == "__main__":
    # Ensure logs directory exists
    Path("logs").mkdir(exist_ok=True)

    logger.info(
        "server_starting",
        host=settings.flask_host,
        port=settings.flask_port,
        backend=settings.llm_backend.value,
        debug=settings.flask_debug,
    )

    app.run(
        host=settings.flask_host,
        port=settings.flask_port,
        debug=settings.flask_debug,
    )
