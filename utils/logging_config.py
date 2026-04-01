"""
Logging Configuration

Sets up structured logging with JSON output for production
and human-readable output for development.
"""

import sys
import logging
from pathlib import Path

import structlog

from config import settings


def setup_logging() -> None:
    """Configure structured logging."""

    # Ensure logs directory exists
    log_dir = Path(settings.log_file).parent
    log_dir.mkdir(parents=True, exist_ok=True)

    # Determine log level
    log_level = getattr(logging, settings.log_level.upper(), logging.INFO)

    # Configure standard logging
    logging.basicConfig(
        format="%(message)s",
        level=log_level,
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(settings.log_file),
        ],
    )

    # Configure structlog
    if settings.log_format == "json":
        # JSON format for production
        processors = [
            structlog.contextvars.merge_contextvars,
            structlog.processors.add_log_level,
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.processors.StackInfoRenderer(),
            structlog.processors.format_exc_info,
            structlog.processors.JSONRenderer(),
        ]
    else:
        # Human-readable format for development
        processors = [
            structlog.contextvars.merge_contextvars,
            structlog.processors.add_log_level,
            structlog.processors.TimeStamper(fmt="%Y-%m-%d %H:%M:%S"),
            structlog.processors.StackInfoRenderer(),
            structlog.dev.ConsoleRenderer(colors=True),
        ]

    structlog.configure(
        processors=processors,
        wrapper_class=structlog.make_filtering_bound_logger(log_level),
        context_class=dict,
        logger_factory=structlog.PrintLoggerFactory(),
        cache_logger_on_first_use=True,
    )


def get_audit_logger():
    """Get a dedicated audit logger."""
    audit_logger = logging.getLogger("audit")

    if not audit_logger.handlers:
        audit_dir = Path(settings.audit_log_file).parent
        audit_dir.mkdir(parents=True, exist_ok=True)

        handler = logging.FileHandler(settings.audit_log_file)
        handler.setFormatter(
            logging.Formatter(
                '{"timestamp": "%(asctime)s", "level": "%(levelname)s", '
                '"event": "%(message)s"}'
            )
        )
        audit_logger.addHandler(handler)
        audit_logger.setLevel(
            getattr(logging, settings.audit_log_level.upper(), logging.INFO)
        )

    return audit_logger
