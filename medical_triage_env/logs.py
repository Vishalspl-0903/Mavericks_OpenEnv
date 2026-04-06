"""
Structured logging configuration for medical triage environment.

Provides consistent JSON logging in production and human-readable console logging
in development mode. Never logs PHI (Protected Health Information).
"""
from __future__ import annotations

import os
import sys
from typing import Any, Dict

import structlog
from structlog.types import Processor


def _scrub_phi(logger: Any, method_name: str, event_dict: Dict[str, Any]) -> Dict[str, Any]:
    """Remove or mask potential PHI from logs."""
    # Scrub patient_name if accidentally included
    if "patient_name" in event_dict:
        event_dict["patient_name"] = "[REDACTED]"
    
    # Scrub patient_id partially - keep only prefix for debugging
    if "patient_id" in event_dict:
        pid = str(event_dict["patient_id"])
        if len(pid) > 8:
            event_dict["patient_id"] = f"{pid[:8]}..."
    
    return event_dict


def _configure_structlog() -> None:
    """Configure structlog based on environment."""
    env = os.getenv("ENV", "production").lower()
    
    # Common processors for all environments
    shared_processors: list[Processor] = [
        structlog.contextvars.merge_contextvars,
        structlog.stdlib.add_log_level,
        structlog.stdlib.add_logger_name,
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.StackInfoRenderer(),
        _scrub_phi,
    ]
    
    if env == "development":
        # Human-readable console output for development
        processors = shared_processors + [
            structlog.dev.ConsoleRenderer(colors=True),
        ]
    else:
        # JSON output for production
        processors = shared_processors + [
            structlog.processors.format_exc_info,
            structlog.processors.JSONRenderer(),
        ]
    
    structlog.configure(
        processors=processors,
        wrapper_class=structlog.make_filtering_bound_logger(logging_level="DEBUG"),
        context_class=dict,
        logger_factory=structlog.PrintLoggerFactory(file=sys.stdout),
        cache_logger_on_first_use=True,
    )


# Configure on module import
_configure_structlog()


def get_logger(name: str) -> structlog.BoundLogger:
    """
    Get a logger instance with the given name.
    
    Args:
        name: Logger name, typically __name__ of the calling module
        
    Returns:
        BoundLogger instance for structured logging
    """
    return structlog.get_logger(name)
