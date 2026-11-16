"""Quantamental AI - Structured Logging Setup.

Configures application-wide logging with standardized formats and handlers.
"""

import logging
import sys
from typing import Optional
from app.config import settings


def setup_logging(log_level: Optional[str] = None) -> None:
    """Configure root logging with standard format and stream handler."""
    level_name = (log_level or settings.LOG_LEVEL).upper()
    numeric_level = getattr(logging, level_name, logging.INFO)

    log_format = (
        "[%(asctime)s] [%(levelname)s] [%(name)s:%(lineno)d] - %(message)s"
    )
    date_format = "%Y-%m-%d %H:%M:%S"

    # Avoid duplicate handlers on re-configuration
    root_logger = logging.getLogger()
    root_logger.setLevel(numeric_level)

    # Clear existing handlers
    if root_logger.hasHandlers():
        root_logger.handlers.clear()

    handler = logging.StreamHandler(sys.stdout)
    handler.setLevel(numeric_level)
    formatter = logging.Formatter(fmt=log_format, datefmt=date_format)
    handler.setFormatter(formatter)
    root_logger.addHandler(handler)


def get_logger(name: str) -> logging.Logger:
    """Return a logger instance configured with the specified module name."""
    return logging.getLogger(name)
