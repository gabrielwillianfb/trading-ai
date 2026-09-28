"""Structured logging configuration with sensitive data sanitization.

Ensures that API keys, secrets, and sensitive tokens are never written to logs.
"""

import logging
import re
import sys
from typing import Optional


class SensitiveDataFilter(logging.Filter):
    """Filter that masks API keys, secrets, and sensitive tokens from log records."""

    # Patterns matching typical api key/secret patterns or json key-value pairs
    PATTERNS = [
        (re.compile(r'(?i)(api[_-]?key["\']?\s*[:=]\s*["\']?)([^"\'\s&]+)(["\']?)'), r"\1***REDACTED***\3"),
        (re.compile(r'(?i)(secret[_-]?key["\']?\s*[:=]\s*["\']?)([^"\'\s&]+)(["\']?)'), r"\1***REDACTED***\3"),
        (re.compile(r'(?i)(api[_-]?secret["\']?\s*[:=]\s*["\']?)([^"\'\s&]+)(["\']?)'), r"\1***REDACTED***\3"),
        (re.compile(r'(?i)(password["\']?\s*[:=]\s*["\']?)([^"\'\s&]+)(["\']?)'), r"\1***REDACTED***\3"),
        (re.compile(r'(?i)(token["\']?\s*[:=]\s*["\']?)([^"\'\s&]+)(["\']?)'), r"\1***REDACTED***\3"),
    ]

    def filter(self, record: logging.LogRecord) -> bool:
        if isinstance(record.msg, str):
            record.msg = self.sanitize(record.msg)
        if record.args:
            if isinstance(record.args, dict):
                record.args = {k: self.sanitize(v) if isinstance(v, str) else v for k, v in record.args.items()}
            elif isinstance(record.args, tuple):
                record.args = tuple(self.sanitize(v) if isinstance(v, str) else v for v in record.args)
        return True

    @classmethod
    def sanitize(cls, text: str) -> str:
        """Replace sensitive patterns with redacted placeholder."""
        result = text
        for pattern, replacement in cls.PATTERNS:
            result = pattern.sub(replacement, result)
        return result


def setup_logging(log_level: Optional[str] = None) -> logging.Logger:
    """Configure structured logging for the application.

    Args:
        log_level: Log level string (DEBUG, INFO, WARNING, ERROR, CRITICAL).
                   If None, defaults to INFO.

    Returns:
        Root logger instance.
    """
    level_name = (log_level or "INFO").upper()
    level = getattr(logging, level_name, logging.INFO)

    root_logger = logging.getLogger()
    root_logger.setLevel(level)

    # Avoid adding duplicate handlers if setup_logging is called multiple times
    if not any(isinstance(h, logging.StreamHandler) for h in root_logger.handlers):
        handler = logging.StreamHandler(sys.stdout)
        handler.setLevel(level)

        formatter = logging.Formatter(
            fmt="%(asctime)s [%(levelname)s] [%(name)s:%(lineno)d] - %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        handler.setFormatter(formatter)
        handler.addFilter(SensitiveDataFilter())

        root_logger.addHandler(handler)
    else:
        for handler in root_logger.handlers:
            handler.setLevel(level)
            # Ensure sensitive data filter is attached
            if not any(isinstance(f, SensitiveDataFilter) for f in handler.filters):
                handler.addFilter(SensitiveDataFilter())

    # Lower noisy third-party loggers
    logging.getLogger("uvicorn.access").setLevel(logging.WARNING)
    logging.getLogger("asyncio").setLevel(logging.WARNING)

    return root_logger
