"""Logging bootstrap utilities."""

import logging
from logging.handlers import RotatingFileHandler


def setup_logging(log_file: str) -> None:
    """Configure console + rotating file logging for the whole application."""
    formatter = logging.Formatter(
        "%(asctime)s | %(levelname)s | %(name)s | %(message)s"
    )
    root = logging.getLogger()
    root.setLevel(logging.INFO)

    if not root.handlers:
        stream_handler = logging.StreamHandler()
        stream_handler.setFormatter(formatter)
        root.addHandler(stream_handler)

        file_handler = RotatingFileHandler(log_file, maxBytes=5_000_000, backupCount=3)
        file_handler.setFormatter(formatter)
        root.addHandler(file_handler)
