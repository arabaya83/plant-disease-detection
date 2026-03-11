"""Central logging setup for the FastAPI application.

This module configures the root logger used by the API and service layers.
The project writes logs both to stdout and to a rotating file so local
development, demos, and report generation can all inspect runtime activity.
"""

import logging
from logging.handlers import RotatingFileHandler


def setup_logging(log_file: str) -> None:
    """Configure application-wide console and file logging.

    Args:
        log_file: Path to the rotating log file used for persistent application
            logs.

    Side Effects:
        Mutates the root logger by attaching stream and rotating file handlers
        when handlers have not already been configured.
    """
    formatter = logging.Formatter(
        "%(asctime)s | %(levelname)s | %(name)s | %(message)s"
    )
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.INFO)

    if not root_logger.handlers:
        stream_handler = logging.StreamHandler()
        stream_handler.setFormatter(formatter)
        root_logger.addHandler(stream_handler)

        file_handler = RotatingFileHandler(log_file, maxBytes=5_000_000, backupCount=3)
        file_handler.setFormatter(formatter)
        root_logger.addHandler(file_handler)
