import logging
import os
from typing import Optional

# Define log file path relative to the project root
LOG_DIR = "./logs"
LOG_FILE_PATH = os.path.join(LOG_DIR, "app.log")

# Create the logs directory if it doesn't exist
os.makedirs(LOG_DIR, exist_ok=True)

def setup_logging():
    """
    Configures the root logger for the application.
    This should be called once at application startup (e.g., in main.py).
    """
    # Create logger
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    # Clear existing handlers to prevent duplicate logs
    if logger.hasHandlers():
        logger.handlers.clear()

    # Create formatters
    formatter = logging.Formatter(
        '[%(asctime)s] [%(levelname)s] [%(name)s] [%(funcName)s] - %(message)s'
    )
    
    # Create handlers
    # Stream Handler for console output
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    stream_handler.setLevel(logging.INFO)

    # File Handler for writing logs to a file
    file_handler = logging.FileHandler(LOG_FILE_PATH)
    file_handler.setFormatter(formatter)
    file_handler.setLevel(logging.INFO)

    # Add handlers to the logger
    logger.addHandler(stream_handler)
    logger.addHandler(file_handler)

def get_logger(name: Optional[str] = None):
    """
    Returns a logger instance for a given module.
    """
    if name is None:
        name = "ml_backend"
    return logging.getLogger(name)

# Ensure logging is set up when this module is imported
setup_logging()