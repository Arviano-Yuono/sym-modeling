import logging
import os
import sys
from datetime import datetime
from pathlib import Path

# Optional: Try to import colorlog for pretty terminal output
# You can install it via: pip install colorlog
try:
    import colorlog

    HAS_COLOR = True
except ImportError:
    HAS_COLOR = False


def setup_logger(
    name: str = "SR_Turbulence", log_dir: str = "output/logs", level=logging.INFO
):
    """
    Configures a singleton logger that writes to console and a file.

    Args:
        name: The name of the logger (keep this consistent across modules).
        log_dir: Directory where log files will be saved.
        level: Logging threshold (DEBUG, INFO, WARNING, ERROR).

    Returns:
        logging.Logger: The configured logger instance.
    """

    # 1. Get the Singleton Instance
    # Python's getLogger always returns the same object for the same name.
    logger = logging.getLogger(name)
    logger.setLevel(level)

    # 2. Prevent Duplicate Handlers
    # If we already configured this logger, don't add handlers again.
    if logger.hasHandlers():
        return logger

    # 3. Create Output Directory
    Path(log_dir).mkdir(parents=True, exist_ok=True)

    # 4. Create Formatters
    # Timestamp | Level | Filename:Line | Message
    file_format = logging.Formatter(
        "[%(asctime)s] [%(levelname)s] [%(filename)s:%(lineno)d] - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    if HAS_COLOR:
        console_format = colorlog.ColoredFormatter(
            "%(log_color)s[%(asctime)s] %(levelname)-8s%(reset)s %(blue)s%(message)s",
            datefmt="%H:%M:%S",
            log_colors={
                "DEBUG": "cyan",
                "INFO": "green",
                "WARNING": "yellow",
                "ERROR": "red",
                "CRITICAL": "red,bg_white",
            },
        )
    else:
        console_format = logging.Formatter("[%(asctime)s] %(levelname)-8s %(message)s")

    # 5. Handler: File (One file per run, timestamped)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(log_dir, f"run_{timestamp}.log")

    file_handler = logging.FileHandler(log_file)
    file_handler.setFormatter(file_format)
    file_handler.setLevel(logging.DEBUG)  # Always capture EVERYTHING to file

    # 6. Handler: Console (Standard Output)
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(console_format)
    console_handler.setLevel(level)  # Respect the user's requested level

    # 7. Add Handlers
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    # Log the initialization event
    logger.info(f"Logger initialized. Saving logs to: {log_file}")

    return logger


def get_logger(name: str = "SR_Turbulence"):
    """
    Helper to just retrieve the logger without configuring it again.
    Useful for sub-modules.
    """
    return logging.getLogger(name)
