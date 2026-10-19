"""Helper functions for initializing and configuring logging for the iMOD Suite."""

import logging
import sys
from pathlib import Path

import imod
from imod.logging.config import LoggerType
from imod.logging.loglevel import LogLevel
from imod.logging.pythonlogger import _formatter

# Loggers in python are singletons, so this is the same instance as the one used in imod.logging.pythonlogger.PythonLogger.
imod_logger = logging.getLogger("imod")


def init_logging() -> None:
    """Initialize logging with stream handler."""
    imod.logging.configure(
        LoggerType.PYTHON,
        log_level=LogLevel.DEBUG,
        add_default_file_handler=False,
        add_default_stream_handler=False,
    )
    # Add streamhandler
    streamhandler = logging.StreamHandler(sys.stdout)
    streamhandler.setLevel(logging.INFO)
    streamhandler.setFormatter(_formatter())
    imod_logger.addHandler(streamhandler)
    imod_logger.info("Logging initialized.")


def add_filehandler(logfile_path: Path, log_level: int = logging.INFO):
    """Add a file handler with the specified log level to the logger."""
    filehandler = logging.FileHandler(logfile_path, mode="w")
    filehandler.setLevel(log_level)
    filehandler.setFormatter(_formatter())
    imod_logger.addHandler(filehandler)
    imod_logger.info(f"More verbose logging output will be written to {logfile_path}")
