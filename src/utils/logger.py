"""Configurable structured logger."""
import logging
import sys

def setup_logger(name: str = "CBIS-DDSM", level: str = "INFO") -> logging.Logger:
    """Setup stdout logging format."""
    logger = logging.getLogger(name)
    logger.setLevel(getattr(logging, level.upper(), logging.INFO))
    if not logger.handlers:
        ch = logging.StreamHandler(sys.stdout)
        ch.setFormatter(logging.Formatter("[%(asctime)s] [%(levelname)s] [%(name)s]: %(message)s", datefmt="%Y-%m-%d %H:%M:%S"))
        logger.addHandler(ch)
    return logger
