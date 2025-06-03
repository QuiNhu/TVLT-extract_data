import logging
import sys

def get_logger(name: str) -> logging.Logger:
    logger = logging.getLogger(name)
    if not logger.handlers:
        logger.setLevel(logging.DEBUG)
        stream_handler = logging.StreamHandler(sys.stderr)
        log_formatter = logging.Formatter("%(asctime)s - [%(levelname)s] - %(name)s: %(message)s")
        stream_handler.setFormatter(log_formatter)
        logger.addHandler(stream_handler)
    return logger
