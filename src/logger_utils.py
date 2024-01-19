import logging
import os


def get_logger(module_name: str) -> logging.Logger:
    logger = logging.getLogger(module_name)
    handler = logging.StreamHandler()
    formatter = logging.Formatter(
        "[%(asctime)s] [%(levelname)s] [%(name)s]: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    # You can change the log level globally using "LOGLEVEL" env variable
    logger.setLevel(os.environ.get("LOGLEVEL", "INFO"))
    logger.propagate = False
    return logger
