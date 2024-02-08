import logging
import os


def get_logger(module_name: str) -> logging.Logger:
    logger = logging.getLogger(module_name)
    handler = logging.StreamHandler()
    formatter = logging.Formatter('[%(levelname)s] %(asctime)s [%(name)s]: %(message)s', datefmt='%H:%M:%S')
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    # You can change the log level globally using "OUR_LOG_LEVEL" env variable
    logger.setLevel(os.environ.get('OUR_LOG_LEVEL', 'INFO'))
    logger.propagate = False
    return logger
