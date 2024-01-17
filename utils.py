import logging


def get_logger(module_name: str, log_level: str = logging.INFO) -> logging.Logger:
    logger = logging.getLogger(module_name)
    handler = logging.StreamHandler()
    formatter = logging.Formatter(
        "[%(asctime)s] [%(levelname)s] [%(name)s]: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.setLevel(log_level)
    logger.propagate = False
    return logger
