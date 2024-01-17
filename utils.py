import logging


def get_logger(module_name):
    log_format = "[%(asctime)s] [%(levelname)s]: %(message)s"
    logging.basicConfig(
        format=log_format, level=logging.INFO, datefmt="%Y-%m-%d %H:%M:%S"
    )
    return logging.getLogger(module_name)
