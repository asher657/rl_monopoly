import logging
from datetime import datetime
import os


if not os.path.exists('log'):
    os.makedirs('log')

LOG_FILE = f'log/monopoly_{datetime.now().strftime("%Y-%m-%d_%H-%M-%S")}.log'

file_handler = logging.FileHandler(LOG_FILE)
file_handler.setLevel(logging.DEBUG)


def get_monopoly_logger(class_name, logging_level: str = 'info'):
    """
    Get a logger given the class name and logging level
    :param class_name: the class name associated with the logger
    :param logging_level: the logging level
    :return: the instantiated logger
    """
    assert logging_level.lower() in ['info', 'debug', 'none'], 'Logging should be one of "info", "debug", "none"'
    logger = logging.getLogger(class_name)
    logger.propagate = False

    if not logger.hasHandlers():  # Prevent adding multiple handlers
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(name)s.%(funcName)s - %(message)s"))
        logger.addHandler(handler)

    if not any(isinstance(h, logging.FileHandler) and h.baseFilename == file_handler.baseFilename
               for h in logger.handlers):
        logger.addHandler(file_handler)

    if logging_level == 'info':
        logger.setLevel(logging.INFO)
    elif logging_level == 'debug':
        logger.setLevel(logging.DEBUG)
    else:
        logger.setLevel(logging.CRITICAL + 1)

    return logger
