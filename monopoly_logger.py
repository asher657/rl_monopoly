import logging


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
        handler.setFormatter(logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s"))
        logger.addHandler(handler)

    if logging_level == 'info':
        logger.setLevel(logging.INFO)
    elif logging_level == 'debug':
        logger.setLevel(logging.DEBUG)
    else:
        logger.setLevel(logging.CRITICAL + 1)

    return logger
