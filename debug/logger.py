import logging

def get_logger(log_file_path="training.log"):
    r"""
    Return a logger object to record information in the console and log file.

    :param log_file_path (str, optional): The path of the log file, where logs will be saved..

    Example Usage:
    ```
    logger = get_logger()
    logger.debug("This is a debug message")
    logger.info("This is an info message")
    logger.error("This is an error message")
    logger.warning("This is a warning message")
    logger.critical("This is a critical message")
    ```
    """
    logger = logging.getLogger("logger001")
    logger.setLevel(logging.INFO)

    # Create and set a FileHandler (output to file)
    file_handler = logging.FileHandler(log_file_path)
    file_handler.setLevel(logging.INFO)

    # Create and set a StreamHandler (output to console)
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)

    # Create Formatter
    formatter = logging.Formatter(
        "%(asctime)s | %(levelname)s | %(message)s", # log format
        datefmt="%Y-%m-%d %H:%M:%S" # time format
    )

    # bind the formatter to the handlers
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)

    # Add handlers to the logger
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    
    return logger

