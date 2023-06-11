import os

from birdnet.configuration import config


def clear_error_log():
    """Clears the error log file.
    For debugging purposes.
    """
    if os.path.isfile(config.ERROR_LOG_FILE):
        os.remove(config.ERROR_LOG_FILE)
