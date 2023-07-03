import traceback

from birdnet.configuration import config


def write_error_log(ex: Exception):
    """Writes an exception to the error log.
    Formats the stacktrace and writes it in the error log file configured in
    the config.

    Args:
        ex: An exception that occurred.
    """
    with open(config.ERROR_LOG_FILE, "a") as error_log:
        error_log.write(
            "".join(
                traceback.TracebackException.from_exception(ex).format()
            )
            +
            "\n"
        )
