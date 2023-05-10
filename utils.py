"""Module containing common function.
"""
import os
import traceback

import config as cfg


def readLines(path: str):
    """Reads the lines into a list.

    Opens the file and reads its contents into a list.
    It is expected to have one line for each species or label.

    Args:
        path: Absolute path to the species file.

    Returns:
        A list of all species inside the file.
    """
    slist: list[str] = []

    if path:
        with open(path, "r", encoding="utf-8") as sfile:
            for line in sfile.readlines():
                slist.append(line.replace("\r", "").replace("\n", ""))

    return slist


def list_subdirectories(path: str):
    """Lists all directories inside a path.

    Retrieves all the subdirectories in a given path without recursion.

    Args:
        path: Directory to be searched.

    Returns:
        A filter sequence containing the absolute paths to all directories.
    """
    return filter(lambda el: os.path.isdir(os.path.join(path, el)), os.listdir(path))


def clearErrorLog():
    """Clears the error log file.

    For debugging purposes.
    """
    if os.path.isfile(cfg.ERROR_LOG_FILE):
        os.remove(cfg.ERROR_LOG_FILE)


def writeErrorLog(ex: Exception):
    """Writes an exception to the error log.

    Formats the stacktrace and writes it in the error log file configured in the config.

    Args:
        ex: An exception that occured.
    """
    with open(cfg.ERROR_LOG_FILE, "a") as elog:
        elog.write("".join(traceback.TracebackException.from_exception(ex).format()) + "\n")
