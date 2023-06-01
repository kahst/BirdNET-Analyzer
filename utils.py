"""Module containing common function.
"""
import os
import traceback
from pathlib import Path

import config as cfg


def collect_audio_files(path: str):
    """Collects all audio files in the given directory.

    Args:
        path: The directory to be searched.

    Returns:
        A sorted list of all audio files in the directory.
    """
    # Get all files in directory with os.walk
    files = []

    for root, _, flist in os.walk(path):
        for f in flist:
            if not f.startswith(".") and f.rsplit(".", 1)[-1].lower() in cfg.ALLOWED_FILETYPES:
                files.append(os.path.join(root, f))

    return sorted(files)


def readLines(path: str):
    """Reads the lines into a list.

    Opens the file and reads its contents into a list.
    It is expected to have one line for each species or label.

    Args:
        path: Absolute path to the species file.

    Returns:
        A list of all species inside the file.
    """
    return Path(path).read_text(encoding="utf-8").splitlines() if path else []


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
        ex: An exception that occurred.
    """
    with open(cfg.ERROR_LOG_FILE, "a") as elog:
        elog.write("".join(traceback.TracebackException.from_exception(ex).format()) + "\n")
