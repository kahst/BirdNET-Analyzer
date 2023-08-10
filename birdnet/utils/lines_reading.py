from pathlib import Path

from birdnet._paths import ROOT_PATH


def read_lines(path: str):
    """Reads the lines into a list.
    Opens the file and reads its contents into a list.
    It is expected to have one line for each species or label.
    Args:
        path: Absolute path to the species file.
    Returns:
        A list of all species inside the file.
    """
    lines = \
        Path(ROOT_PATH / path).read_text(encoding="utf-8").splitlines() \
        if path else []
    return lines
