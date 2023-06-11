from pathlib import Path


def read_lines(path: str):
    """Reads the lines into a list.
    Opens the file and reads its contents into a list.
    It is expected to have one line for each species or label.
    Args:
        path: Absolute path to the species file.
    Returns:
        A list of all species inside the file.
    """
    return Path(path).read_text(encoding="utf-8").splitlines() if path else []
