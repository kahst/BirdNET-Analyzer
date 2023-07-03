import os


def list_subdirectories(path: str):
    """Lists all directories inside a path.
    Retrieves all the subdirectories in a given path without recursion.
    Args:
        path: Directory to be searched.
    Returns:
        A filter sequence containing the absolute paths to all directories.
    """
    return filter(
        lambda el:
        os.path.isdir(os.path.join(path, el)),
        os.listdir(path)
    )
