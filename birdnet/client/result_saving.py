import json
import os


def save_result(data, fpath):
    """Saves the server response.

    Args:
        data: The response data.
        fpath: The path to save the data at.
    """
    # Make directory
    dir_path = os.path.dirname(fpath)
    os.makedirs(dir_path, exist_ok=True)

    # Save result
    with open(fpath, "w") as f:
        json.dump(data, f, indent=4)
