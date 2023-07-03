import os

from birdnet.configuration import config


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
            if not f.startswith(".") and f.rsplit(".", 1)[-1].lower() in \
                    config.ALLOWED_FILETYPES:
                files.append(os.path.join(root, f))

    return sorted(files)
