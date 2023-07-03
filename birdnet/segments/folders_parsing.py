from birdnet.configuration import config

import os


def parse_folders(
    apath: str,
    rpath: str,
    allowed_result_filetypes: list[str] = ["txt", "csv"]
) -> list[dict]:
    """Read audio and result files.
    Reads all audio files and BirdNET output inside directory recursively.
    Args:
        apath: Path to search for audio files.
        rpath: Path to search for result files.
        allowed_result_filetypes: List of extensions for the result files.
    Returns:
        A list of {"audio": path_to_audio, "result": path_to_result }.
    """
    data = {}
    apath = apath.replace("/", os.sep).replace("\\", os.sep)
    rpath = rpath.replace("/", os.sep).replace("\\", os.sep)

    # Get all audio files
    for root, _, files in os.walk(apath):
        for f in files:
            if f.rsplit(".", 1)[-1].lower() in config.ALLOWED_FILETYPES:
                data[f.rsplit(".", 1)[0]] = \
                    {"audio": os.path.join(root, f), "result": ""}

    # Get all result files
    for root, _, files in os.walk(rpath):
        for f in files:
            if f.rsplit(".", 1)[-1] in allowed_result_filetypes and \
                    ".BirdNET." in f:
                data[f.split(".BirdNET.", 1)[0]]["result"] = \
                    os.path.join(root, f)

    # Convert to list
    flist = [f for f in data.values() if f["result"]]

    print(f"Found {len(flist)} audio files with valid result file.")

    return flist
