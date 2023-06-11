from birdnet.configuration import config


import json


def load_codes():
    """Loads the eBird codes.
    Returns:
        A dictionary containing the eBird codes.
    """
    with open(config.CODES_FILE, "r") as cfile:
        codes = json.load(cfile)

    return codes
