from birdnet._paths import ROOT_PATH
from birdnet.configuration import config

import json


def load_codes():
    """Loads the eBird codes.
    Returns:
        A dictionary containing the eBird codes.
    """
    with open(ROOT_PATH / config.CODES_FILE, "r") as codes_file:
        codes = json.load(codes_file)

    return codes
