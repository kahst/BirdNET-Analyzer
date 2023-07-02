import os
from collections import namedtuple

from birdnet.configuration import config
from birdnet.species.species_list_getting import get_species_list
from birdnet.utils.lines_reading import read_lines
from tests._paths import ROOT_PATH


def test_species():
    Arguments = namedtuple(
        'Arguments', "o lat lon week threshold sortby"
    )
    arguments: Arguments = Arguments(
        o=str(ROOT_PATH / 'example') + os.sep,
        lat=0.0,
        lon=0.0,
        week=-1,
        threshold=0.05,
        sortby='freq',
    )

    # Set paths relative to script path (requested in #3)
    config.LABELS_FILE = str(ROOT_PATH / config.LABELS_FILE)
    config.MDATA_MODEL_PATH = str(ROOT_PATH / config.MDATA_MODEL_PATH)

    # Load eBird codes, labels
    config.LABELS = read_lines(config.LABELS_FILE)

    # Set output path
    config.OUTPUT_PATH = arguments.o

    if (ROOT_PATH / config.OUTPUT_PATH).is_dir():
        config.OUTPUT_PATH = \
            str(
                ROOT_PATH / config.OUTPUT_PATH / "species_list.txt"
            )

    # Set config
    config.LATITUDE, config.LONGITUDE, config.WEEK = arguments.lat, arguments.lon, arguments.week
    config.LOCATION_FILTER_THRESHOLD = arguments.threshold

    print(f"Getting species list for {config.LATITUDE}/{config.LONGITUDE}, Week {config.WEEK}...", end="", flush=True)

    # Get species list
    species_list = get_species_list(
        config.LATITUDE,
        config.LONGITUDE,
        config.WEEK,
        config.LOCATION_FILTER_THRESHOLD,
        arguments.sortby != "freq",
    )

    print(f"Done. {len(species_list)} species on list.", flush=True)

    # Save species list
    with open(config.OUTPUT_PATH, "w") as f:
        for s in species_list:
            f.write(s + "\n")

    # A few examples to test
    # python3 species.py --o example/ --lat 42.5 --lon -76.45 --week -1
    # python3 species.py --o example/species_list.txt --lat 42.5 --lon -76.45 --week 4 --threshold 0.05 --sortby alpha
