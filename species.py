"""Module for predicting a species list.

Can be used to predict a species list using coordinates and weeks.
"""
import argparse
import os
import sys

import config
import model
import utils


def get_species_list(lat: float, lon: float, week: int, threshold=0.05, sort=False) -> List[str]:
    """Predict a species list.

    Uses the model to predict the species list for the given coordinates and filters by threshold.

    Args:
        lat: The latitude.
        lon: The longitude.
        week: The week of the year [1-48]. Use -1 for year-round.
        threshold: Only values above or equal to threshold will be shown.
        sort: If the species list should be sorted.

    Returns:
        A list of all eligible species.
    """
    # Extract species from model
    pred = model.explore(lat, lon, week)

    # Make species list
    slist = [p[1] for p in pred if p[0] >= threshold]

    return sorted(slist) if sort else slist


if __name__ == "__main__":
    # Parse arguments
    parser = argparse.ArgumentParser(
        description="Get list of species for a given location with BirdNET. Sorted by occurrence frequency."
    )
    parser.add_argument(
        "--o",
        default="example/",
        help="Path to output file or folder. If this is a folder, file will be named 'species_list.txt'.",
    )
    parser.add_argument("--lat", type=float, help="Recording location latitude.")
    parser.add_argument("--lon", type=float, help="Recording location longitude.")
    parser.add_argument(
        "--week",
        type=int,
        default=-1,
        help="Week of the year when the recording was made. Values in [1, 48] (4 weeks per month). Set -1 for year-round species list.",
    )
    parser.add_argument("--threshold", type=float, default=0.05, help="Occurrence frequency threshold. Defaults to 0.05.")
    parser.add_argument(
        "--sortby",
        default="freq",
        help="Sort species by occurrence frequency or alphabetically. Values in ['freq', 'alpha']. Defaults to 'freq'.",
    )

    args = parser.parse_args()

    # Set paths relative to script path (requested in #3)
    config.LABELS_FILE = os.path.join(os.path.dirname(os.path.abspath(sys.argv[0])), config.LABELS_FILE)
    config.MDATA_MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(sys.argv[0])), config.MDATA_MODEL_PATH)

    # Load eBird codes, labels
    config.LABELS = utils.read_lines(config.LABELS_FILE)

    # Set output path
    config.OUTPUT_PATH = args.o

    if os.path.isdir(config.OUTPUT_PATH):
        config.OUTPUT_PATH = os.path.join(config.OUTPUT_PATH, "species_list.txt")

    # Set config
    config.LATITUDE, config.LONGITUDE, config.WEEK = args.lat, args.lon, args.week
    config.LOCATION_FILTER_THRESHOLD = args.threshold

    print(f"Getting species list for {config.LATITUDE}/{config.LONGITUDE}, Week {config.WEEK}...", end="", flush=True)

    # Get species list
    species_list = get_species_list(
        config.LATITUDE, config.LONGITUDE, config.WEEK, config.LOCATION_FILTER_THRESHOLD, False if args.sortby == "freq" else True
    )

    print(f"Done. {len(species_list)} species on list.", flush=True)

    # Save species list
    with open(config.OUTPUT_PATH, "w") as f:
        for s in species_list:
            f.write(s + "\n")

    # A few examples to test
    # python3 species.py --o example/ --lat 42.5 --lon -76.45 --week -1
    # python3 species.py --o example/species_list.txt --lat 42.5 --lon -76.45 --week 4 --threshold 0.05 --sortby alpha
