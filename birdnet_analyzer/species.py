"""Module for predicting a species list.

Can be used to predict a species list using coordinates and weeks.
"""

import argparse
import os

import birdnet_analyzer.config as cfg
import birdnet_analyzer.model as model
import birdnet_analyzer.utils as utils

SCRIPT_DIR = os.path.abspath(os.path.dirname(__file__))


def getSpeciesList(lat: float, lon: float, week: int, threshold=0.05, sort=False) -> list[str]:
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


def run(output_path, lat, lon, week, threshold, sortby):
    """
    Generates a species list for a given location and time, and saves it to the specified output path.
    Args:
        output_path (str): The path where the species list will be saved. If it's a directory, the list will be saved as "species_list.txt" inside it.
        lat (float): Latitude of the location.
        lon (float): Longitude of the location.
        week (int): Week of the year (1-52) for which the species list is generated.
        threshold (float): Threshold for location filtering.
        sortby (str): Sorting criteria for the species list. Can be "freq" for frequency or any other value for alphabetical sorting.
    Returns:
        None
    """
    # Set paths relative to script path (requested in #3)
    cfg.LABELS_FILE = os.path.join(SCRIPT_DIR, cfg.LABELS_FILE)
    cfg.MDATA_MODEL_PATH = os.path.join(SCRIPT_DIR, cfg.MDATA_MODEL_PATH)

    # Load eBird codes, labels
    cfg.LABELS = utils.readLines(cfg.LABELS_FILE)

    # Set output path
    cfg.OUTPUT_PATH = output_path

    if os.path.isdir(cfg.OUTPUT_PATH):
        cfg.OUTPUT_PATH = os.path.join(cfg.OUTPUT_PATH, "species_list.txt")

    # Set config
    cfg.LATITUDE, cfg.LONGITUDE, cfg.WEEK = lat, lon, week
    cfg.LOCATION_FILTER_THRESHOLD = threshold

    print(f"Getting species list for {cfg.LATITUDE}/{cfg.LONGITUDE}, Week {cfg.WEEK}...", end="", flush=True)

    # Get species list
    species_list = getSpeciesList(
        cfg.LATITUDE, cfg.LONGITUDE, cfg.WEEK, cfg.LOCATION_FILTER_THRESHOLD, False if sortby == "freq" else True
    )

    print(f"Done. {len(species_list)} species on list.", flush=True)

    # Save species list
    with open(cfg.OUTPUT_PATH, "w") as f:
        for s in species_list:
            f.write(s + "\n")


if __name__ == "__main__":
    # Parse arguments
    parser = argparse.ArgumentParser(
        description="Get list of species for a given location with BirdNET. Sorted by occurrence frequency.",
        parents=[utils.species_args()],
    )
    parser.add_argument(
        "output",
        metavar="OUTPUT",
        nargs="?",
        default=cfg.OUTPUT_PATH,
        help="Path to output file or folder. If this is a folder, file will be named 'species_list.txt'.",
    )

    parser.add_argument(
        "--sortby",
        default="freq",
        choices=["freq", "alpha"],
        help="Sort species by occurrence frequency or alphabetically. Values in ['freq', 'alpha']. Defaults to 'freq'.",
    )

    args = parser.parse_args()

    run(args.output, args.lat, args.lon, args.week, args.sf_thresh, args.sortby)
