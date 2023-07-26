"""Module to analyze audio samples.
"""

import argparse
import os
import sys
from multiprocessing import Pool, freeze_support

from birdnet.analysis.codes_loading import load_codes
from birdnet.analysis.file_analysing import analyze_file
from birdnet.configuration import config
from birdnet.utils.audio_file_collecting import collect_audio_files
from birdnet.utils.lines_reading import read_lines
from birdnet.species.species_list_getting import get_species_list


if __name__ == "__main__":
    # Freeze support for executable
    freeze_support()

    # Parse arguments
    parser = argparse.ArgumentParser(
        description="Analyze audio files with BirdNET",
    )
    parser.add_argument(
        "--i",
        default="example/",
        help=
        "Path to input file or folder. "
        "If this is a file, --o needs to be a file too.",
    )
    parser.add_argument(
        "--o",
        default="example/",
        help=
        "Path to output file or folder. "
        "If this is a file, --i needs to be a file too.",
    )
    parser.add_argument(
        "--lat",
        type=float,
        default=-1,
        help="Recording location latitude. Set -1 to ignore.",
    )
    parser.add_argument(
        "--lon",
        type=float,
        default=-1,
        help="Recording location longitude. Set -1 to ignore.",
    )
    parser.add_argument(
        "--week",
        type=int,
        default=-1,
        help=
        "Week of the year when the recording was made. "
        "Values in [1, 48] (4 weeks per month). "
        "Set -1 for year-round species list.",
    )
    parser.add_argument(
        "--slist",
        default="",
        help=
        'Path to species list file or folder. '
        'If folder is provided, species list needs to be named '
        '"species_list.txt". '
        'If lat and lon are provided, this list will be ignored.',
    )
    parser.add_argument(
        "--sensitivity",
        type=float,
        default=1.0,
        help=
        "Detection sensitivity; Higher values result in higher sensitivity. "
        "Values in [0.5, 1.5]. Defaults to 1.0.",
    )
    parser.add_argument(
        "--min_conf",
        type=float,
        default=0.1,
        help=
        "Minimum confidence threshold. "
        "Values in [0.01, 0.99]. Defaults to 0.1.",
    )
    parser.add_argument(
        "--overlap",
        type=float,
        default=0.0,
        help=
        "Overlap of prediction segments. "
        "Values in [0.0, 2.9]. Defaults to 0.0.",
    )
    parser.add_argument(
        "--rtype",
        default="table",
        help=
        "Specifies output format. "
        "Values in ['table', 'audacity', 'r',  'kaleidoscope', 'csv']. "
        "Defaults to 'table' (Raven selection table).",
    )
    parser.add_argument(
        "--threads",
        type=int,
        default=4,
        help="Number of CPU threads.",
    )
    parser.add_argument(
        "--batchsize",
        type=int,
        default=1,
        help="Number of samples to process at the same time. Defaults to 1.",
    )
    parser.add_argument(
        "--locale",
        default="en",
        help=
        "Locale for translated species common names. "
        "Values in ['af', 'de', 'it', ...] Defaults to 'en'.",
    )
    parser.add_argument(
        "--sf_thresh",
        type=float,
        default=0.03,
        help=
        "Minimum species occurrence frequency threshold for location filter. "
        "Values in [0.01, 0.99]. Defaults to 0.03.",
    )
    parser.add_argument(
        "--classifier",
        default=None,
        help=
        "Path to custom trained classifier. "
        "Defaults to None. If set, --lat, --lon and --locale are ignored.",
    )

    args = parser.parse_args()

    # Set paths relative to script path (requested in #3)
    script_dir = os.path.dirname(os.path.abspath(sys.argv[0]))
    config.MODEL_PATH = os.path.join(script_dir, config.MODEL_PATH)
    config.LABELS_FILE = os.path.join(script_dir, config.LABELS_FILE)
    config.TRANSLATED_LABELS_PATH = \
        os.path.join(script_dir, config.TRANSLATED_LABELS_PATH)
    config.MDATA_MODEL_PATH = os.path.join(script_dir, config.MDATA_MODEL_PATH)
    config.CODES_FILE = os.path.join(script_dir, config.CODES_FILE)
    config.ERROR_LOG_FILE = os.path.join(script_dir, config.ERROR_LOG_FILE)

    # Load eBird codes, labels
    config.CODES = load_codes()
    config.LABELS = read_lines(config.LABELS_FILE)

    # Set custom classifier?
    if args.classifier is not None:
        # we treat this as absolute path, so no need to join with dirname
        config.CUSTOM_CLASSIFIER = args.classifier
        # same for labels file
        config.LABELS_FILE = args.classifier.replace(".tflite", "_Labels.txt")

        config.LABELS = read_lines(config.LABELS_FILE)
        args.lat = -1
        args.lon = -1
        args.locale = "en"

    # Load translated labels
    lfile = os.path.join(
        config.TRANSLATED_LABELS_PATH,
        os.path.basename(
            config.LABELS_FILE
        ).replace(
            ".txt", "_{}.txt".format(args.locale)
        )
    )

    if not args.locale in ["en"] and os.path.isfile(lfile):
        config.TRANSLATED_LABELS = read_lines(lfile)
    else:
        config.TRANSLATED_LABELS = config.LABELS

    ### Make sure to comment out appropriately if you are not using args. ###

    # Load species list from location filter or provided list
    config.LATITUDE, config.LONGITUDE, config.WEEK = \
        args.lat, args.lon, args.week
    config.LOCATION_FILTER_THRESHOLD = \
        max(0.01, min(0.99, float(args.sf_thresh)))

    if config.LATITUDE == -1 and config.LONGITUDE == -1:
        if not args.slist:
            config.SPECIES_LIST_FILE = None
        else:
            config.SPECIES_LIST_FILE = os.path.join(script_dir, args.slist)

            if os.path.isdir(config.SPECIES_LIST_FILE):
                config.SPECIES_LIST_FILE = \
                    os.path.join(config.SPECIES_LIST_FILE, "species_list.txt")

        config.SPECIES_LIST = read_lines(config.SPECIES_LIST_FILE)
    else:
        config.SPECIES_LIST_FILE = None
        config.SPECIES_LIST = get_species_list(
            config.LATITUDE,
            config.LONGITUDE,
            config.WEEK,
            config.LOCATION_FILTER_THRESHOLD,
        )

    if not config.SPECIES_LIST:
        print(f"Species list contains {len(config.LABELS)} species")
    else:
        print(f"Species list contains {len(config.SPECIES_LIST)} species")

    # Set input and output path
    config.INPUT_PATH = args.i
    config.OUTPUT_PATH = args.o

    # Parse input files
    if os.path.isdir(config.INPUT_PATH):
        config.FILE_LIST = collect_audio_files(config.INPUT_PATH)
        print(f"Found {len(config.FILE_LIST)} files to analyze")
    else:
        config.FILE_LIST = [config.INPUT_PATH]

    # Set confidence threshold
    config.MIN_CONFIDENCE = max(0.01, min(0.99, float(args.min_conf)))

    # Set sensitivity
    config.SIGMOID_SENSITIVITY = max(
        0.5,
        min(1.0 - (float(args.sensitivity) - 1.0), 1.5),
    )

    # Set overlap
    config.SIG_OVERLAP = max(0.0, min(2.9, float(args.overlap)))

    # Set result type
    config.RESULT_TYPE = args.rtype.lower()

    if config.RESULT_TYPE not in [
        "table",
        "audacity",
        "r",
        "kaleidoscope",
        "csv",
    ]:
        config.RESULT_TYPE = "table"

    # Set number of threads
    if os.path.isdir(config.INPUT_PATH):
        config.CPU_THREADS = max(1, int(args.threads))
        config.TFLITE_THREADS = 1
    else:
        config.CPU_THREADS = 1
        config.TFLITE_THREADS = max(1, int(args.threads))

    # Set batch size
    config.BATCH_SIZE = max(1, int(args.batchsize))

    # Add config items to each file list entry.
    # We have to do this for Windows which does not
    # support fork() and thus each process has to
    # have its own config. USE LINUX!
    flist = [(f, config.get_config()) for f in config.FILE_LIST]

    # Analyze files
    if config.CPU_THREADS < 2:
        for entry in flist:
            analyze_file(entry)
    else:
        with Pool(config.CPU_THREADS) as p:
            p.map(analyze_file, flist)

    # A few examples to test
    # PYTHONPATH=. python3 birdnet/analysis/main.py --i example/ --o example/ --slist example/ \
    # --min_conf 0.5 --threads 4
    # PYTHONPATH=. python3 birdnet/analysis/main.py --i example/soundscape.wav --o \
    # example/soundscape.BirdNET.selection.table.txt --slist \
    # example/species_list.txt --threads 8
    # PYTHONPATH=. python3 birdnet/analysis/main.py --i example/ --o example/ --lat 42.5 --lon -76.45 \
    # --week 4 --sensitivity 1.0 --rtype table --locale de
