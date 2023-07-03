"""Extract segments from audio files based on BirdNET detections.

Can be used to save the segments of the audio files for each detection.
"""
import argparse
from multiprocessing import Pool

import numpy

from birdnet.segments.segments_extracting import extract_segments
from birdnet.segments.files_parsing import parse_files
from birdnet.segments.folders_parsing import parse_folders

from birdnet.configuration import config

# Set numpy random seed
numpy.random.seed(config.RANDOM_SEED)


if __name__ == "__main__":
    # Parse arguments
    parser = argparse.ArgumentParser(
        description=
        "Extract segments from audio files based on BirdNET detections."
    )
    parser.add_argument(
        "--audio",
        default="example/",
        help="Path to folder containing audio files.",
    )
    parser.add_argument(
        "--results",
        default="example/",
        help="Path to folder containing result files.",
    )
    parser.add_argument(
        "--o",
        default="example/",
        help="Output folder path for extracted segments.",
    )
    parser.add_argument(
        "--min_conf",
        type=float,
        default=0.1,
        help=
        "Minimum confidence threshold. "
        "Values in [0.01, 0.99]. "
        "Defaults to 0.1.",
    )
    parser.add_argument(
        "--max_segments",
        type=int,
        default=100,
        help="Number of randomly extracted segments per species.",
    )
    parser.add_argument(
        "--seg_length",
        type=float,
        default=3.0,
        help="Length of extracted segments in seconds. Defaults to 3.0.",
    )
    parser.add_argument(
        "--threads",
        type=int,
        default=4,
        help="Number of CPU threads.",
    )

    args = parser.parse_args()

    # Parse audio and result folders
    config.FILE_LIST = parse_folders(args.audio, args.results)

    # Set output folder
    config.OUTPUT_PATH = args.o

    # Set number of threads
    config.CPU_THREADS = int(args.threads)

    # Set confidence threshold
    config.MIN_CONFIDENCE = max(0.01, min(0.99, float(args.min_conf)))

    # Parse file list and make list of segments
    config.FILE_LIST = \
        parse_files(config.FILE_LIST, max(1, int(args.max_segments)))

    # Add config items to each file list entry.
    # We have to do this for Windows which does not
    # support fork() and thus each process has to
    # have its own config. USE LINUX!
    flist = [
        (
            entry,
            max(config.SIG_LENGTH, float(args.seg_length)),
            config.get_config(),
        )
        for entry in config.FILE_LIST
    ]

    # Extract segments
    if config.CPU_THREADS < 2:
        for entry in flist:
            extract_segments(entry)
    else:
        with Pool(config.CPU_THREADS) as p:
            p.map(extract_segments, flist)

    # A few examples to test
    # python3 segments.py --audio example/ --results example/ --o \
    # example/segments/
    # python3 segments.py --audio example/ --results example/ --o \
    # example/segments/ --seg_length 5.0 --min_conf 0.1 --max_segments 100 \
    # --threads 4
