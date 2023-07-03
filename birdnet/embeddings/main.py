"""Module used to extract embeddings for samples.
"""
import argparse
import os
import sys
from multiprocessing import Pool

from birdnet.configuration import config

from birdnet.embeddings.file_analysing import analyze_file
from birdnet.utils.audio_file_collecting import collect_audio_files

if __name__ == "__main__":
    # Parse arguments
    parser = argparse.ArgumentParser(
        description="Analyze audio files with BirdNET",
    )
    parser.add_argument(
        "--i",
        default="example/",
        help=
        "Path to input file or folder. "
        "If this is a file, --o needs to be a file too."
    )
    parser.add_argument(
        "--o",
        default="example/",
        help=
        "Path to output file or folder. "
        "If this is a file, --i needs to be a file too."
    )
    parser.add_argument(
        "--overlap",
        type=float,
        default=0.0,
        help=
        "Overlap of prediction segments. "
        "Values in [0.0, 2.9]. "
        "Defaults to 0.0.",
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
        help=
        "Number of samples to process at the same time. "
        "Defaults to 1.",
    )

    args = parser.parse_args()

    # Set paths relative to script path (requested in #3)
    config.MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(sys.argv[0])), config.MODEL_PATH)
    config.ERROR_LOG_FILE = os.path.join(os.path.dirname(os.path.abspath(sys.argv[0])), config.ERROR_LOG_FILE)

    ### Make sure to comment out appropriately if you are not using args. ###

    # Set input and output path
    config.INPUT_PATH = args.i
    config.OUTPUT_PATH = args.o

    # Parse input files
    if os.path.isdir(config.INPUT_PATH):
        config.FILE_LIST = collect_audio_files(config.INPUT_PATH)
    else:
        config.FILE_LIST = [config.INPUT_PATH]

    # Set overlap
    config.SIG_OVERLAP = max(0.0, min(2.9, float(args.overlap)))

    # Set number of threads
    if os.path.isdir(config.INPUT_PATH):
        config.CPU_THREADS = int(args.threads)
        config.TFLITE_THREADS = 1
    else:
        config.CPU_THREADS = 1
        config.TFLITE_THREADS = int(args.threads)

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
    # python3 embeddings.py --i example/ --o example/ --threads 4
    # python3 embeddings.py --i example/soundscape.wav --o \
    # example/soundscape.birdnet.embeddings.txt --threads 4
