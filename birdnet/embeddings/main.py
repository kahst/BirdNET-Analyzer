"""Module used to extract embeddings for samples.
"""
import argparse
import datetime
import os
import sys
from multiprocessing import Pool
from typing import Dict

import numpy

import analyze
from birdnet.configuration import config
import birdnet.utils.error_log_writing
import birdnet.utils.utils as utils
from birdnet.model.main import extract_embeddings


def write_error_log(msg):
    with open(config.ERROR_LOG_FILE, "a") as error_log:
        error_log.write(msg + "\n")


def save_as_embeddings_file(results: Dict[str], fpath: str):
    """Write embeddings to file
    
    Args:
        results: A dictionary containing the embeddings at timestamp.
        fpath: The path for the embeddings file.
    """
    with open(fpath, "w") as f:
        for timestamp in results:
            f.write(timestamp.replace("-", "\t") + "\t" + ",".join(map(str, results[timestamp])) + "\n")


def analyze_file(item):
    """Extracts the embeddings for a file.

    Args:
        item: (filepath, config)
    """
    # Get file path and restore cfg
    fpath: str = item[0]
    config.set_config(item[1])

    # Start time
    start_time = datetime.datetime.now()

    # Status
    print(f"Analyzing {fpath}", flush=True)

    try:
        # Open audio file and split into 3-second chunks
        chunks = analyze.get_raw_audio_from_file(fpath)
    except Exception as ex:
        print(f"Error: Cannot open audio file {fpath}", flush=True)
        birdnet.utils.error_log_writing.write_error_log(ex)

        return
    
    # If no chunks, show error and skip
    if len(chunks) == 0:
        msg = f"Error: Cannot open audio file {fpath}"
        print(msg, flush=True)
        write_error_log(msg)

        return

    # Process each chunk
    try:
        start, end = 0, config.SIG_LENGTH
        results = {}
        samples = []
        timestamps = []

        for c in range(len(chunks)):
            # Add to batch
            samples.append(chunks[c])
            timestamps.append([start, end])

            # Advance start and end
            start += config.SIG_LENGTH - config.SIG_OVERLAP
            end = start + config.SIG_LENGTH

            # Check if batch is full or last chunk
            if len(samples) < config.BATCH_SIZE and c < len(chunks) - 1:
                continue

            # Prepare sample and pass through model
            data = numpy.array(samples, dtype="float32")
            e = extract_embeddings(data)

            # Add to results
            for i in range(len(samples)):
                # Get timestamp
                s_start, s_end = timestamps[i]

                # Get prediction
                embeddings = e[i]

                # Store embeddings
                results[str(s_start) + "-" + str(s_end)] = embeddings

            # Reset batch
            samples = []
            timestamps = []

    except Exception as ex:
        # Write error log
        print(f"Error: Cannot analyze audio file {fpath}.", flush=True)
        birdnet.utils.error_log_writing.write_error_log(ex)

        return

    # Save as embeddings file
    try:
        # We have to check if output path is a file or directory
        if not config.OUTPUT_PATH.rsplit(".", 1)[-1].lower() in ["txt", "csv"]:
            fpath = fpath.replace(config.INPUT_PATH, "")
            fpath = fpath[1:] if fpath[0] in ["/", "\\"] else fpath

            # Make target directory if it doesn't exist
            fdir = os.path.join(config.OUTPUT_PATH, os.path.dirname(fpath))
            os.makedirs(fdir, exist_ok=True)

            save_as_embeddings_file(results, os.path.join(config.OUTPUT_PATH, fpath.rsplit(".", 1)[0] + ".birdnet.embeddings.txt"))
        else:
            save_as_embeddings_file(results, config.OUTPUT_PATH)

    except Exception as ex:
        # Write error log
        print(f"Error: Cannot save embeddings for {fpath}.", flush=True)
        birdnet.utils.error_log_writing.write_error_log(ex)

        return

    delta_time = (datetime.datetime.now() - start_time).total_seconds()
    print("Finished {} in {:.2f} seconds".format(fpath, delta_time), flush=True)


if __name__ == "__main__":
    # Parse arguments
    parser = argparse.ArgumentParser(description="Analyze audio files with BirdNET")
    parser.add_argument(
        "--i", default="example/", help="Path to input file or folder. If this is a file, --o needs to be a file too."
    )
    parser.add_argument(
        "--o", default="example/", help="Path to output file or folder. If this is a file, --i needs to be a file too."
    )
    parser.add_argument(
        "--overlap", type=float, default=0.0, help="Overlap of prediction segments. Values in [0.0, 2.9]. Defaults to 0.0."
    )
    parser.add_argument("--threads", type=int, default=4, help="Number of CPU threads.")
    parser.add_argument(
        "--batchsize", type=int, default=1, help="Number of samples to process at the same time. Defaults to 1."
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
        config.FILE_LIST = utils.collect_audio_files(config.INPUT_PATH)
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
    # python3 embeddings.py --i example/soundscape.wav --o example/soundscape.birdnet.embeddings.txt --threads 4
