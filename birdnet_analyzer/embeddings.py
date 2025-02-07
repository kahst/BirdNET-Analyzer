"""Module used to extract embeddings for samples."""

import argparse
import datetime
import os
from multiprocessing import Pool

import numpy as np

import birdnet_analyzer.analyze as analyze
import birdnet_analyzer.audio as audio
import birdnet_analyzer.config as cfg
import birdnet_analyzer.model as model
import birdnet_analyzer.utils as utils


def writeErrorLog(msg):
    """
    Appends an error message to the error log file.

    Args:
        msg (str): The error message to be logged.
    """
    with open(cfg.ERROR_LOG_FILE, "a") as elog:
        elog.write(msg + "\n")


def saveAsEmbeddingsFile(results: dict[str], fpath: str):
    """Write embeddings to file

    Args:
        results: A dictionary containing the embeddings at timestamp.
        fpath: The path for the embeddings file.
    """
    with open(fpath, "w") as f:
        for timestamp in results:
            f.write(timestamp.replace("-", "\t") + "\t" + ",".join(map(str, results[timestamp])) + "\n")


def analyzeFile(item):
    """Extracts the embeddings for a file.

    Args:
        item: (filepath, config)
    """
    # Get file path and restore cfg
    fpath: str = item[0]
    cfg.setConfig(item[1])

    offset = 0
    duration = cfg.FILE_SPLITTING_DURATION
    fileLengthSeconds = int(audio.getAudioFileLength(fpath, cfg.SAMPLE_RATE))
    results = {}

    # Start time
    start_time = datetime.datetime.now()

    # Status
    print(f"Analyzing {fpath}", flush=True)

    # Process each chunk
    try:
        while offset < fileLengthSeconds:
            chunks = analyze.getRawAudioFromFile(fpath, offset, duration)
            start, end = offset, cfg.SIG_LENGTH + offset
            samples = []
            timestamps = []

            for c in range(len(chunks)):
                # Add to batch
                samples.append(chunks[c])
                timestamps.append([start, end])

                # Advance start and end
                start += cfg.SIG_LENGTH - cfg.SIG_OVERLAP
                end = start + cfg.SIG_LENGTH

                # Check if batch is full or last chunk
                if len(samples) < cfg.BATCH_SIZE and c < len(chunks) - 1:
                    continue

                # Prepare sample and pass through model
                data = np.array(samples, dtype="float32")
                e = model.embeddings(data)

                # Add to results
                for i in range(len(samples)):
                    # Get timestamp
                    s_start, s_end = timestamps[i]

                    # Get prediction
                    embeddings = e[i]

                    # Store embeddings
                    results[f"{s_start}-{s_end}"] = embeddings

                # Reset batch
                samples = []
                timestamps = []

            offset = offset + duration

    except Exception as ex:
        # Write error log
        print(f"Error: Cannot analyze audio file {fpath}.", flush=True)
        utils.writeErrorLog(ex)

        return

    # Save as embeddings file
    try:
        # We have to check if output path is a file or directory
        if cfg.OUTPUT_PATH.rsplit(".", 1)[-1].lower() not in ["txt", "csv"]:
            fpath = fpath.replace(cfg.INPUT_PATH, "")
            fpath = fpath[1:] if fpath[0] in ["/", "\\"] else fpath

            # Make target directory if it doesn't exist
            fdir = os.path.join(cfg.OUTPUT_PATH, os.path.dirname(fpath))
            os.makedirs(fdir, exist_ok=True)

            saveAsEmbeddingsFile(
                results, os.path.join(cfg.OUTPUT_PATH, fpath.rsplit(".", 1)[0] + ".birdnet.embeddings.txt")
            )
        else:
            saveAsEmbeddingsFile(results, cfg.OUTPUT_PATH)

    except Exception as ex:
        # Write error log
        print(f"Error: Cannot save embeddings for {fpath}.", flush=True)
        utils.writeErrorLog(ex)

        return

    delta_time = (datetime.datetime.now() - start_time).total_seconds()
    print("Finished {} in {:.2f} seconds".format(fpath, delta_time), flush=True)


if __name__ == "__main__":
    # Parse arguments
    parser = argparse.ArgumentParser(
        description="Extract feature embeddings with BirdNET",
        parents=[utils.io_args(), utils.bandpass_args(), utils.overlap_args(), utils.threads_args(), utils.bs_args()],
    )

    args = parser.parse_args()

    ### Make sure to comment out appropriately if you are not using args. ###

    # Set input and output path
    cfg.INPUT_PATH = args.input

    if not args.output:
        if os.path.isfile(cfg.INPUT_PATH):
            cfg.OUTPUT_PATH = os.path.dirname(cfg.INPUT_PATH)
        else:
            cfg.OUTPUT_PATH = cfg.INPUT_PATH
    else:
        cfg.OUTPUT_PATH = args.output

    # Parse input files
    if os.path.isdir(cfg.INPUT_PATH):
        cfg.FILE_LIST = utils.collect_audio_files(cfg.INPUT_PATH)
    else:
        cfg.FILE_LIST = [cfg.INPUT_PATH]

    # Set overlap
    cfg.SIG_OVERLAP = args.overlap

    # Set bandpass frequency range
    cfg.BANDPASS_FMIN = args.fmin
    cfg.BANDPASS_FMAX = args.fmax

    # Set number of threads
    if os.path.isdir(cfg.INPUT_PATH):
        cfg.CPU_THREADS = args.threads
        cfg.TFLITE_THREADS = 1
    else:
        cfg.CPU_THREADS = 1
        cfg.TFLITE_THREADS = args.threads

    # Set batch size
    cfg.BATCH_SIZE = args.batchsize

    # Add config items to each file list entry.
    # We have to do this for Windows which does not
    # support fork() and thus each process has to
    # have its own config. USE LINUX!
    flist = [(f, cfg.getConfig()) for f in cfg.FILE_LIST]

    # Analyze files
    if cfg.CPU_THREADS < 2:
        for entry in flist:
            analyzeFile(entry)
    else:
        with Pool(cfg.CPU_THREADS) as p:
            p.map(analyzeFile, flist)
