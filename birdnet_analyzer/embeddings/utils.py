"""Module used to extract embeddings for samples."""

import datetime
import os

import numpy as np

import birdnet_analyzer.analyze as analyze
import birdnet_analyzer.audio as audio
import birdnet_analyzer.config as cfg
import birdnet_analyzer.model as model
import birdnet_analyzer.utils as utils


def write_error_log(msg):
    """
    Appends an error message to the error log file.

    Args:
        msg (str): The error message to be logged.
    """
    with open(cfg.ERROR_LOG_FILE, "a") as elog:
        elog.write(msg + "\n")


def save_as_embeddingsfile(results: dict[str], fpath: str):
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
    cfg.set_config(item[1])

    offset = 0
    duration = cfg.FILE_SPLITTING_DURATION
    fileLengthSeconds = int(audio.get_audio_file_Length(fpath, cfg.SAMPLE_RATE))
    results = {}

    # Start time
    start_time = datetime.datetime.now()

    # Status
    print(f"Analyzing {fpath}", flush=True)

    # Process each chunk
    try:
        while offset < fileLengthSeconds:
            chunks = analyze.get_raw_audio_from_file(fpath, offset, duration)
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
        utils.write_error_log(ex)

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

            save_as_embeddingsfile(
                results, os.path.join(cfg.OUTPUT_PATH, fpath.rsplit(".", 1)[0] + ".birdnet.embeddings.txt")
            )
        else:
            save_as_embeddingsfile(results, cfg.OUTPUT_PATH)

    except Exception as ex:
        # Write error log
        print(f"Error: Cannot save embeddings for {fpath}.", flush=True)
        utils.write_error_log(ex)

        return

    delta_time = (datetime.datetime.now() - start_time).total_seconds()
    print("Finished {} in {:.2f} seconds".format(fpath, delta_time), flush=True)
