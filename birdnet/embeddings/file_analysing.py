import datetime
import os

import numpy

import birdnet.utils
from birdnet.analysis.raw_audio_from_file_getting import \
    get_raw_audio_from_file
from birdnet.configuration import config
from birdnet.embeddings.as_embeddings_file_saving import \
    save_as_embeddings_file
from birdnet.embeddings.error_log_writing import write_error_log
from birdnet.model.main import extract_embeddings


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
        chunks = get_raw_audio_from_file(fpath)
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
        write_error_log(ex)

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

            save_as_embeddings_file(
                results,
                os.path.join(
                    config.OUTPUT_PATH,
                    fpath.rsplit(".", 1)[0] + ".birdnet.embeddings.txt"
                )
            )
        else:
            save_as_embeddings_file(results, config.OUTPUT_PATH)

    except Exception as ex:
        # Write error log
        print(f"Error: Cannot save embeddings for {fpath}.", flush=True)
        write_error_log(ex)

        return

    delta_time = (datetime.datetime.now() - start_time).total_seconds()
    print(
        "Finished {} in {:.2f} seconds".format(fpath, delta_time),
        flush=True,
    )
