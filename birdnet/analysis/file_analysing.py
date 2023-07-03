import datetime
import operator
import os

from birdnet.analysis.raw_audio_from_file_getting import \
    get_raw_audio_from_file
from birdnet.analysis.result_file_saving import save_result_file
from birdnet.utils.error_log_writing import write_error_log
from birdnet.analysis.samples_prediction import predict_classes
from birdnet.configuration import config


def analyze_file(item):
    """Analyzes a file.
    Predicts the scores for the file and saves the results.
    Args:
        item: Tuple containing (file path, config)
    Returns:
        The `True` if the file was analyzed successfully.
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

    # If no chunks, show error and skip
    except Exception as ex:
        print(f"Error: Cannot open audio file {fpath}", flush=True)
        write_error_log(ex)

        return False

    # Process each chunk
    try:
        start, end = 0, config.SIG_LENGTH
        results = {}
        samples = []
        timestamps = []

        for chunk_index, chunk in enumerate(chunks):
            # Add to batch
            samples.append(chunk)
            timestamps.append([start, end])

            # Advance start and end
            start += config.SIG_LENGTH - config.SIG_OVERLAP
            end = start + config.SIG_LENGTH

            # Check if batch is full or last chunk
            if len(samples) < config.BATCH_SIZE and \
                    chunk_index < len(chunks) - 1:
                continue

            # Predict
            p = predict_classes(samples)

            # Add to results
            for i in range(len(samples)):
                # Get timestamp
                s_start, s_end = timestamps[i]

                # Get prediction
                pred = p[i]

                # Assign scores to labels
                p_labels = zip(config.LABELS, pred)

                # Sort by score
                p_sorted = sorted(
                    p_labels,
                    key=operator.itemgetter(1),
                    reverse=True,
                )

                # Store top 5 results and advance indices
                results[str(s_start) + "-" + str(s_end)] = p_sorted

            # Clear batch
            samples = []
            timestamps = []

    except Exception as ex:
        # Write error log
        print(f"Error: Cannot analyze audio file {fpath}.\n", flush=True)
        write_error_log(ex)

        return False

    # Save as selection table
    try:
        # We have to check if output path is a file or directory
        if not config.OUTPUT_PATH.rsplit(".", 1)[-1].lower() in ["txt", "csv"]:
            rpath = fpath.replace(config.INPUT_PATH, "")
            rpath = rpath[1:] if rpath[0] in ["/", "\\"] else rpath

            # Make target directory if it doesn't exist
            rdir = os.path.join(config.OUTPUT_PATH, os.path.dirname(rpath))

            os.makedirs(rdir, exist_ok=True)

            if config.RESULT_TYPE == "table":
                rtype = ".BirdNET.selection.table.txt"
            elif config.RESULT_TYPE == "audacity":
                rtype = ".BirdNET.results.txt"
            else:
                rtype = ".BirdNET.results.csv"

            save_result_file(
                results,
                os.path.join(
                    config.OUTPUT_PATH, rpath.rsplit(".", 1)[0] + rtype
                ),
                fpath,
            )
        else:
            save_result_file(
                results,
                config.OUTPUT_PATH,
                fpath,
            )

    except Exception as ex:
        # Write error log
        print(f"Error: Cannot save result for {fpath}.\n", flush=True)
        write_error_log(ex)

        return False

    delta_time = (datetime.datetime.now() - start_time).total_seconds()
    print(
        "Finished {} in {:.2f} seconds".format(fpath, delta_time),
        flush=True,
    )

    return True
