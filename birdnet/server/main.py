"""Module to create a remote endpoint for classification.

Can be used to start up a server and feed it classification requests.
"""
import argparse
import json
import os
import tempfile
from datetime import date, datetime
from multiprocessing import freeze_support

import bottle

import birdnet.embeddings.error_log_writing
import birdnet.embeddings.file_analysing
from birdnet.analysis.codes_loading import load_codes
from birdnet.server.result_pooling import pool_results
from birdnet.configuration import config
from birdnet.utils.error_log_writing import write_error_log
from birdnet import species
from birdnet.utils.lines_reading import read_lines


@bottle.route("/healthcheck", method="GET")
def get_health_check():
    """Checks the health of the running server.
    Returns:
        A json message.
    """
    return json.dumps({"msg": "Server is healthy."})


@bottle.route("/analyze", method="POST")
def post_analyze():
    """Handles a classification request.

    Takes a POST request and tries to analyze it.

    The response contains the result or error message.

    Returns:
        A json response with the result.
    """
    # Print divider
    print(f"{'#' * 20}  {datetime.now()}  {'#' * 20}")

    # Get request payload
    upload = bottle.request.files.get("audio")
    mdata = json.loads(bottle.request.forms.get("meta", {}))

    if not upload:
        return json.dumps({"msg": "No audio file."})

    print(mdata)

    # Get filename
    name, ext = os.path.splitext(upload.filename.lower())
    file_path = upload.filename
    file_path_tmp = None

    # Save file
    try:
        if ext[1:].lower() in config.ALLOWED_FILETYPES:
            if mdata.get("save", False):
                save_path = \
                    os.path.join(config.FILE_STORAGE_PATH, str(date.today()))

                os.makedirs(save_path, exist_ok=True)

                file_path = os.path.join(save_path, name + ext)
            else:
                save_path = ""
                file_path_tmp = tempfile.NamedTemporaryFile(
                    suffix=ext.lower(),
                    delete=False,
                )
                file_path_tmp.close()
                file_path = file_path_tmp.name

            upload.save(file_path, overwrite=True)
        else:
            return json.dumps({"msg": "Filetype not supported."})

    except Exception as ex:
        if file_path_tmp:
            os.unlink(file_path_tmp.name)

        # Write error log
        print(f"Error: Cannot save file {file_path}.", flush=True)
        write_error_log(ex)

        # Return error
        return json.dumps({"msg": "Error while saving file."})

    # Analyze file
    try:
        # Set config based on mdata
        if "lat" in mdata and "lon" in mdata:
            config.LATITUDE = float(mdata["lat"])
            config.LONGITUDE = float(mdata["lon"])
        else:
            config.LATITUDE = -1
            config.LONGITUDE = -1

        config.WEEK = int(mdata.get("week", -1))
        config.SIG_OVERLAP = \
            max(0.0, min(2.9, float(mdata.get("overlap", 0.0))))
        config.SIGMOID_SENSITIVITY = \
            max(
                0.5,
                min(1.0 - (float(mdata.get("sensitivity", 1.0)) - 1.0), 1.5)
            )
        config.LOCATION_FILTER_THRESHOLD = \
            max(0.01, min(0.99, float(mdata.get("sf_thresh", 0.03))))

        # Set species list
        if not config.LATITUDE == -1 and not config.LONGITUDE == -1:
            config.SPECIES_LIST_FILE = None
            config.SPECIES_LIST = \
                species.get_species_list(
                    config.LATITUDE,
                    config.LONGITUDE,
                    config.WEEK,
                    config.LOCATION_FILTER_THRESHOLD,
                )
        else:
            config.SPECIES_LIST_FILE = None
            config.SPECIES_LIST = []

        # Analyze file
        success = \
            birdnet.embeddings.file_analysing.analyze_file(
                (file_path, config.get_config()),
            )

        # Parse results
        if success:
            # Open result file
            lines = read_lines(config.OUTPUT_PATH)
            pmode = mdata.get("pmode", "avg").lower()

            # Pool results
            if pmode not in ["avg", "max"]:
                pmode = "avg"

            num_results = min(99, max(1, int(mdata.get("num_results", 5))))

            results = pool_results(lines, num_results, pmode)

            # Prepare response
            data = {"msg": "success", "results": results, "meta": mdata}

            # Save response as metadata file
            if mdata.get("save", False):
                with open(file_path.rsplit(".", 1)[0] + ".json", "w") as f:
                    json.dump(data, f, indent=2)

            # Return response
            del data["meta"]

            return json.dumps(data)

        else:
            return json.dumps({"msg": "Error during analysis."})

    except Exception as e:
        # Write error log
        print(f"Error: Cannot analyze file {file_path}.", flush=True)
        birdnet.embeddings.error_log_writing.write_error_log(e)

        data = {"msg": f"Error during analysis: {e}"}

        return json.dumps(data)
    finally:
        if file_path_tmp:
            os.unlink(file_path_tmp.name)


if __name__ == "__main__":
    # Freeze support for executable
    freeze_support()

    # Parse arguments
    parser = argparse.ArgumentParser(
        description="API endpoint server to analyze files remotely.",
    )
    parser.add_argument(
        "--host",
        default="0.0.0.0",
        help=
        "Host name or IP address of API endpoint server. "
        "Defaults to '0.0.0.0'"
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8080,
        help="Port of API endpoint server. Defaults to 8080.",
    )
    parser.add_argument(
        "--spath",
        default="uploads/",
        help=
        "Path to folder where uploaded files should be stored."
        "Defaults to '/uploads'.",
    )
    parser.add_argument(
        "--threads",
        type=int,
        default=4,
        help="Number of CPU threads for analysis. Defaults to 4.",
    )
    parser.add_argument(
        "--locale",
        default="en",
        help=
        "Locale for translated species common names. "
        "Values in ['af', 'de', 'it', ...]. "
        "Defaults to 'en'.",
    )

    args = parser.parse_args()

    # Load eBird codes, labels
    config.CODES = load_codes()
    config.LABELS = read_lines(config.LABELS_FILE)

    # Load translated labels
    lfile = os.path.join(
        config.TRANSLATED_LABELS_PATH,
        os.path.basename(
            config.LABELS_FILE
        ).replace(
            ".txt",
            "_{}.txt".format(args.locale),
        )
    )

    if not args.locale in ["en"] and os.path.isfile(lfile):
        config.TRANSLATED_LABELS = read_lines(lfile)
    else:
        config.TRANSLATED_LABELS = config.LABELS

    # Set storage file path
    config.FILE_STORAGE_PATH = args.spath

    # Set min_conf to 0.0, because we want all results
    config.MIN_CONFIDENCE = 0.0

    output_file = tempfile.NamedTemporaryFile(suffix=".txt", delete=False)
    output_file.close()

    # Set path for temporary result file
    config.OUTPUT_PATH = output_file.name

    # Set result type
    config.RESULT_TYPE = "audacity"

    # Set number of TFLite threads
    config.TFLITE_THREADS = max(1, int(args.threads))

    # Run server
    print(f"UP AND RUNNING! LISTENING ON {args.host}:{args.port}", flush=True)

    try:
        bottle.run(host=args.host, port=args.port, quiet=True)
    finally:
        os.unlink(output_file.name)
