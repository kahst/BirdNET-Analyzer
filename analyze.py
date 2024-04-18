"""Module to analyze audio samples.
"""

import argparse
import datetime
import json
import operator
import os
import sys
import multiprocessing
from multiprocessing import Pool, freeze_support

import numpy as np

import audio
import config as cfg
import model
import species
import utils

#                    0       1      2           3             4              5               6                7           8             9           10         11
RTABLE_HEADER = "Selection\tView\tChannel\tBegin Time (s)\tEnd Time (s)\tLow Freq (Hz)\tHigh Freq (Hz)\tCommon Name\tSpecies Code\tConfidence\tBegin Path\tFile Offset (s)\n"


def loadCodes():
    """Loads the eBird codes.

    Returns:
        A dictionary containing the eBird codes.
    """
    with open(cfg.CODES_FILE, "r") as cfile:
        codes = json.load(cfile)

    return codes


def saveResultFile(r: dict[str, list], path: str, afile_path: str):
    """Saves the results to the hard drive.

    Args:
        r: The dictionary with {segment: scores}.
        path: The path where the result should be saved.
        afile_path: The path to audio file.
    """
    # Make folder if it doesn't exist
    if os.path.dirname(path):
        os.makedirs(os.path.dirname(path), exist_ok=True)

    # Selection table
    out_string = ""

    if cfg.RESULT_TYPE == "table":
        selection_id = 0
        filename = os.path.basename(afile_path)

        # Write header
        out_string += RTABLE_HEADER

        # Read native sample rate
        high_freq = audio.get_sample_rate(afile_path) / 2

        if high_freq > cfg.SIG_FMAX:
            high_freq = cfg.SIG_FMAX

        high_freq = min(high_freq, cfg.BANDPASS_FMAX)
        low_freq = max(cfg.SIG_FMIN, cfg.BANDPASS_FMIN)

        # Extract valid predictions for every timestamp
        for timestamp in getSortedTimestamps(r):
            rstring = ""
            start, end = timestamp.split("-", 1)

            for c in r[timestamp]:
                if c[1] > cfg.MIN_CONFIDENCE and (not cfg.SPECIES_LIST or c[0] in cfg.SPECIES_LIST):
                    selection_id += 1
                    label = cfg.TRANSLATED_LABELS[cfg.LABELS.index(c[0])]
                    code = cfg.CODES[c[0]] if c[0] in cfg.CODES else c[0]
                    rstring += f"{selection_id}\tSpectrogram 1\t1\t{start}\t{end}\t{low_freq}\t{high_freq}\t{label.split('_', 1)[-1]}\t{code}\t{c[1]:.4f}\t{afile_path}\t{start}\n"

            # Write result string to file
            out_string += rstring

        # If we don't have any valid predictions, we still need to add a line to the selection table in case we want to combine results
        # TODO: That's a weird way to do it, but it works for now. It would be better to keep track of file durations during the analysis.
        if len(out_string) == len(RTABLE_HEADER) and cfg.OUTPUT_PATH is not None:
            selection_id += 1
            out_string += f"{selection_id}\tSpectrogram 1\t1\t0\t3\t{low_freq}\t{high_freq}\tnocall\tnocall\t1.0\t{afile_path}\t0\n"

    elif cfg.RESULT_TYPE == "audacity":
        # Audacity timeline labels
        for timestamp in getSortedTimestamps(r):
            rstring = ""

            for c in r[timestamp]:
                if c[1] > cfg.MIN_CONFIDENCE and (not cfg.SPECIES_LIST or c[0] in cfg.SPECIES_LIST):
                    label = cfg.TRANSLATED_LABELS[cfg.LABELS.index(c[0])]
                    ts = timestamp.replace("-", "\t")
                    lbl = label.replace("_", ", ")
                    rstring += f"{ts}\t{lbl}\t{c[1]:.4f}\n"

            # Write result string to file
            out_string += rstring

    elif cfg.RESULT_TYPE == "r":
        # Output format for R
        header = "filepath,start,end,scientific_name,common_name,confidence,lat,lon,week,overlap,sensitivity,min_conf,species_list,model"
        out_string += header

        for timestamp in getSortedTimestamps(r):
            rstring = ""
            start, end = timestamp.split("-", 1)

            for c in r[timestamp]:
                if c[1] > cfg.MIN_CONFIDENCE and (not cfg.SPECIES_LIST or c[0] in cfg.SPECIES_LIST):
                    label = cfg.TRANSLATED_LABELS[cfg.LABELS.index(c[0])]
                    rstring += "\n{},{},{},{},{},{:.4f},{:.4f},{:.4f},{},{},{},{},{},{}".format(
                        afile_path,
                        start,
                        end,
                        label.split("_", 1)[0],
                        label.split("_", 1)[-1],
                        c[1],
                        cfg.LATITUDE,
                        cfg.LONGITUDE,
                        cfg.WEEK,
                        cfg.SIG_OVERLAP,
                        (1.0 - cfg.SIGMOID_SENSITIVITY) + 1.0,
                        cfg.MIN_CONFIDENCE,
                        cfg.SPECIES_LIST_FILE,
                        os.path.basename(cfg.MODEL_PATH),
                    )

            # Write result string to file
            out_string += rstring

    elif cfg.RESULT_TYPE == "kaleidoscope":
        # Output format for kaleidoscope
        header = "INDIR,FOLDER,IN FILE,OFFSET,DURATION,scientific_name,common_name,confidence,lat,lon,week,overlap,sensitivity"
        out_string += header

        folder_path, filename = os.path.split(afile_path)
        parent_folder, folder_name = os.path.split(folder_path)

        for timestamp in getSortedTimestamps(r):
            rstring = ""
            start, end = timestamp.split("-", 1)

            for c in r[timestamp]:
                if c[1] > cfg.MIN_CONFIDENCE and (not cfg.SPECIES_LIST or c[0] in cfg.SPECIES_LIST):
                    label = cfg.TRANSLATED_LABELS[cfg.LABELS.index(c[0])]
                    rstring += "\n{},{},{},{},{},{},{},{:.4f},{:.4f},{:.4f},{},{},{}".format(
                        parent_folder.rstrip("/"),
                        folder_name,
                        filename,
                        start,
                        float(end) - float(start),
                        label.split("_", 1)[0],
                        label.split("_", 1)[-1],
                        c[1],
                        cfg.LATITUDE,
                        cfg.LONGITUDE,
                        cfg.WEEK,
                        cfg.SIG_OVERLAP,
                        (1.0 - cfg.SIGMOID_SENSITIVITY) + 1.0,
                    )

            # Write result string to file
            out_string += rstring

    else:
        # CSV output file
        header = "Start (s),End (s),Scientific name,Common name,Confidence\n"

        # Write header
        out_string += header

        for timestamp in getSortedTimestamps(r):
            rstring = ""

            for c in r[timestamp]:
                start, end = timestamp.split("-", 1)

                if c[1] > cfg.MIN_CONFIDENCE and (not cfg.SPECIES_LIST or c[0] in cfg.SPECIES_LIST):
                    label = cfg.TRANSLATED_LABELS[cfg.LABELS.index(c[0])]
                    rstring += "{},{},{},{},{:.4f}\n".format(
                        start, end, label.split("_", 1)[0], label.split("_", 1)[-1], c[1]
                    )

            # Write result string to file
            out_string += rstring

    # Save as file
    with open(path, "w", encoding="utf-8") as rfile:
        rfile.write(out_string)


def combineResults(folder: str, output_file: str):

    # Read all files
    files = utils.collect_all_files(folder, ["txt"], pattern="BirdNET.selection.table")

    # Combine all files
    s_id = 1
    time_offset = 0
    audiofiles = set()

    with open(os.path.join(folder, output_file), "w", encoding="utf-8") as f:
        f.write(RTABLE_HEADER)

        for rfile in files:
            with open(rfile, "r", encoding="utf-8") as rf:

                try:
                    lines = rf.readlines()

                    # make sure it's a selection table
                    if not "Selection" in lines[0] or not "File Offset" in lines[0]:
                        continue

                    # skip header and add to file
                    f_name = lines[1].split("\t")[10]
                    f_duration = audio.getAudioFileLength(f_name, cfg.SAMPLE_RATE)

                    audiofiles.add(f_name)

                    for line in lines[1:]:

                        # empty line?
                        if not line.strip():
                            continue

                        # Is species code and common name == 'nocall'?
                        # If so, that's a dummy line and we can skip it
                        if line.split("\t")[7] == "nocall" and line.split("\t")[8] == "nocall":
                            continue

                        # adjust selection id
                        line = line.split("\t")
                        line[0] = str(s_id)
                        s_id += 1

                        # adjust time
                        line[3] = str(float(line[3]) + time_offset)
                        line[4] = str(float(line[4]) + time_offset)

                        # write line
                        f.write("\t".join(line))

                    # adjust time offset
                    time_offset += f_duration

                except Exception as ex:
                    print(f"Error: Cannot combine results from {rfile}.\n", flush=True)
                    utils.writeErrorLog(ex)

    listfilesname = output_file.rsplit(".", 1)[0] + ".list.txt"

    with open(os.path.join(folder, listfilesname), "w", encoding="utf-8") as f:
        f.writelines((f + "\n" for f in audiofiles))


def getSortedTimestamps(results: dict[str, list]):
    """Sorts the results based on the segments.

    Args:
        results: The dictionary with {segment: scores}.

    Returns:
        Returns the sorted list of segments and their scores.
    """
    return sorted(results, key=lambda t: float(t.split("-", 1)[0]))


def getRawAudioFromFile(fpath: str, offset, duration):
    """Reads an audio file.

    Reads the file and splits the signal into chunks.

    Args:
        fpath: Path to the audio file.

    Returns:
        The signal split into a list of chunks.
    """
    # Open file
    sig, rate = audio.openAudioFile(fpath, cfg.SAMPLE_RATE, offset, duration, cfg.BANDPASS_FMIN, cfg.BANDPASS_FMAX)

    # Split into raw audio chunks
    chunks = audio.splitSignal(sig, rate, cfg.SIG_LENGTH, cfg.SIG_OVERLAP, cfg.SIG_MINLEN)

    return chunks


def predict(samples):
    """Predicts the classes for the given samples.

    Args:
        samples: Samples to be predicted.

    Returns:
        The prediction scores.
    """
    # Prepare sample and pass through model
    data = np.array(samples, dtype="float32")
    prediction = model.predict(data)

    # Logits or sigmoid activations?
    if cfg.APPLY_SIGMOID:
        prediction = model.flat_sigmoid(np.array(prediction), sensitivity=-cfg.SIGMOID_SENSITIVITY)

    return prediction


def get_result_file_name(fpath: str):
    # We have to check if output path is a file or directory
    if not cfg.OUTPUT_PATH.rsplit(".", 1)[-1].lower() in ["txt", "csv"]:
        rpath = fpath.replace(cfg.INPUT_PATH, "")
        rpath = rpath[1:] if rpath[0] in ["/", "\\"] else rpath

        # Make target directory if it doesn't exist
        rdir = os.path.join(cfg.OUTPUT_PATH, os.path.dirname(rpath))

        os.makedirs(rdir, exist_ok=True)

        if cfg.RESULT_TYPE == "table":
            rtype = ".BirdNET.selection.table.txt"
        elif cfg.RESULT_TYPE == "audacity":
            rtype = ".BirdNET.results.txt"
        else:
            rtype = ".BirdNET.results.csv"

        return os.path.join(cfg.OUTPUT_PATH, rpath.rsplit(".", 1)[0] + rtype)

    return cfg.OUTPUT_PATH


def analyzeFile(item):
    """Analyzes a file.

    Predicts the scores for the file and saves the results.

    Args:
        item: Tuple containing (file path, config)

    Returns:
        The `True` if the file was analyzed successfully.
    """
    # Get file path and restore cfg
    fpath: str = item[0]
    cfg.setConfig(item[1])

    # Start time
    start_time = datetime.datetime.now()
    offset = 0
    duration = cfg.FILE_SPLITTING_DURATION
    start, end = 0, cfg.SIG_LENGTH
    fileLengthSeconds = audio.getAudioFileLength(fpath, cfg.SAMPLE_RATE)
    results = {}
    result_file_name = get_result_file_name(fpath)

    if cfg.SKIP_EXISTING_RESULTS and os.path.exists(result_file_name):
        print(f"Skipping {fpath} as it has already been analyzed", flush=True)
        return True

    # Status
    print(f"Analyzing {fpath}", flush=True)

    # Process each chunk
    try:
        while offset < fileLengthSeconds:
            chunks = getRawAudioFromFile(fpath, offset, duration)
            samples = []
            timestamps = []

            for chunk_index, chunk in enumerate(chunks):
                # Add to batch
                samples.append(chunk)
                timestamps.append([start, end])

                # Advance start and end
                start += cfg.SIG_LENGTH - cfg.SIG_OVERLAP
                end = start + cfg.SIG_LENGTH

                # Check if batch is full or last chunk
                if len(samples) < cfg.BATCH_SIZE and chunk_index < len(chunks) - 1:
                    continue

                # Predict
                p = predict(samples)

                # Add to results
                for i in range(len(samples)):
                    # Get timestamp
                    s_start, s_end = timestamps[i]

                    # Get prediction
                    pred = p[i]

                    # Assign scores to labels
                    p_labels = zip(cfg.LABELS, pred)

                    # Sort by score
                    p_sorted = sorted(p_labels, key=operator.itemgetter(1), reverse=True)

                    # Store top 5 results and advance indices
                    results[str(s_start) + "-" + str(s_end)] = p_sorted

                # Clear batch
                samples = []
                timestamps = []
            offset = offset + duration

    except Exception as ex:
        # Write error log
        print(f"Error: Cannot analyze audio file {fpath}.\n", flush=True)
        utils.writeErrorLog(ex)

        return False

    # Save as selection table
    try:
        saveResultFile(results, result_file_name, fpath)

    except Exception as ex:
        # Write error log
        print(f"Error: Cannot save result for {fpath}.\n", flush=True)
        utils.writeErrorLog(ex)

        return False

    delta_time = (datetime.datetime.now() - start_time).total_seconds()
    print(f"Finished {fpath} in {delta_time:.2f} seconds", flush=True)

    return True


if __name__ == "__main__":
    # Freeze support for executable
    freeze_support()

    # Parse arguments
    parser = argparse.ArgumentParser(description="Analyze audio files with BirdNET")
    parser.add_argument(
        "--i", default="example/", help="Path to input file or folder. If this is a file, --o needs to be a file too."
    )
    parser.add_argument(
        "--o", default="example/", help="Path to output file or folder. If this is a file, --i needs to be a file too."
    )
    parser.add_argument("--lat", type=float, default=-1, help="Recording location latitude. Set -1 to ignore.")
    parser.add_argument("--lon", type=float, default=-1, help="Recording location longitude. Set -1 to ignore.")
    parser.add_argument(
        "--week",
        type=int,
        default=-1,
        help="Week of the year when the recording was made. Values in [1, 48] (4 weeks per month). Set -1 for year-round species list.",
    )
    parser.add_argument(
        "--slist",
        default="",
        help='Path to species list file or folder. If folder is provided, species list needs to be named "species_list.txt". If lat and lon are provided, this list will be ignored.',
    )
    parser.add_argument(
        "--sensitivity",
        type=float,
        default=1.0,
        help="Detection sensitivity; Higher values result in higher sensitivity. Values in [0.5, 1.5]. Defaults to 1.0.",
    )
    parser.add_argument(
        "--min_conf",
        type=float,
        default=0.1,
        help="Minimum confidence threshold. Values in [0.01, 0.99]. Defaults to 0.1.",
    )
    parser.add_argument(
        "--overlap",
        type=float,
        default=0.0,
        help="Overlap of prediction segments. Values in [0.0, 2.9]. Defaults to 0.0.",
    )
    parser.add_argument(
        "--rtype",
        default="table",
        help="Specifies output format. Values in ['table', 'audacity', 'r',  'kaleidoscope', 'csv']. Defaults to 'table' (Raven selection table).",
    )
    parser.add_argument(
        "--output_file",
        default=None,
        help="Path to combined Raven selection table. If set and rtype is 'table', all results will be combined into this file. Defaults to None.",
    )
    parser.add_argument(
        "--threads", type=int, default=min(8, max(1, multiprocessing.cpu_count() // 2)), help="Number of CPU threads."
    )
    parser.add_argument(
        "--batchsize", type=int, default=1, help="Number of samples to process at the same time. Defaults to 1."
    )
    parser.add_argument(
        "--locale",
        default="en",
        help="Locale for translated species common names. Values in ['af', 'en_UK', 'de', 'it', ...] Defaults to 'en' (US English).",
    )
    parser.add_argument(
        "--sf_thresh",
        type=float,
        default=0.03,
        help="Minimum species occurrence frequency threshold for location filter. Values in [0.01, 0.99]. Defaults to 0.03.",
    )
    parser.add_argument(
        "--classifier",
        default=None,
        help="Path to custom trained classifier. Defaults to None. If set, --lat, --lon and --locale are ignored.",
    )
    parser.add_argument(
        "--fmin",
        type=int,
        default=cfg.SIG_FMIN,
        help=f"Minimum frequency for bandpass filter in Hz. Defaults to {cfg.SIG_FMIN} Hz.",
    )
    parser.add_argument(
        "--fmax",
        type=int,
        default=cfg.SIG_FMAX,
        help=f"Maximum frequency for bandpass filter in Hz. Defaults to {cfg.SIG_FMAX} Hz.",
    )
    parser.add_argument(
        "--skip_existing_results",
        action="store_true",
        help="Skip files that have already been analyzed. Defaults to False.",
    )

    args = parser.parse_args()

    # Set paths relative to script path (requested in #3)
    script_dir = os.path.dirname(os.path.abspath(sys.argv[0]))
    cfg.MODEL_PATH = os.path.join(script_dir, cfg.MODEL_PATH)
    cfg.LABELS_FILE = os.path.join(script_dir, cfg.LABELS_FILE)
    cfg.TRANSLATED_LABELS_PATH = os.path.join(script_dir, cfg.TRANSLATED_LABELS_PATH)
    cfg.MDATA_MODEL_PATH = os.path.join(script_dir, cfg.MDATA_MODEL_PATH)
    cfg.CODES_FILE = os.path.join(script_dir, cfg.CODES_FILE)
    cfg.ERROR_LOG_FILE = os.path.join(script_dir, cfg.ERROR_LOG_FILE)

    # Load eBird codes, labels
    cfg.CODES = loadCodes()
    cfg.LABELS = utils.readLines(cfg.LABELS_FILE)

    cfg.SKIP_EXISTING_RESULTS = args.skip_existing_results

    # Set custom classifier?
    if args.classifier is not None:
        cfg.CUSTOM_CLASSIFIER = args.classifier  # we treat this as absolute path, so no need to join with dirname

        if args.classifier.endswith(".tflite"):
            cfg.LABELS_FILE = args.classifier.replace(".tflite", "_Labels.txt")  # same for labels file
            cfg.LABELS = utils.readLines(cfg.LABELS_FILE)
        else:
            cfg.APPLY_SIGMOID = False
            cfg.LABELS_FILE = os.path.join(args.classifier, "labels", "label_names.csv")
            cfg.LABELS = [line.split(",")[1] for line in utils.readLines(cfg.LABELS_FILE)]

        args.lat = -1
        args.lon = -1
        args.locale = "en"

    # Load translated labels
    lfile = os.path.join(
        cfg.TRANSLATED_LABELS_PATH, os.path.basename(cfg.LABELS_FILE).replace(".txt", "_{}.txt".format(args.locale))
    )

    if not args.locale in ["en"] and os.path.isfile(lfile):
        cfg.TRANSLATED_LABELS = utils.readLines(lfile)
    else:
        cfg.TRANSLATED_LABELS = cfg.LABELS

    ### Make sure to comment out appropriately if you are not using args. ###

    # Load species list from location filter or provided list
    cfg.LATITUDE, cfg.LONGITUDE, cfg.WEEK = args.lat, args.lon, args.week
    cfg.LOCATION_FILTER_THRESHOLD = max(0.01, min(0.99, float(args.sf_thresh)))

    if cfg.LATITUDE == -1 and cfg.LONGITUDE == -1:
        if not args.slist:
            cfg.SPECIES_LIST_FILE = None
        else:
            cfg.SPECIES_LIST_FILE = os.path.join(script_dir, args.slist)

            if os.path.isdir(cfg.SPECIES_LIST_FILE):
                cfg.SPECIES_LIST_FILE = os.path.join(cfg.SPECIES_LIST_FILE, "species_list.txt")

        cfg.SPECIES_LIST = utils.readLines(cfg.SPECIES_LIST_FILE)
    else:
        cfg.SPECIES_LIST_FILE = None
        cfg.SPECIES_LIST = species.getSpeciesList(cfg.LATITUDE, cfg.LONGITUDE, cfg.WEEK, cfg.LOCATION_FILTER_THRESHOLD)

    if not cfg.SPECIES_LIST:
        print(f"Species list contains {len(cfg.LABELS)} species")
    else:
        print(f"Species list contains {len(cfg.SPECIES_LIST)} species")

    # Set input and output path
    cfg.INPUT_PATH = args.i
    cfg.OUTPUT_PATH = args.o

    # Parse input files
    if os.path.isdir(cfg.INPUT_PATH):
        cfg.FILE_LIST = utils.collect_audio_files(cfg.INPUT_PATH)
        print(f"Found {len(cfg.FILE_LIST)} files to analyze")
    else:
        cfg.FILE_LIST = [cfg.INPUT_PATH]

    # Set confidence threshold
    cfg.MIN_CONFIDENCE = max(0.01, min(0.99, float(args.min_conf)))

    # Set sensitivity
    cfg.SIGMOID_SENSITIVITY = max(0.5, min(1.0 - (float(args.sensitivity) - 1.0), 1.5))

    # Set overlap
    cfg.SIG_OVERLAP = max(0.0, min(2.9, float(args.overlap)))

    # Set bandpass frequency range
    cfg.BANDPASS_FMIN = max(0, min(cfg.SIG_FMAX, int(args.fmin)))
    cfg.BANDPASS_FMAX = max(cfg.SIG_FMIN, min(cfg.SIG_FMAX, int(args.fmax)))

    # Set result type
    cfg.RESULT_TYPE = args.rtype.lower()

    if not cfg.RESULT_TYPE in ["table", "audacity", "r", "kaleidoscope", "csv"]:
        cfg.RESULT_TYPE = "table"

    # Set output file
    if args.output_file is not None and cfg.RESULT_TYPE == "table":
        cfg.OUTPUT_FILE = args.output_file
    else:
        cfg.OUTPUT_FILE = None

    # Set number of threads
    if os.path.isdir(cfg.INPUT_PATH):
        cfg.CPU_THREADS = max(1, int(args.threads))
        cfg.TFLITE_THREADS = 1
    else:
        cfg.CPU_THREADS = 1
        cfg.TFLITE_THREADS = max(1, int(args.threads))

    # Set batch size
    cfg.BATCH_SIZE = max(1, int(args.batchsize))

    # Add config items to each file list entry.
    # We have to do this for Windows which does not
    # support fork() and thus each process has to
    # have its own config. USE LINUX!
    flist = [(f, cfg.getConfig()) for f in cfg.FILE_LIST]

    # Analyze files
    if cfg.CPU_THREADS < 2 or len(flist) < 2:
        for entry in flist:
            analyzeFile(entry)
    else:
        with Pool(cfg.CPU_THREADS) as p:
            # Map analyzeFile function to each entry in flist
            results = p.map_async(analyzeFile, flist)
            # Wait for all tasks to complete
            results.wait()

    # Combine results?
    if not cfg.OUTPUT_FILE is None:
        print(f"Combining results into {cfg.OUTPUT_FILE}...", end="", flush=True)
        combineResults(cfg.OUTPUT_PATH, cfg.OUTPUT_FILE)
        print("done!", flush=True)

    # A few examples to test
    # python3 analyze.py --i example/ --o example/ --slist example/ --min_conf 0.5 --threads 4
    # python3 analyze.py --i example/soundscape.wav --o example/soundscape.BirdNET.selection.table.txt --slist example/species_list.txt --threads 8
    # python3 analyze.py --i example/ --o example/ --lat 42.5 --lon -76.45 --week 4 --sensitivity 1.0 --rtype table --locale de
