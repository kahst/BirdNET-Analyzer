"""Module to analyze audio samples."""

import argparse
import datetime
import json
import multiprocessing
import operator
import os
from multiprocessing import Pool, freeze_support

import numpy as np

import birdnet_analyzer.audio as audio
import birdnet_analyzer.config as cfg
import birdnet_analyzer.model as model
import birdnet_analyzer.species as species
import birdnet_analyzer.utils as utils

#                    0       1      2           3             4              5               6                7           8             9           10         11
RAVEN_TABLE_HEADER = "Selection\tView\tChannel\tBegin Time (s)\tEnd Time (s)\tLow Freq (Hz)\tHigh Freq (Hz)\tCommon Name\tSpecies Code\tConfidence\tBegin Path\tFile Offset (s)\n"
RTABLE_HEADER = "filepath,start,end,scientific_name,common_name,confidence,lat,lon,week,overlap,sensitivity,min_conf,species_list,model\n"
KALEIDOSCOPE_HEADER = (
    "INDIR,FOLDER,IN FILE,OFFSET,DURATION,scientific_name,common_name,confidence,lat,lon,week,overlap,sensitivity\n"
)
CSV_HEADER = "Start (s),End (s),Scientific name,Common name,Confidence,File\n"
SCRIPT_DIR = os.path.abspath(os.path.dirname(__file__))
ASCII_LOGO = r"""                        
                          .                                     
                       .-=-                                     
                    .:=++++.                                    
                 ..-======#=:.                                  
                .-%%%#*+=-#+++:..                               
              .-+***======++++++=..                             
                  .=====+==++++++++-.                           
                  .=+++====++++++++++=:.                        
                  .++++++++=======----===:                      
                   =+++++++====-----+++++++-.                   
                   .=++++==========-=++=====+=:.                
                     -++======---:::::-=++++***+:.              
                     ..---::::::::::::::::-=*****+-.            
                       ..--------::::::::::::--+##-.:.          
  ++++=::::::...         ..-------------::::::-::.::.           
           ..::-------:::.-=.:::::+-....   ....:--:..           
                    ..::-======--+::......      .:---:.         
                              ..:--==+++++==-..    .-+==-       
                                   ......::----:      **=--     
                                            ..-=-:.     *+=:=   
                                              ..-====  +++ =+** 
                                                 ========+      
                                                 **=====        
                                               ***+==           
                                              ****+             
"""


def loadCodes():
    """Loads the eBird codes.

    Returns:
        A dictionary containing the eBird codes.
    """
    with open(os.path.join(SCRIPT_DIR, cfg.CODES_FILE), "r") as cfile:
        codes = json.load(cfile)

    return codes


def generate_raven_table(timestamps: list[str], result: dict[str, list], afile_path: str, result_path: str) -> str:
    selection_id = 0
    out_string = RAVEN_TABLE_HEADER

    # Read native sample rate
    high_freq = audio.get_sample_rate(afile_path) / 2

    if high_freq > cfg.SIG_FMAX:
        high_freq = cfg.SIG_FMAX

    high_freq = min(high_freq, cfg.BANDPASS_FMAX)
    low_freq = max(cfg.SIG_FMIN, cfg.BANDPASS_FMIN)

    # Extract valid predictions for every timestamp
    for timestamp in timestamps:
        rstring = ""
        start, end = timestamp.split("-", 1)

        for c in result[timestamp]:
            if c[1] > cfg.MIN_CONFIDENCE and (not cfg.SPECIES_LIST or c[0] in cfg.SPECIES_LIST):
                selection_id += 1
                label = cfg.TRANSLATED_LABELS[cfg.LABELS.index(c[0])]
                code = cfg.CODES[c[0]] if c[0] in cfg.CODES else c[0]
                rstring += f"{selection_id}\tSpectrogram 1\t1\t{start}\t{end}\t{low_freq}\t{high_freq}\t{label.split('_', 1)[-1]}\t{code}\t{c[1]:.4f}\t{afile_path}\t{start}\n"

        # Write result string to file
        out_string += rstring

    # If we don't have any valid predictions, we still need to add a line to the selection table in case we want to combine results
    # TODO: That's a weird way to do it, but it works for now. It would be better to keep track of file durations during the analysis.
    if len(out_string) == len(RAVEN_TABLE_HEADER) and cfg.OUTPUT_PATH is not None:
        selection_id += 1
        out_string += (
            f"{selection_id}\tSpectrogram 1\t1\t0\t3\t{low_freq}\t{high_freq}\tnocall\tnocall\t1.0\t{afile_path}\t0\n"
        )

    utils.save_result_file(result_path, out_string)


def generate_audacity(timestamps: list[str], result: dict[str, list], result_path: str) -> str:
    out_string = ""

    # Audacity timeline labels
    for timestamp in timestamps:
        rstring = ""

        for c in result[timestamp]:
            if c[1] > cfg.MIN_CONFIDENCE and (not cfg.SPECIES_LIST or c[0] in cfg.SPECIES_LIST):
                label = cfg.TRANSLATED_LABELS[cfg.LABELS.index(c[0])]
                ts = timestamp.replace("-", "\t")
                lbl = label.replace("_", ", ")
                rstring += f"{ts}\t{lbl}\t{c[1]:.4f}\n"

        # Write result string to file
        out_string += rstring

    utils.save_result_file(result_path, out_string)


def generate_rtable(timestamps: list[str], result: dict[str, list], afile_path: str, result_path: str) -> str:
    out_string = RTABLE_HEADER

    for timestamp in timestamps:
        rstring = ""
        start, end = timestamp.split("-", 1)

        for c in result[timestamp]:
            if c[1] > cfg.MIN_CONFIDENCE and (not cfg.SPECIES_LIST or c[0] in cfg.SPECIES_LIST):
                label = cfg.TRANSLATED_LABELS[cfg.LABELS.index(c[0])]
                rstring += "{},{},{},{},{},{:.4f},{:.4f},{:.4f},{},{},{},{},{},{}\n".format(
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

    utils.save_result_file(result_path, out_string)


def generate_kaleidoscope(timestamps: list[str], result: dict[str, list], afile_path: str, result_path: str) -> str:
    out_string = KALEIDOSCOPE_HEADER

    folder_path, filename = os.path.split(afile_path)
    parent_folder, folder_name = os.path.split(folder_path)

    for timestamp in timestamps:
        rstring = ""
        start, end = timestamp.split("-", 1)

        for c in result[timestamp]:
            if c[1] > cfg.MIN_CONFIDENCE and (not cfg.SPECIES_LIST or c[0] in cfg.SPECIES_LIST):
                label = cfg.TRANSLATED_LABELS[cfg.LABELS.index(c[0])]
                rstring += "{},{},{},{},{},{},{},{:.4f},{:.4f},{:.4f},{},{},{}\n".format(
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

    utils.save_result_file(result_path, out_string)


def generate_csv(timestamps: list[str], result: dict[str, list], afile_path: str, result_path: str) -> str:
    out_string = CSV_HEADER

    for timestamp in timestamps:
        rstring = ""

        for c in result[timestamp]:
            start, end = timestamp.split("-", 1)

            if c[1] > cfg.MIN_CONFIDENCE and (not cfg.SPECIES_LIST or c[0] in cfg.SPECIES_LIST):
                label = cfg.TRANSLATED_LABELS[cfg.LABELS.index(c[0])]
                rstring += f"{start},{end},{label.split('_', 1)[0]},{label.split('_', 1)[-1]},{c[1]:.4f},{afile_path}\n"

        # Write result string to file
        out_string += rstring

    utils.save_result_file(result_path, out_string)


def saveResultFiles(r: dict[str, list], result_files: dict[str, str], afile_path: str):
    """Saves the results to the hard drive.

    Args:
        r: The dictionary with {segment: scores}.
        path: The path where the result should be saved.
        afile_path: The path to audio file.
    """

    os.makedirs(cfg.OUTPUT_PATH, exist_ok=True)

    # Selection table
    timestamps = getSortedTimestamps(r)

    if "table" in result_files:
        generate_raven_table(timestamps, r, afile_path, result_files["table"])

    if "audacity" in cfg.RESULT_TYPES:
        generate_audacity(timestamps, r, result_files["audacity"])

    if "r" in cfg.RESULT_TYPES:
        generate_rtable(timestamps, r, afile_path, result_files["r"])

    if "kaleidoscope" in cfg.RESULT_TYPES:
        generate_kaleidoscope(timestamps, r, afile_path, result_files["kaleidoscope"])

    if "csv" in cfg.RESULT_TYPES:
        generate_csv(timestamps, r, afile_path, result_files["csv"])


def combine_raven_tables(saved_results: list[str]):
    # Combine all files
    s_id = 1
    time_offset = 0
    audiofiles = []

    with open(os.path.join(cfg.OUTPUT_PATH, cfg.OUTPUT_RAVEN_FILENAME), "w", encoding="utf-8") as f:
        f.write(RAVEN_TABLE_HEADER)

        for rfile in saved_results:
            if not rfile:
                continue
            with open(rfile, "r", encoding="utf-8") as rf:
                try:
                    lines = rf.readlines()

                    # make sure it's a selection table
                    if "Selection" not in lines[0] or "File Offset" not in lines[0]:
                        continue

                    # skip header and add to file
                    f_name = lines[1].split("\t")[10]
                    f_duration = audio.getAudioFileLength(f_name, cfg.SAMPLE_RATE)

                    audiofiles.append(f_name)

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

    listfilesname = cfg.OUTPUT_RAVEN_FILENAME.rsplit(".", 1)[0] + ".list.txt"

    with open(os.path.join(cfg.OUTPUT_PATH, listfilesname), "w", encoding="utf-8") as f:
        f.writelines((f + "\n" for f in audiofiles))


def combine_rtable_files(saved_results: list[str]):
    # Combine all files
    with open(os.path.join(cfg.OUTPUT_PATH, cfg.OUTPUT_RTABLE_FILENAME), "w", encoding="utf-8") as f:
        f.write(RTABLE_HEADER)

        for rfile in saved_results:
            with open(rfile, "r", encoding="utf-8") as rf:
                try:
                    lines = rf.readlines()

                    # make sure it's a selection table
                    if "filepath" not in lines[0] or "model" not in lines[0]:
                        continue

                    # skip header and add to file
                    for line in lines[1:]:
                        f.write(line)

                except Exception as ex:
                    print(f"Error: Cannot combine results from {rfile}.\n", flush=True)
                    utils.writeErrorLog(ex)


def combine_kaleidoscope_files(saved_results: list[str]):
    # Combine all files
    with open(os.path.join(cfg.OUTPUT_PATH, cfg.OUTPUT_KALEIDOSCOPE_FILENAME), "w", encoding="utf-8") as f:
        f.write(KALEIDOSCOPE_HEADER)

        for rfile in saved_results:
            with open(rfile, "r", encoding="utf-8") as rf:
                try:
                    lines = rf.readlines()

                    # make sure it's a selection table
                    if "INDIR" not in lines[0] or "sensitivity" not in lines[0]:
                        continue

                    # skip header and add to file
                    for line in lines[1:]:
                        f.write(line)

                except Exception as ex:
                    print(f"Error: Cannot combine results from {rfile}.\n", flush=True)
                    utils.writeErrorLog(ex)


def combine_csv_files(saved_results: list[str]):
    # Combine all files
    with open(os.path.join(cfg.OUTPUT_PATH, cfg.OUTPUT_CSV_FILENAME), "w", encoding="utf-8") as f:
        f.write(CSV_HEADER)

        for rfile in saved_results:
            with open(rfile, "r", encoding="utf-8") as rf:
                try:
                    lines = rf.readlines()

                    # make sure it's a selection table
                    if "Start (s)" not in lines[0] or "Confidence" not in lines[0]:
                        continue

                    # skip header and add to file
                    for line in lines[1:]:
                        f.write(line)

                except Exception as ex:
                    print(f"Error: Cannot combine results from {rfile}.\n", flush=True)
                    utils.writeErrorLog(ex)


def combineResults(saved_results: list[dict[str, str]]):
    if "table" in cfg.RESULT_TYPES:
        combine_raven_tables([f["table"] for f in saved_results if f])

    if "r" in cfg.RESULT_TYPES:
        combine_rtable_files([f["r"] for f in saved_results if f])

    if "kaleidoscope" in cfg.RESULT_TYPES:
        combine_kaleidoscope_files([f["kaleidoscope"] for f in saved_results if f])

    if "csv" in cfg.RESULT_TYPES:
        combine_csv_files([f["csv"] for f in saved_results if f])


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


def get_result_file_names(fpath: str):
    result_names = {}

    rpath = fpath.replace(cfg.INPUT_PATH, "")

    if rpath:
        rpath = rpath[1:] if rpath[0] in ["/", "\\"] else rpath
    else:
        rpath = os.path.basename(fpath)

    file_shorthand = rpath.rsplit(".", 1)[0]

    if "table" in cfg.RESULT_TYPES:
        result_names["table"] = os.path.join(cfg.OUTPUT_PATH, file_shorthand + ".BirdNET.selection.table.txt")
    if "audacity" in cfg.RESULT_TYPES:
        result_names["audacity"] = os.path.join(cfg.OUTPUT_PATH, file_shorthand + ".BirdNET.results.txt")
    if "r" in cfg.RESULT_TYPES:
        result_names["r"] = os.path.join(cfg.OUTPUT_PATH, file_shorthand + ".BirdNET.results.r.csv")
    if "kaleidoscope" in cfg.RESULT_TYPES:
        result_names["kaleidoscope"] = os.path.join(
            cfg.OUTPUT_PATH, file_shorthand + ".BirdNET.results.kaleidoscope.csv"
        )
    if "csv" in cfg.RESULT_TYPES:
        result_names["csv"] = os.path.join(cfg.OUTPUT_PATH, file_shorthand + ".BirdNET.results.csv")

    return result_names


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

    result_file_names = get_result_file_names(fpath)

    if cfg.SKIP_EXISTING_RESULTS:
        if all(os.path.exists(f) for f in result_file_names.values()):
            print(f"Skipping {fpath} as it has already been analyzed", flush=True)
            return None  # or return path to combine later? TODO

    # Start time
    start_time = datetime.datetime.now()
    offset = 0
    duration = cfg.FILE_SPLITTING_DURATION
    start, end = 0, cfg.SIG_LENGTH
    results = {}

    # Status
    print(f"Analyzing {fpath}", flush=True)

    try:
        fileLengthSeconds = int(audio.getAudioFileLength(fpath, cfg.SAMPLE_RATE))
    except Exception as ex:
        # Write error log
        print(f"Error: Cannot analyze audio file {fpath}. File corrupt?\n", flush=True)
        utils.writeErrorLog(ex)

        return None

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
                    p_labels = zip(cfg.LABELS, pred, strict=True)

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

        return None

    # Save as selection table
    try:
        saveResultFiles(results, result_file_names, fpath)

    except Exception as ex:
        # Write error log
        print(f"Error: Cannot save result for {fpath}.\n", flush=True)
        utils.writeErrorLog(ex)

        return None

    delta_time = (datetime.datetime.now() - start_time).total_seconds()
    print(f"Finished {fpath} in {delta_time:.2f} seconds", flush=True)

    return result_file_names


if __name__ == "__main__":
    # Freeze support for executable
    freeze_support()

    # Parse arguments
    parser = argparse.ArgumentParser(
        description=ASCII_LOGO,
        formatter_class=argparse.RawDescriptionHelpFormatter,
        usage="python -m birdnet_analyzer.analyze [options]",
    )
    parser.add_argument("--i", default=os.path.join(SCRIPT_DIR, "example/"), help="Path to input file or folder.")
    parser.add_argument("--o", default=os.path.join(SCRIPT_DIR, "example/"), help="Path to output folder.")
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

    class UniqueSetAction(argparse.Action):
        def __call__(self, parser, args, values, option_string=None):
            setattr(args, self.dest, {v.lower() for v in values})

    parser.add_argument(
        "--rtype",
        default={"table"},
        choices=["table", "audacity", "r", "kaleidoscope", "csv"],
        nargs="+",
        help="Specifies output format. Values in ['table', 'audacity', 'r',  'kaleidoscope', 'csv']. Defaults to 'table' (Raven selection table).",
        action=UniqueSetAction,
    )
    parser.add_argument(
        "--combine_results",
        help="Also outputs a combined file for all the selected result types. If not set combined tables will be generated. Defaults to False.",
        action=argparse.BooleanOptionalAction,
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

    try:
        if os.get_terminal_size().columns >= 64:
            print(ASCII_LOGO, flush=True)
    except Exception:
        pass

    # Set paths relative to script path (requested in #3)
    cfg.MODEL_PATH = os.path.join(SCRIPT_DIR, cfg.MODEL_PATH)
    cfg.LABELS_FILE = os.path.join(SCRIPT_DIR, cfg.LABELS_FILE)
    cfg.TRANSLATED_LABELS_PATH = os.path.join(SCRIPT_DIR, cfg.TRANSLATED_LABELS_PATH)
    cfg.MDATA_MODEL_PATH = os.path.join(SCRIPT_DIR, cfg.MDATA_MODEL_PATH)
    cfg.CODES_FILE = os.path.join(SCRIPT_DIR, cfg.CODES_FILE)
    cfg.ERROR_LOG_FILE = os.path.join(SCRIPT_DIR, cfg.ERROR_LOG_FILE)

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

    if args.locale not in ["en"] and os.path.isfile(lfile):
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
            cfg.SPECIES_LIST_FILE = os.path.join(SCRIPT_DIR, args.slist)

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
    cfg.RESULT_TYPES = args.rtype

    # Set output file
    cfg.COMBINE_RESULTS = args.combine_results

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
    result_files = []

    # Analyze files
    if cfg.CPU_THREADS < 2 or len(flist) < 2:
        for entry in flist:
            result_files.append(analyzeFile(entry))
    else:
        with Pool(cfg.CPU_THREADS) as p:
            # Map analyzeFile function to each entry in flist
            results = p.map_async(analyzeFile, flist)
            # Wait for all tasks to complete
            results.wait()
            result_files = results.get()

    # Combine results?
    if cfg.COMBINE_RESULTS:
        print(f"Combining results, writing to {cfg.OUTPUT_PATH}...", end="", flush=True)
        combineResults(result_files)
        print("done!", flush=True)

    # A few examples to test
    # python3 analyze.py --i example/ --o example/ --slist example/ --min_conf 0.5 --threads 4
    # python3 analyze.py --i example/soundscape.wav --o example/soundscape.BirdNET.selection.table.txt --slist example/species_list.txt --threads 8
    # python3 analyze.py --i example/ --o example/ --lat 42.5 --lon -76.45 --week 4 --sensitivity 1.0 --rtype table --locale de
