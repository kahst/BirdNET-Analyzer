import os
from multiprocessing import Pool, freeze_support

import birdnet_analyzer.cli as cli
import birdnet_analyzer.config as cfg
import birdnet_analyzer.utils as utils

# Freeze support for executable
freeze_support()

parser = cli.analyzer_parser()

args = parser.parse_args()


try:
    if os.get_terminal_size().columns >= 64:
        print(cli.ASCII_LOGO, flush=True)
except Exception:
    pass

import birdnet_analyzer.species as species  # noqa: E402
from birdnet_analyzer.analyze.utils import analyze_file, combine_results, load_codes, save_analysis_params  # noqa: E402

# Load eBird codes, labels
cfg.CODES = load_codes()
cfg.LABELS = utils.read_lines(cfg.LABELS_FILE)

cfg.SKIP_EXISTING_RESULTS = args.skip_existing_results

# Set custom classifier?
if args.classifier is not None:
    cfg.CUSTOM_CLASSIFIER = args.classifier  # we treat this as absolute path, so no need to join with dirname

    if args.classifier.endswith(".tflite"):
        cfg.LABELS_FILE = args.classifier.replace(".tflite", "_Labels.txt")  # same for labels file

        if not os.path.isfile(cfg.LABELS_FILE):
            cfg.LABELS_FILE = args.classifier.replace("Model_FP32.tflite", "Labels.txt")

        cfg.LABELS = utils.read_lines(cfg.LABELS_FILE)
    else:
        cfg.APPLY_SIGMOID = False
        cfg.LABELS_FILE = os.path.join(args.classifier, "labels", "label_names.csv")
        cfg.LABELS = [line.split(",")[1] for line in utils.read_lines(cfg.LABELS_FILE)]

    args.lat = -1
    args.lon = -1
    args.locale = "en"

# Load translated labels
lfile = os.path.join(
    cfg.TRANSLATED_LABELS_PATH, os.path.basename(cfg.LABELS_FILE).replace(".txt", "_{}.txt".format(args.locale))
)

if args.locale not in ["en"] and os.path.isfile(lfile):
    cfg.TRANSLATED_LABELS = utils.read_lines(lfile)
else:
    cfg.TRANSLATED_LABELS = cfg.LABELS

### Make sure to comment out appropriately if you are not using args. ###

# Load species list from location filter or provided list
cfg.LATITUDE, cfg.LONGITUDE, cfg.WEEK = args.lat, args.lon, args.week
cfg.LOCATION_FILTER_THRESHOLD = args.sf_thresh

if cfg.LATITUDE == -1 and cfg.LONGITUDE == -1:
    if not args.slist:
        cfg.SPECIES_LIST_FILE = None
    else:
        cfg.SPECIES_LIST_FILE = args.slist

        if os.path.isdir(cfg.SPECIES_LIST_FILE):
            cfg.SPECIES_LIST_FILE = os.path.join(cfg.SPECIES_LIST_FILE, "species_list.txt")

    cfg.SPECIES_LIST = utils.read_lines(cfg.SPECIES_LIST_FILE)
else:
    cfg.SPECIES_LIST_FILE = None
    cfg.SPECIES_LIST = species.get_species_list(cfg.LATITUDE, cfg.LONGITUDE, cfg.WEEK, cfg.LOCATION_FILTER_THRESHOLD)

if not cfg.SPECIES_LIST:
    print(f"Species list contains {len(cfg.LABELS)} species")
else:
    print(f"Species list contains {len(cfg.SPECIES_LIST)} species")

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
    print(f"Found {len(cfg.FILE_LIST)} files to analyze")
else:
    cfg.FILE_LIST = [cfg.INPUT_PATH]

# Set confidence threshold
cfg.MIN_CONFIDENCE = args.min_conf

# Set sensitivity
cfg.SIGMOID_SENSITIVITY = args.sensitivity

# Set overlap
cfg.SIG_OVERLAP = args.overlap

# Set bandpass frequency range
cfg.BANDPASS_FMIN = args.fmin
cfg.BANDPASS_FMAX = args.fmax

# Set audio speed
cfg.AUDIO_SPEED = args.audio_speed

# Set result type
cfg.RESULT_TYPES = args.rtype

# Set output file
cfg.COMBINE_RESULTS = args.combine_results

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
flist = [(f, cfg.get_config()) for f in cfg.FILE_LIST]
result_files = []

# Analyze files
if cfg.CPU_THREADS < 2 or len(flist) < 2:
    for entry in flist:
        result_files.append(analyze_file(entry))
else:
    with Pool(cfg.CPU_THREADS) as p:
        # Map analyzeFile function to each entry in flist
        results = p.map_async(analyze_file, flist)
        # Wait for all tasks to complete
        results.wait()
        result_files = results.get()

# Combine results?
if cfg.COMBINE_RESULTS:
    print(f"Combining results, writing to {cfg.OUTPUT_PATH}...", end="", flush=True)
    combine_results(result_files)
    print("done!", flush=True)

save_analysis_params(os.path.join(cfg.OUTPUT_PATH, cfg.ANALYSIS_PARAMS_FILENAME))
