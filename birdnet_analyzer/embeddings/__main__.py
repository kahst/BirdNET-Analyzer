import os
from multiprocessing import Pool

import birdnet_analyzer.config as cfg
import birdnet_analyzer.utils as utils
import birdnet_analyzer.cli as cli

parser = cli.embeddings_parser()

args = parser.parse_args()

from birdnet_analyzer.embeddings.utils import analyze_file  # noqa: E402

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
flist = [(f, cfg.get_config()) for f in cfg.FILE_LIST]

# Analyze files
if cfg.CPU_THREADS < 2:
    for entry in flist:
        analyze_file(entry)
else:
    with Pool(cfg.CPU_THREADS) as p:
        p.map(analyze_file, flist)
