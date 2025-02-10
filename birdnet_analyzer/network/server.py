import os
from multiprocessing import freeze_support
import shutil
import tempfile

import birdnet_analyzer.config as cfg
import birdnet_analyzer.cli as cli
import birdnet_analyzer.utils as utils

# Freeze support for executable
freeze_support()

# Parse arguments
parser = cli.server_parser()

args = parser.parse_args()

import bottle  # noqa: E402

import birdnet_analyzer.analyze.utils as analyze  # noqa: E402

# Load eBird codes, labels
cfg.CODES = analyze.load_codes()
cfg.LABELS = utils.read_lines(cfg.LABELS_FILE)

# Load translated labels
lfile = os.path.join(
    cfg.TRANSLATED_LABELS_PATH, os.path.basename(cfg.LABELS_FILE).replace(".txt", "_{}.txt".format(args.locale))
)

if args.locale not in ["en"] and os.path.isfile(lfile):
    cfg.TRANSLATED_LABELS = utils.read_lines(lfile)
else:
    cfg.TRANSLATED_LABELS = cfg.LABELS

# Set storage file path
cfg.FILE_STORAGE_PATH = args.spath

# Set min_conf to 0.0, because we want all results
cfg.MIN_CONFIDENCE = 0.0

# Set path for temporary result file
cfg.OUTPUT_PATH = tempfile.mkdtemp()

# Set result types
cfg.RESULT_TYPES = ["audacity"]

# Set number of TFLite threads
cfg.TFLITE_THREADS = args.threads

# Run server
print(f"UP AND RUNNING! LISTENING ON {args.host}:{args.port}", flush=True)

try:
    bottle.run(host=args.host, port=args.port, quiet=True)
finally:
    shutil.rmtree(cfg.OUTPUT_PATH)
