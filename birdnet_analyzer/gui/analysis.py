import concurrent.futures
import os
from pathlib import Path

import gradio as gr

import birdnet_analyzer.analyze as analyze
import birdnet_analyzer.config as cfg
import birdnet_analyzer.gui.utils as gu
import birdnet_analyzer.localization as loc
import birdnet_analyzer.model as model
import birdnet_analyzer.species as species
import birdnet_analyzer.utils as utils

SCRIPT_DIR = os.path.abspath(os.path.dirname(__file__))
ORIGINAL_LABELS_FILE = str(Path(SCRIPT_DIR).parent / cfg.LABELS_FILE)


def analyzeFile_wrapper(entry):
    return (entry[0], analyze.analyzeFile(entry))


def runAnalysis(
    input_path: str,
    output_path: str | None,
    confidence: float,
    sensitivity: float,
    overlap: float,
    fmin: int,
    fmax: int,
    species_list_choice: str,
    species_list_file,
    lat: float,
    lon: float,
    week: int,
    use_yearlong: bool,
    sf_thresh: float,
    custom_classifier_file,
    output_types: str,
    combine_tables: bool,
    locale: str,
    batch_size: int,
    threads: int,
    input_dir: str,
    skip_existing: bool,
    progress: gr.Progress | None,
):
    """Starts the analysis.

    Args:
        input_path: Either a file or directory.
        output_path: The output path for the result, if None the input_path is used
        confidence: The selected minimum confidence.
        sensitivity: The selected sensitivity.
        overlap: The selected segment overlap.
        fmin: The selected minimum bandpass frequency.
        fmax: The selected maximum bandpass frequency.
        species_list_choice: The choice for the species list.
        species_list_file: The selected custom species list file.
        lat: The selected latitude.
        lon: The selected longitude.
        week: The selected week of the year.
        use_yearlong: Use yearlong instead of week.
        sf_thresh: The threshold for the predicted species list.
        custom_classifier_file: Custom classifier to be used.
        output_type: The type of result to be generated.
        output_filename: The filename for the combined output.
        locale: The translation to be used.
        batch_size: The number of samples in a batch.
        threads: The number of threads to be used.
        input_dir: The input directory.
        progress: The gradio progress bar.
    """
    if progress is not None:
        progress(0, desc=f"{loc.localize('progress-preparing')} ...")

    locale = locale.lower()
    # Load eBird codes, labels
    cfg.CODES = analyze.loadCodes()
    cfg.LABELS = utils.readLines(ORIGINAL_LABELS_FILE)
    cfg.LATITUDE, cfg.LONGITUDE, cfg.WEEK = lat, lon, -1 if use_yearlong else week
    cfg.LOCATION_FILTER_THRESHOLD = sf_thresh
    cfg.SKIP_EXISTING_RESULTS = skip_existing

    if species_list_choice == gu._CUSTOM_SPECIES:
        if not species_list_file or not species_list_file.name:
            cfg.SPECIES_LIST_FILE = None
        else:
            cfg.SPECIES_LIST_FILE = species_list_file.name

            if os.path.isdir(cfg.SPECIES_LIST_FILE):
                cfg.SPECIES_LIST_FILE = os.path.join(cfg.SPECIES_LIST_FILE, "species_list.txt")

        cfg.SPECIES_LIST = utils.readLines(cfg.SPECIES_LIST_FILE)
        cfg.CUSTOM_CLASSIFIER = None
    elif species_list_choice == gu._PREDICT_SPECIES:
        cfg.SPECIES_LIST_FILE = None
        cfg.CUSTOM_CLASSIFIER = None
        cfg.SPECIES_LIST = species.getSpeciesList(cfg.LATITUDE, cfg.LONGITUDE, cfg.WEEK, cfg.LOCATION_FILTER_THRESHOLD)
    elif species_list_choice == gu._CUSTOM_CLASSIFIER:
        if custom_classifier_file is None:
            raise gr.Error(loc.localize("validation-no-custom-classifier-selected"))

        model.resetCustomClassifier()

        # Set custom classifier?
        cfg.CUSTOM_CLASSIFIER = (
            custom_classifier_file  # we treat this as absolute path, so no need to join with dirname
        )
        cfg.LABELS_FILE = custom_classifier_file.replace(".tflite", "_Labels.txt")  # same for labels file
        cfg.LABELS = utils.readLines(cfg.LABELS_FILE)
        cfg.LATITUDE = -1
        cfg.LONGITUDE = -1
        cfg.SPECIES_LIST_FILE = None
        cfg.SPECIES_LIST = []
        locale = "en"
    else:
        cfg.SPECIES_LIST_FILE = None
        cfg.SPECIES_LIST = []
        cfg.CUSTOM_CLASSIFIER = None

    # Load translated labels
    lfile = os.path.join(
        gu.ORIGINAL_TRANSLATED_LABELS_PATH, os.path.basename(cfg.LABELS_FILE).replace(".txt", f"_{locale}.txt")
    )
    if locale not in ["en"] and os.path.isfile(lfile):
        cfg.TRANSLATED_LABELS = utils.readLines(lfile)
    else:
        cfg.TRANSLATED_LABELS = cfg.LABELS

    if len(cfg.SPECIES_LIST) == 0:
        print(f"Species list contains {len(cfg.LABELS)} species")
    else:
        print(f"Species list contains {len(cfg.SPECIES_LIST)} species")

    # Set input and output path
    cfg.INPUT_PATH = input_path

    if input_dir:
        cfg.OUTPUT_PATH = output_path if output_path else input_dir
    else:
        cfg.OUTPUT_PATH = output_path if output_path else os.path.dirname(input_path)

    # Parse input files
    if input_dir:
        cfg.FILE_LIST = utils.collect_audio_files(input_dir)
        cfg.INPUT_PATH = input_dir
    elif os.path.isdir(cfg.INPUT_PATH):
        cfg.FILE_LIST = utils.collect_audio_files(cfg.INPUT_PATH)
    else:
        cfg.FILE_LIST = [cfg.INPUT_PATH]

    gu.validate(cfg.FILE_LIST, loc.localize("validation-no-audio-files-found"))

    # Set confidence threshold
    cfg.MIN_CONFIDENCE = confidence

    # Set sensitivity
    cfg.SIGMOID_SENSITIVITY = max(0.5, min(1.0 - (float(sensitivity) - 1.0), 1.5))

    # Set overlap
    cfg.SIG_OVERLAP = max(0.0, min(2.9, float(overlap)))

    # Set frequency range
    cfg.BANDPASS_FMIN = max(0, min(cfg.SIG_FMAX, int(fmin)))
    cfg.BANDPASS_FMAX = max(cfg.SIG_FMIN, min(cfg.SIG_FMAX, int(fmax)))

    # Set result type
    cfg.RESULT_TYPES = output_types
    cfg.COMBINE_RESULTS = combine_tables

    # Set number of threads
    if input_dir:
        cfg.CPU_THREADS = max(1, int(threads))
        cfg.TFLITE_THREADS = 1
    else:
        cfg.CPU_THREADS = 1
        cfg.TFLITE_THREADS = max(1, int(threads))

    # Set batch size
    cfg.BATCH_SIZE = max(1, int(batch_size))

    flist = []

    for f in cfg.FILE_LIST:
        flist.append((f, cfg.getConfig()))

    result_list = []

    if progress is not None:
        progress(0, desc=f"{loc.localize('progress-starting')} ...")

    # Analyze files
    if cfg.CPU_THREADS < 2:
        for entry in flist:
            result_list.append(analyzeFile_wrapper(entry))
    else:
        with concurrent.futures.ProcessPoolExecutor(max_workers=cfg.CPU_THREADS) as executor:
            futures = (executor.submit(analyzeFile_wrapper, arg) for arg in flist)
            for i, f in enumerate(concurrent.futures.as_completed(futures), start=1):
                if progress is not None:
                    progress((i, len(flist)), total=len(flist), unit="files")
                result = f.result()

                result_list.append(result)

    # Combine results?
    if cfg.COMBINE_RESULTS:
        combine_list = [[r[1] for r in result_list if r[0] == i[0]][0] for i in flist]
        print(f"Combining results, writing to {cfg.OUTPUT_PATH}...", end="", flush=True)
        analyze.combineResults(combine_list)
        print("done!", flush=True)

    return (
        [[os.path.relpath(r[0], input_dir), bool(r[1])] for r in result_list] if input_dir else result_list[0][1]["csv"]
    )
