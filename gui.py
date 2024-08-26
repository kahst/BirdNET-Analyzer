from contextlib import suppress
import concurrent.futures
import os
import sys
from pathlib import Path
from functools import partial

import config as cfg


if getattr(sys, "frozen", False) and hasattr(sys, "_MEIPASS"):
    # divert stdout & stderr to logs.txt file since we have no console when deployed
    userdir = Path.home()

    if sys.platform == "win32":
        userdir /= "AppData/Roaming"
    elif sys.platform == "linux":
        userdir /= ".local/share"
    elif sys.platform == "darwin":
        userdir /= "Library/Application Support"

    appdir = userdir / "BirdNET-Analyzer-GUI"

    appdir.mkdir(parents=True, exist_ok=True)

    sys.stderr = sys.stdout = open(str(appdir / "logs.txt"), "w")
    cfg.ERROR_LOG_FILE = str(appdir / cfg.ERROR_LOG_FILE)
    FROZEN = True
else:
    FROZEN = False


import multiprocessing

import gradio as gr
import librosa
import webview

import analyze
import segments
import species
import utils
from train import trainModel
import localization as loc

loc.load_local_state()

SCRIPT_DIR = os.path.abspath(os.path.dirname(__file__))

_WINDOW: webview.Window
OUTPUT_TYPE_MAP = {
    "Raven selection table": "table",
    "Audacity": "audacity",
    "R": "r",
    "CSV": "csv",
    "Kaleidoscope": "kaleidoscope",
}
ORIGINAL_LABELS_FILE = cfg.LABELS_FILE
ORIGINAL_TRANSLATED_LABELS_PATH = cfg.TRANSLATED_LABELS_PATH
POSITIVE_LABEL_DIR = "Positive"
NEGATIVE_LABEL_DIR = "Negative"


def analyzeFile_wrapper(entry):
    return (entry[0], analyze.analyzeFile(entry))


def extractSegments_wrapper(entry):
    return (entry[0][0], segments.extractSegments(entry))


def validate(value, msg):
    """Checks if the value ist not falsy.

    If the value is falsy, an error will be raised.

    Args:
        value: Value to be tested.
        msg: Message in case of an error.
    """
    if not value:
        raise gr.Error(msg)


def run_species_list(out_path, filename, lat, lon, week, use_yearlong, sf_thresh, sortby):
    validate(out_path, loc.localize("validation-no-directory-selected"))

    species.run(
        os.path.join(out_path, filename if filename else "species_list.txt"),
        lat,
        lon,
        -1 if use_yearlong else week,
        sf_thresh,
        sortby,
    )

    gr.Info(f"{loc.localize('species-tab-finish-info')} {cfg.OUTPUT_PATH}")


def runSingleFileAnalysis(
    input_path,
    confidence,
    sensitivity,
    overlap,
    fmin,
    fmax,
    species_list_choice,
    species_list_file,
    lat,
    lon,
    week,
    use_yearlong,
    sf_thresh,
    custom_classifier_file,
    locale,
):
    validate(input_path, loc.localize("validation-no-file-selected"))

    return runAnalysis(
        input_path,
        None,
        confidence,
        sensitivity,
        overlap,
        fmin,
        fmax,
        species_list_choice,
        species_list_file,
        lat,
        lon,
        week,
        use_yearlong,
        sf_thresh,
        custom_classifier_file,
        "csv",
        None,
        "en" if not locale else locale,
        1,
        4,
        None,
        skip_existing=False,
        progress=None,
    )


def runBatchAnalysis(
    output_path,
    confidence,
    sensitivity,
    overlap,
    fmin,
    fmax,
    species_list_choice,
    species_list_file,
    lat,
    lon,
    week,
    use_yearlong,
    sf_thresh,
    custom_classifier_file,
    output_type,
    combine_tables,
    locale,
    batch_size,
    threads,
    input_dir,
    skip_existing,
    progress=gr.Progress(),
):
    validate(input_dir, loc.localize("validation-no-directory-selected"))
    batch_size = int(batch_size)
    threads = int(threads)

    if species_list_choice == _CUSTOM_SPECIES:
        validate(species_list_file, loc.localize("validation-no-species-list-selected"))

    return runAnalysis(
        None,
        output_path,
        confidence,
        sensitivity,
        overlap,
        fmin,
        fmax,
        species_list_choice,
        species_list_file,
        lat,
        lon,
        week,
        use_yearlong,
        sf_thresh,
        custom_classifier_file,
        output_type,
        combine_tables,
        "en" if not locale else locale,
        batch_size if batch_size and batch_size > 0 else 1,
        threads if threads and threads > 0 else 4,
        input_dir,
        skip_existing,
        progress,
    )


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
    cfg.LABELS = utils.readLines(os.path.join(SCRIPT_DIR, ORIGINAL_LABELS_FILE))
    cfg.LATITUDE, cfg.LONGITUDE, cfg.WEEK = lat, lon, -1 if use_yearlong else week
    cfg.LOCATION_FILTER_THRESHOLD = sf_thresh
    cfg.SKIP_EXISTING_RESULTS = skip_existing

    if species_list_choice == _CUSTOM_SPECIES:
        if not species_list_file or not species_list_file.name:
            cfg.SPECIES_LIST_FILE = None
        else:
            cfg.SPECIES_LIST_FILE = species_list_file.name

            if os.path.isdir(cfg.SPECIES_LIST_FILE):
                cfg.SPECIES_LIST_FILE = os.path.join(cfg.SPECIES_LIST_FILE, "species_list.txt")

        cfg.SPECIES_LIST = utils.readLines(cfg.SPECIES_LIST_FILE)
        cfg.CUSTOM_CLASSIFIER = None
    elif species_list_choice == _PREDICT_SPECIES:
        cfg.SPECIES_LIST_FILE = None
        cfg.CUSTOM_CLASSIFIER = None
        cfg.SPECIES_LIST = species.getSpeciesList(cfg.LATITUDE, cfg.LONGITUDE, cfg.WEEK, cfg.LOCATION_FILTER_THRESHOLD)
    elif species_list_choice == _CUSTOM_CLASSIFIER:
        if custom_classifier_file is None:
            raise gr.Error(loc.localize("validation-no-custom-classifier-selected"))

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
        SCRIPT_DIR, cfg.TRANSLATED_LABELS_PATH, os.path.basename(cfg.LABELS_FILE).replace(".txt", f"_{locale}.txt")
    )
    if not locale in ["en"] and os.path.isfile(lfile):
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

    validate(cfg.FILE_LIST, loc.localize("validation-no-audio-files-found"))

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
        print(f"Combining results, writing to {cfg.OUTPUT_PATH}...", end="", flush=True)
        analyze.combineResults([i[1] for i in result_list])
        print("done!", flush=True)

    return (
        [[os.path.relpath(r[0], input_dir), bool(r[1])] for r in result_list] if input_dir else result_list[0][1]["csv"]
    )


_CUSTOM_SPECIES = loc.localize("species-list-radio-option-custom-list")
_PREDICT_SPECIES = loc.localize("species-list-radio-option-predict-list")
_CUSTOM_CLASSIFIER = loc.localize("species-list-radio-option-custom-classifier")
_ALL_SPECIES = loc.localize("species-list-radio-option-all")


def show_species_choice(choice: str):
    """Sets the visibility of the species list choices.

    Args:
        choice: The label of the currently active choice.

    Returns:
        A list of [
            Row update,
            File update,
            Column update,
            Column update,
        ]
    """
    if choice == _CUSTOM_SPECIES:
        return [
            gr.Row(visible=False),
            gr.File(visible=True),
            gr.Column(visible=False),
            gr.Column(visible=False),
        ]
    elif choice == _PREDICT_SPECIES:
        return [
            gr.Row(visible=True),
            gr.File(visible=False),
            gr.Column(visible=False),
            gr.Column(visible=False),
        ]
    elif choice == _CUSTOM_CLASSIFIER:
        return [
            gr.Row(visible=False),
            gr.File(visible=False),
            gr.Column(visible=True),
            gr.Column(visible=False),
        ]

    return [
        gr.Row(visible=False),
        gr.File(visible=False),
        gr.Column(visible=False),
        gr.Column(visible=True),
    ]


def select_subdirectories(state_key=None):
    """Creates a directory selection dialog.

    Returns:
        A tuples of (directory, list of subdirectories) or (None, None) if the dialog was canceled.
    """

    initial_dir = loc.get_state(state_key, "") if state_key else ""
    dir_name = _WINDOW.create_file_dialog(webview.FOLDER_DIALOG, directory=initial_dir)

    if dir_name:
        if state_key:
            loc.set_state(state_key, dir_name[0])

        subdirs = utils.list_subdirectories(dir_name[0])
        labels = []

        for folder in subdirs:
            labels_in_folder = folder.split(",")

            for label in labels_in_folder:
                if not label in labels:
                    labels.append(label)

        return dir_name[0], [[label] for label in sorted(labels)]

    return None, None


def select_file(filetypes=(), state_key=None):
    """Creates a file selection dialog.

    Args:
        filetypes: List of filetypes to be filtered in the dialog.

    Returns:
        The selected file or None of the dialog was canceled.
    """
    initial_selection = loc.get_state(state_key, "") if state_key else ""
    files = _WINDOW.create_file_dialog(webview.OPEN_DIALOG, file_types=filetypes, directory=initial_selection)

    if files:
        if state_key:
            loc.set_state(state_key, files[0])

        return files[0]

    return None


def format_seconds(secs: float):
    """Formats a number of seconds into a string.

    Formats the seconds into the format "h:mm:ss.ms"

    Args:
        secs: Number of seconds.

    Returns:
        A string with the formatted seconds.
    """
    hours, secs = divmod(secs, 3600)
    minutes, secs = divmod(secs, 60)

    return f"{hours:2.0f}:{minutes:02.0f}:{secs:06.3f}"


def select_directory(collect_files=True, max_files=None, state_key=None):
    """Shows a directory selection system dialog.

    Uses the pywebview to create a system dialog.

    Args:
        collect_files: If True, also lists a files inside the directory.

    Returns:
        If collect_files==True, returns (directory path, list of (relative file path, audio length))
        else just the directory path.
        All values will be None of the dialog is cancelled.
    """
    initial_dir = loc.get_state(state_key, "") if state_key else ""
    dir_name = _WINDOW.create_file_dialog(webview.FOLDER_DIALOG, directory=initial_dir)

    if dir_name and state_key:
        loc.set_state(state_key, dir_name[0])

    if collect_files:
        if not dir_name:
            return None, None

        files = utils.collect_audio_files(dir_name[0], max_files=max_files)

        return dir_name[0], [
            [os.path.relpath(file, dir_name[0]), format_seconds(librosa.get_duration(filename=file))] for file in files
        ]

    return dir_name[0] if dir_name else None


def start_training(
    data_dir,
    crop_mode,
    crop_overlap,
    fmin,
    fmax,
    output_dir,
    classifier_name,
    model_save_mode,
    cache_mode,
    cache_file,
    cache_file_name,
    autotune,
    autotune_trials,
    autotune_executions_per_trials,
    epochs,
    batch_size,
    learning_rate,
    hidden_units,
    use_mixup,
    upsampling_ratio,
    upsampling_mode,
    model_format,
    progress=gr.Progress(),
):
    """Starts the training of a custom classifier.

    Args:
        data_dir: Directory containing the training data.
        output_dir: Directory for the new classifier.
        classifier_name: File name of the classifier.
        epochs: Number of epochs to train for.
        batch_size: Number of samples in one batch.
        learning_rate: Learning rate for training.
        hidden_units: If > 0 the classifier contains a further hidden layer.
        progress: The gradio progress bar.

    Returns:
        Returns a matplotlib.pyplot figure.
    """
    validate(data_dir, loc.localize("validation-no-training-data-selected"))
    validate(output_dir, loc.localize("validation-no-directory-for-classifier-selected"))
    validate(classifier_name, loc.localize("validation-no-valid-classifier-name"))

    if not epochs or epochs < 0:
        raise gr.Error(loc.localize("validation-no-valid-epoch-number"))

    if not batch_size or batch_size < 0:
        raise gr.Error(loc.localize("validation-no-valid-batch-size"))

    if not learning_rate or learning_rate < 0:
        raise gr.Error(loc.localize("validation-no-valid-learning-rate"))

    if fmin < cfg.SIG_FMIN or fmax > cfg.SIG_FMAX or fmin > fmax:
        raise gr.Error(f"{loc.localize('validation-no-valid-frequency')} [{cfg.SIG_FMIN}, {cfg.SIG_FMAX}]")

    if not hidden_units or hidden_units < 0:
        hidden_units = 0

    if progress is not None:
        progress((0, epochs), desc=loc.localize("progress-build-classifier"), unit="epochs")

    cfg.TRAIN_DATA_PATH = data_dir
    cfg.SAMPLE_CROP_MODE = crop_mode
    cfg.SIG_OVERLAP = max(0.0, min(2.9, float(crop_overlap)))
    cfg.CUSTOM_CLASSIFIER = str(Path(output_dir) / classifier_name)
    cfg.TRAIN_EPOCHS = int(epochs)
    cfg.TRAIN_BATCH_SIZE = int(batch_size)
    cfg.TRAIN_LEARNING_RATE = learning_rate
    cfg.TRAIN_HIDDEN_UNITS = int(hidden_units)
    cfg.TRAIN_WITH_MIXUP = use_mixup
    cfg.UPSAMPLING_RATIO = min(max(0, upsampling_ratio), 1)
    cfg.UPSAMPLING_MODE = upsampling_mode
    cfg.TRAINED_MODEL_OUTPUT_FORMAT = model_format

    cfg.BANDPASS_FMIN = max(0, min(cfg.SIG_FMAX, int(fmin)))
    cfg.BANDPASS_FMAX = max(cfg.SIG_FMIN, min(cfg.SIG_FMAX, int(fmax)))

    cfg.TRAINED_MODEL_SAVE_MODE = model_save_mode
    cfg.TRAIN_CACHE_MODE = cache_mode
    cfg.TRAIN_CACHE_FILE = os.path.join(cache_file, cache_file_name) if cache_mode == "save" else cache_file
    cfg.TFLITE_THREADS = 1
    cfg.CPU_THREADS = max(1, multiprocessing.cpu_count() - 1)  # let's use everything we have (well, almost)

    cfg.AUTOTUNE = autotune
    cfg.AUTOTUNE_TRIALS = autotune_trials
    cfg.AUTOTUNE_EXECUTIONS_PER_TRIAL = int(autotune_executions_per_trials)

    def dataLoadProgression(num_files, num_total_files, label):
        if progress is not None:
            progress(
                (num_files, num_total_files),
                total=num_total_files,
                unit="files",
                desc=f"{loc.localize('progress-loading-data')} '{label}'",
            )

    def epochProgression(epoch, logs=None):
        if progress is not None:
            if epoch + 1 == epochs:
                progress(
                    (epoch + 1, epochs),
                    total=epochs,
                    unit="epochs",
                    desc=f"{loc.localize('progress-saving')} {cfg.CUSTOM_CLASSIFIER}",
                )
            else:
                progress((epoch + 1, epochs), total=epochs, unit="epochs", desc=loc.localize("progress-training"))

    def trialProgression(trial):
        if progress is not None:
            progress(
                (trial, autotune_trials), total=autotune_trials, unit="trials", desc=loc.localize("progress-autotune")
            )

    try:
        history = trainModel(
            on_epoch_end=epochProgression,
            on_trial_result=trialProgression,
            on_data_load_end=dataLoadProgression,
            autotune_directory=appdir if FROZEN else "autotune",
        )
    except Exception as e:
        if e.args and len(e.args) > 1:
            raise gr.Error(loc.localize(e.args[1]))
        else:
            raise gr.Error(f"{e}")

    if len(history.epoch) < epochs:
        gr.Info(loc.localize("training-tab-early-stoppage-msg"))

    auprc = history.history["val_AUPRC"]
    auroc = history.history["val_AUROC"]

    import matplotlib.pyplot as plt

    fig = plt.figure()
    plt.plot(auprc, label="AUPRC")
    plt.plot(auroc, label="AUROC")
    plt.legend()
    plt.xlabel("Epoch")

    return fig


def extract_segments(audio_dir, result_dir, output_dir, min_conf, num_seq, seq_length, threads, progress=gr.Progress()):
    validate(audio_dir, loc.localize("validation-no-audio-directory-selected"))

    if not result_dir:
        result_dir = audio_dir

    if not output_dir:
        output_dir = audio_dir

    if progress is not None:
        progress(0, desc=f"{loc.localize('progress-search')} ...")

    # Parse audio and result folders
    cfg.FILE_LIST = segments.parseFolders(audio_dir, result_dir)

    # Set output folder
    cfg.OUTPUT_PATH = output_dir

    # Set number of threads
    cfg.CPU_THREADS = int(threads)

    # Set confidence threshold
    cfg.MIN_CONFIDENCE = max(0.01, min(0.99, min_conf))

    # Parse file list and make list of segments
    cfg.FILE_LIST = segments.parseFiles(cfg.FILE_LIST, max(1, int(num_seq)))

    # Add config items to each file list entry.
    # We have to do this for Windows which does not
    # support fork() and thus each process has to
    # have its own config. USE LINUX!
    flist = [(entry, max(cfg.SIG_LENGTH, float(seq_length)), cfg.getConfig()) for entry in cfg.FILE_LIST]

    result_list = []

    # Extract segments
    if cfg.CPU_THREADS < 2:
        for i, entry in enumerate(flist):
            result = extractSegments_wrapper(entry)
            result_list.append(result)

            if progress is not None:
                progress((i, len(flist)), total=len(flist), unit="files")
    else:
        with concurrent.futures.ProcessPoolExecutor(max_workers=cfg.CPU_THREADS) as executor:
            futures = (executor.submit(extractSegments_wrapper, arg) for arg in flist)
            for i, f in enumerate(concurrent.futures.as_completed(futures), start=1):
                if progress is not None:
                    progress((i, len(flist)), total=len(flist), unit="files")
                result = f.result()

                result_list.append(result)

    return [[os.path.relpath(r[0], audio_dir), r[1]] for r in result_list]


def sample_sliders(opened=True):
    """Creates the gradio accordion for the inference settings.

    Args:
        opened: If True the accordion is open on init.

    Returns:
        A tuple with the created elements:
        (Slider (min confidence), Slider (sensitivity), Slider (overlap))
    """
    with gr.Accordion(loc.localize("inference-settings-accordion-label"), open=opened):
        with gr.Row():
            confidence_slider = gr.Slider(
                minimum=0,
                maximum=1,
                value=0.5,
                step=0.01,
                label=loc.localize("inference-settings-confidence-slider-label"),
                info=loc.localize("inference-settings-confidence-slider-info"),
            )
            sensitivity_slider = gr.Slider(
                minimum=0.5,
                maximum=1.5,
                value=1,
                step=0.01,
                label=loc.localize("inference-settings-sensitivity-slider-label"),
                info=loc.localize("inference-settings-sensitivity-slider-info"),
            )
            overlap_slider = gr.Slider(
                minimum=0,
                maximum=2.99,
                value=0,
                step=0.01,
                label=loc.localize("inference-settings-overlap-slider-label"),
                info=loc.localize("inference-settings-overlap-slider-info"),
            )

        with gr.Row():
            fmin_number = gr.Number(
                cfg.SIG_FMIN,
                minimum=0,
                label=loc.localize("inference-settings-fmin-number-label"),
                info=loc.localize("inference-settings-fmin-number-info"),
            )

            fmax_number = gr.Number(
                cfg.SIG_FMAX,
                minimum=0,
                label=loc.localize("inference-settings-fmax-number-label"),
                info=loc.localize("inference-settings-fmax-number-info"),
            )

        return confidence_slider, sensitivity_slider, overlap_slider, fmin_number, fmax_number


def locale():
    """Creates the gradio elements for locale selection

    Reads the translated labels inside the checkpoints directory.

    Returns:
        The dropdown element.
    """
    label_files = os.listdir(os.path.join(SCRIPT_DIR, ORIGINAL_TRANSLATED_LABELS_PATH))
    options = ["EN"] + [label_file.rsplit("_", 1)[-1].split(".")[0].upper() for label_file in label_files]

    return gr.Dropdown(
        options,
        value="EN",
        label=loc.localize("analyze-locale-dropdown-label"),
        info=loc.localize("analyze-locale-dropdown-info"),
    )


def species_list_coordinates():
    lat_number = gr.Slider(
        minimum=-90,
        maximum=90,
        value=0,
        step=1,
        label=loc.localize("species-list-coordinates-lat-number-label"),
        info=loc.localize("species-list-coordinates-lat-number-info"),
    )
    lon_number = gr.Slider(
        minimum=-180,
        maximum=180,
        value=0,
        step=1,
        label=loc.localize("species-list-coordinates-lon-number-label"),
        info=loc.localize("species-list-coordinates-lon-number-info"),
    )
    with gr.Row():
        yearlong_checkbox = gr.Checkbox(True, label=loc.localize("species-list-coordinates-yearlong-checkbox-label"))
        week_number = gr.Slider(
            minimum=1,
            maximum=48,
            value=1,
            step=1,
            interactive=False,
            label=loc.localize("species-list-coordinates-week-slider-label"),
            info=loc.localize("species-list-coordinates-week-slider-info"),
        )

        def onChange(use_yearlong):
            return gr.Slider(interactive=(not use_yearlong))

        yearlong_checkbox.change(onChange, inputs=yearlong_checkbox, outputs=week_number, show_progress=False)
    sf_thresh_number = gr.Slider(
        minimum=0.01,
        maximum=0.99,
        value=0.03,
        step=0.01,
        label=loc.localize("species-list-coordinates-threshold-slider-label"),
        info=loc.localize("species-list-coordinates-threshold-slider-info"),
    )

    return lat_number, lon_number, week_number, sf_thresh_number, yearlong_checkbox


def species_lists(opened=True):
    """Creates the gradio accordion for species selection.

    Args:
        opened: If True the accordion is open on init.

    Returns:
        A tuple with the created elements:
        (Radio (choice), File (custom species list), Slider (lat), Slider (lon), Slider (week), Slider (threshold), Checkbox (yearlong?), State (custom classifier))
    """
    with gr.Accordion(loc.localize("species-list-accordion-label"), open=opened):
        with gr.Row():
            species_list_radio = gr.Radio(
                [_CUSTOM_SPECIES, _PREDICT_SPECIES, _CUSTOM_CLASSIFIER, _ALL_SPECIES],
                value=_ALL_SPECIES,
                label=loc.localize("species-list-radio-label"),
                info=loc.localize("species-list-radio-info"),
                elem_classes="d-block",
            )

            with gr.Column(visible=False) as position_row:
                lat_number, lon_number, week_number, sf_thresh_number, yearlong_checkbox = species_list_coordinates()

            species_file_input = gr.File(
                file_types=[".txt"], visible=False, label=loc.localize("species-list-custom-list-file-label")
            )
            empty_col = gr.Column()

            with gr.Column(visible=False) as custom_classifier_selector:
                classifier_selection_button = gr.Button(
                    loc.localize("species-list-custom-classifier-selection-button-label")
                )
                classifier_file_input = gr.Files(file_types=[".tflite"], visible=False, interactive=False)
                selected_classifier_state = gr.State()

                def on_custom_classifier_selection_click():
                    file = select_file(("TFLite classifier (*.tflite)",), state_key="custom_classifier_file")

                    if file:
                        labels = os.path.splitext(file)[0] + "_Labels.txt"

                        return file, gr.File(value=[file, labels], visible=True)

                    return None, None

                classifier_selection_button.click(
                    on_custom_classifier_selection_click,
                    outputs=[selected_classifier_state, classifier_file_input],
                    show_progress=False,
                )

            species_list_radio.change(
                show_species_choice,
                inputs=[species_list_radio],
                outputs=[position_row, species_file_input, custom_classifier_selector, empty_col],
                show_progress=False,
            )

            return (
                species_list_radio,
                species_file_input,
                lat_number,
                lon_number,
                week_number,
                sf_thresh_number,
                yearlong_checkbox,
                selected_classifier_state,
            )


if __name__ == "__main__":
    multiprocessing.freeze_support()

    def build_header():
        # Custom HTML header with gr.Markdown
        # There has to be another way, but this works for now; paths are weird in gradio
        with gr.Row():
            gr.Markdown(
                f"""
                <div style='display: flex; align-items: center;'>
                    <img src='data:image/png;base64,{utils.img2base64(os.path.join(SCRIPT_DIR, "gui/img/birdnet_logo.png"))}' style='width: 50px; height: 50px; margin-right: 10px;'>
                    <h2>BirdNET Analyzer</h2>
                </div>
                """
            )

    def build_footer():
        with gr.Row():
            gr.Markdown(
                f"""
                <div style='display: flex; justify-content: space-around; align-items: center; padding: 10px; text-align: center'>
                    <div>
                        <div style="display: flex;flex-direction: row;">GUI version:&nbsp<span id="current-version">{os.environ['GUI_VERSION'] if FROZEN else 'main'}</span><span style="display: none" id="update-available"><a>+</a></span></div>
                        <div>Model version: {cfg.MODEL_VERSION}</div>
                    </div>
                    <div>K. Lisa Yang Center for Conservation Bioacoustics<br>Chemnitz University of Technology</div>
                    <div>{loc.localize('footer-help')}:<br><a href='https://birdnet.cornell.edu/analyzer' target='_blank'>birdnet.cornell.edu/analyzer</a></div>
                </div>
                """
            )

    def build_single_analysis_tab():
        with gr.Tab(loc.localize("single-tab-title")):
            audio_input = gr.Audio(type="filepath", label=loc.localize("single-audio-label"), sources=["upload"])
            audio_path_state = gr.State()

            confidence_slider, sensitivity_slider, overlap_slider, fmin_number, fmax_number = sample_sliders(False)

            (
                species_list_radio,
                species_file_input,
                lat_number,
                lon_number,
                week_number,
                sf_thresh_number,
                yearlong_checkbox,
                selected_classifier_state,
            ) = species_lists(False)
            locale_radio = locale()

            def get_audio_path(i):
                if i:
                    return i["path"], gr.Audio(i["path"], type="filepath", label=os.path.basename(i["path"]))
                else:
                    return None, None

            audio_input.change(
                get_audio_path, inputs=audio_input, outputs=[audio_path_state, audio_input], preprocess=False
            )

            inputs = [
                audio_path_state,
                confidence_slider,
                sensitivity_slider,
                overlap_slider,
                fmin_number,
                fmax_number,
                species_list_radio,
                species_file_input,
                lat_number,
                lon_number,
                week_number,
                yearlong_checkbox,
                sf_thresh_number,
                selected_classifier_state,
                locale_radio,
            ]

            output_dataframe = gr.Dataframe(
                type="pandas",
                headers=[
                    loc.localize("single-tab-output-header-start"),
                    loc.localize("single-tab-output-header-end"),
                    loc.localize("single-tab-output-header-sci-name"),
                    loc.localize("single-tab-output-header-common-name"),
                    loc.localize("single-tab-output-header-confidence"),
                ],
                elem_classes="matrix-mh-200",
            )

            single_file_analyze = gr.Button(loc.localize("analyze-start-button-label"))

            single_file_analyze.click(runSingleFileAnalysis, inputs=inputs, outputs=output_dataframe)

    def build_multi_analysis_tab():
        with gr.Tab(loc.localize("multi-tab-title")):
            input_directory_state = gr.State()
            output_directory_predict_state = gr.State()

            with gr.Row():
                with gr.Column():
                    select_directory_btn = gr.Button(loc.localize("multi-tab-input-selection-button-label"))
                    directory_input = gr.Matrix(
                        interactive=False,
                        elem_classes="matrix-mh-200",
                        headers=[
                            loc.localize("multi-tab-samples-dataframe-column-subpath-header"),
                            loc.localize("multi-tab-samples-dataframe-column-duration-header"),
                        ],
                    )

                    def select_directory_on_empty():
                        res = select_directory(max_files=101, state_key="batch-analysis-data-dir")

                        if res[1]:
                            if len(res[1]) > 100:
                                return [res[0], res[1][:100] + [["..."]]]  # hopefully fixes issue#272

                            return res

                        return [res[0], [[loc.localize("multi-tab-samples-dataframe-no-files-found")]]]

                    select_directory_btn.click(
                        select_directory_on_empty, outputs=[input_directory_state, directory_input], show_progress=True
                    )

                with gr.Column():
                    select_out_directory_btn = gr.Button(loc.localize("multi-tab-output-selection-button-label"))
                    selected_out_textbox = gr.Textbox(
                        label=loc.localize("multi-tab-output-textbox-label"),
                        interactive=False,
                        placeholder=loc.localize("multi-tab-output-textbox-placeholder"),
                    )

                    def select_directory_wrapper():
                        return (select_directory(collect_files=False, state_key="batch-analysis-output-dir"),) * 2

                    select_out_directory_btn.click(
                        select_directory_wrapper,
                        outputs=[output_directory_predict_state, selected_out_textbox],
                        show_progress=False,
                    )

            confidence_slider, sensitivity_slider, overlap_slider, fmin_number, fmax_number = sample_sliders()

            (
                species_list_radio,
                species_file_input,
                lat_number,
                lon_number,
                week_number,
                sf_thresh_number,
                yearlong_checkbox,
                selected_classifier_state,
            ) = species_lists()

            with gr.Accordion(loc.localize("multi-tab-output-accordion-label"), open=True):
                output_type_radio = gr.CheckboxGroup(
                    list(OUTPUT_TYPE_MAP.items()),
                    value="table",
                    label=loc.localize("multi-tab-output-radio-label"),
                    info=loc.localize("multi-tab-output-radio-info"),
                )

                with gr.Row():
                    with gr.Column():
                        combine_tables_checkbox = gr.Checkbox(
                            False,
                            label=loc.localize("multi-tab-output-combine-tables-checkbox-label"),
                            info=loc.localize("multi-tab-output-combine-tables-checkbox-info"),
                        )

                with gr.Row():
                    skip_existing_checkbox = gr.Checkbox(
                        False,
                        label=loc.localize("multi-tab-skip-existing-checkbox-label"),
                        info=loc.localize("multi-tab-skip-existing-checkbox-info"),
                    )

            with gr.Row():
                batch_size_number = gr.Number(
                    precision=1,
                    label=loc.localize("multi-tab-batchsize-number-label"),
                    value=1,
                    info=loc.localize("multi-tab-batchsize-number-info"),
                    minimum=1,
                )
                threads_number = gr.Number(
                    precision=1,
                    label=loc.localize("multi-tab-threads-number-label"),
                    value=4,
                    info=loc.localize("multi-tab-threads-number-info"),
                    minimum=1,
                )

            locale_radio = locale()

            start_batch_analysis_btn = gr.Button(loc.localize("analyze-start-button-label"))

            result_grid = gr.Matrix(
                headers=[
                    loc.localize("multi-tab-result-dataframe-column-file-header"),
                    loc.localize("multi-tab-result-dataframe-column-execution-header"),
                ],
                elem_classes="matrix-mh-200",
            )

            inputs = [
                output_directory_predict_state,
                confidence_slider,
                sensitivity_slider,
                overlap_slider,
                fmin_number,
                fmax_number,
                species_list_radio,
                species_file_input,
                lat_number,
                lon_number,
                week_number,
                yearlong_checkbox,
                sf_thresh_number,
                selected_classifier_state,
                output_type_radio,
                combine_tables_checkbox,
                locale_radio,
                batch_size_number,
                threads_number,
                input_directory_state,
                skip_existing_checkbox,
            ]

            start_batch_analysis_btn.click(runBatchAnalysis, inputs=inputs, outputs=result_grid)

    def build_train_tab():
        with gr.Tab(loc.localize("training-tab-title")):
            input_directory_state = gr.State()
            output_directory_state = gr.State()

            with gr.Row():
                with gr.Column():
                    select_directory_btn = gr.Button(loc.localize("training-tab-input-selection-button-label"))
                    directory_input = gr.List(
                        headers=[loc.localize("training-tab-classes-dataframe-column-classes-header")],
                        interactive=False,
                        elem_classes="matrix-mh-200",
                    )
                    select_directory_btn.click(
                        partial(select_subdirectories, state_key="train-data-dir"),
                        outputs=[input_directory_state, directory_input],
                        show_progress=False,
                    )

                with gr.Column():
                    select_directory_btn = gr.Button(loc.localize("training-tab-select-output-button-label"))

                    with gr.Column():
                        classifier_name = gr.Textbox(
                            "CustomClassifier",
                            visible=False,
                            info=loc.localize("training-tab-classifier-textbox-info"),
                        )
                        output_format = gr.Radio(
                            ["tflite", "raven", (loc.localize("training-tab-output-format-both"), "both")],
                            value="tflite",
                            label=loc.localize("training-tab-output-format-radio-label"),
                            info=loc.localize("training-tab-output-format-radio-info"),
                            visible=False,
                        )

                    def select_directory_and_update_tb():
                        initial_dir = loc.get_state("train-output-dir", "")
                        dir_name = _WINDOW.create_file_dialog(webview.FOLDER_DIALOG, directory=initial_dir)

                        if dir_name:
                            loc.set_state("train-output-dir", dir_name[0])
                            return (
                                dir_name[0],
                                gr.Textbox(label=dir_name[0] + "\\", visible=True),
                                gr.Radio(visible=True, interactive=True),
                            )

                        return None, None

                    select_directory_btn.click(
                        select_directory_and_update_tb,
                        outputs=[output_directory_state, classifier_name, output_format],
                        show_progress=False,
                    )

            autotune_cb = gr.Checkbox(
                False,
                label=loc.localize("training-tab-autotune-checkbox-label"),
                info=loc.localize("training-tab-autotune-checkbox-info"),
            )

            with gr.Column(visible=False) as autotune_params:
                with gr.Row():
                    autotune_trials = gr.Number(
                        50,
                        label=loc.localize("training-tab-autotune-trials-number-label"),
                        info=loc.localize("training-tab-autotune-trials-number-info"),
                        minimum=1,
                    )
                    autotune_executions_per_trials = gr.Number(
                        1,
                        minimum=1,
                        label=loc.localize("training-tab-autotune-executions-number-label"),
                        info=loc.localize("training-tab-autotune-executions-number-info"),
                    )

            with gr.Column() as custom_params:
                with gr.Row():
                    epoch_number = gr.Number(
                        50,
                        minimum=1,
                        label=loc.localize("training-tab-epochs-number-label"),
                        info=loc.localize("training-tab-epochs-number-info"),
                    )
                    batch_size_number = gr.Number(
                        32,
                        minimum=1,
                        label=loc.localize("training-tab-batchsize-number-label"),
                        info=loc.localize("training-tab-batchsize-number-info"),
                    )
                    learning_rate_number = gr.Number(
                        0.001,
                        minimum=0.0001,
                        label=loc.localize("training-tab-learningrate-number-label"),
                        info=loc.localize("training-tab-learningrate-number-info"),
                    )

                with gr.Row():
                    upsampling_mode = gr.Radio(
                        [
                            (loc.localize("training-tab-upsampling-radio-option-repeat"), "repeat"),
                            (loc.localize("training-tab-upsampling-radio-option-mean"), "mean"),
                            ("SMOTE", "smote"),
                        ],
                        value="repeat",
                        label=loc.localize("training-tab-upsampling-radio-label"),
                        info=loc.localize("training-tab-upsampling-radio-info"),
                    )
                    upsampling_ratio = gr.Slider(
                        0.0,
                        1.0,
                        0.0,
                        step=0.01,
                        label=loc.localize("training-tab-upsampling-ratio-slider-label"),
                        info=loc.localize("training-tab-upsampling-ratio-slider-info"),
                    )

                with gr.Row():
                    hidden_units_number = gr.Number(
                        0,
                        minimum=0,
                        label=loc.localize("training-tab-hiddenunits-number-label"),
                        info=loc.localize("training-tab-hiddenunits-number-info"),
                    )
                    use_mixup = gr.Checkbox(
                        False,
                        label=loc.localize("training-tab-use-mixup-checkbox-label"),
                        info=loc.localize("training-tab-use-mixup-checkbox-info"),
                        show_label=True,
                    )

            def on_autotune_change(value):
                return gr.Column(visible=not value), gr.Column(visible=value)

            autotune_cb.change(
                on_autotune_change, inputs=autotune_cb, outputs=[custom_params, autotune_params], show_progress=False
            )

            with gr.Row():

                fmin_number = gr.Number(
                    cfg.SIG_FMIN,
                    minimum=0,
                    label=loc.localize("inference-settings-fmin-number-label"),
                    info=loc.localize("inference-settings-fmin-number-info"),
                )

                fmax_number = gr.Number(
                    cfg.SIG_FMAX,
                    minimum=0,
                    label=loc.localize("inference-settings-fmax-number-label"),
                    info=loc.localize("inference-settings-fmax-number-info"),
                )

            with gr.Row():
                crop_mode = gr.Radio(
                    [
                        (loc.localize("training-tab-crop-mode-radio-option-center"), "center"),
                        (loc.localize("training-tab-crop-mode-radio-option-first"), "first"),
                        (loc.localize("training-tab-crop-mode-radio-option-segments"), "segments"),
                    ],
                    value="center",
                    label=loc.localize("training-tab-crop-mode-radio-label"),
                    info=loc.localize("training-tab-crop-mode-radio-info"),
                )

                crop_overlap = gr.Slider(
                    minimum=0,
                    maximum=2.99,
                    value=0,
                    step=0.01,
                    label=loc.localize("training-tab-crop-overlap-number-label"),
                    info=loc.localize("training-tab-crop-overlap-number-info"),
                    visible=False,
                )

                def on_crop_select(new_crop_mode):
                    return gr.Number(visible=new_crop_mode == "segments", interactive=new_crop_mode == "segments")

                crop_mode.change(on_crop_select, inputs=crop_mode, outputs=crop_overlap)

            model_save_mode = gr.Radio(
                [
                    (loc.localize("training-tab-model-save-mode-radio-option-replace"), "replace"),
                    (loc.localize("training-tab-model-save-mode-radio-option-append"), "append"),
                ],
                value="replace",
                label=loc.localize("training-tab-model-save-mode-radio-label"),
                info=loc.localize("training-tab-model-save-mode-radio-info"),
            )

            with gr.Row():
                cache_file_state = gr.State()
                cache_mode = gr.Radio(
                    [
                        (loc.localize("training-tab-cache-mode-radio-option-none"), "none"),
                        (loc.localize("training-tab-cache-mode-radio-option-load"), "load"),
                        (loc.localize("training-tab-cache-mode-radio-option-save"), "save"),
                    ],
                    value="none",
                    label=loc.localize("training-tab-cache-mode-radio-label"),
                    info=loc.localize("training-tab-cache-mode-radio-info"),
                )
                with gr.Column(visible=False) as new_cache_file_row:
                    select_cache_file_directory_btn = gr.Button(
                        loc.localize("training-tab-cache-select-directory-button-label")
                    )

                    with gr.Column():
                        cache_file_name = gr.Textbox(
                            "train_cache.npz",
                            visible=False,
                            info=loc.localize("training-tab-cache-file-name-textbox-info"),
                        )

                    def select_directory_and_update():
                        initial_dir = loc.get_state("train-data-cache-file-output", "")
                        dir_name = _WINDOW.create_file_dialog(webview.FOLDER_DIALOG, directory=initial_dir)

                        if dir_name:
                            loc.set_state("train-data-cache-file-output", dir_name[0])
                            return (
                                dir_name[0],
                                gr.Textbox(label=dir_name[0] + "\\", visible=True),
                            )

                        return None, None

                    select_cache_file_directory_btn.click(
                        select_directory_and_update,
                        outputs=[cache_file_state, cache_file_name],
                        show_progress=False,
                    )

                with gr.Column(visible=False) as load_cache_file_row:
                    selected_cache_file_btn = gr.Button(loc.localize("training-tab-cache-select-file-button-label"))
                    cache_file_input = gr.File(file_types=[".npz"], visible=False, interactive=False)

                    def on_cache_file_selection_click():
                        file = select_file(("NPZ file (*.npz)",), state_key="train_data_cache_file")

                        if file:
                            return file, gr.File(value=file, visible=True)

                        return None, None

                    selected_cache_file_btn.click(
                        on_cache_file_selection_click,
                        outputs=[cache_file_state, cache_file_input],
                        show_progress=False,
                    )

                def on_cache_mode_change(value):
                    return gr.Row(visible=value == "save"), gr.Row(visible=value == "load")

                cache_mode.change(
                    on_cache_mode_change, inputs=cache_mode, outputs=[new_cache_file_row, load_cache_file_row]
                )

            train_history_plot = gr.Plot()

            start_training_button = gr.Button(loc.localize("training-tab-start-training-button-label"))

            start_training_button.click(
                start_training,
                inputs=[
                    input_directory_state,
                    crop_mode,
                    crop_overlap,
                    fmin_number,
                    fmax_number,
                    output_directory_state,
                    classifier_name,
                    model_save_mode,
                    cache_mode,
                    cache_file_state,
                    cache_file_name,
                    autotune_cb,
                    autotune_trials,
                    autotune_executions_per_trials,
                    epoch_number,
                    batch_size_number,
                    learning_rate_number,
                    hidden_units_number,
                    use_mixup,
                    upsampling_ratio,
                    upsampling_mode,
                    output_format,
                ],
                outputs=[train_history_plot],
            )

    def build_segments_tab():
        with gr.Tab(loc.localize("segments-tab-title")):
            audio_directory_state = gr.State()
            result_directory_state = gr.State()
            output_directory_state = gr.State()

            def select_directory_to_state_and_tb(state_key):
                return (select_directory(collect_files=False, state_key=state_key),) * 2

            with gr.Row():
                select_audio_directory_btn = gr.Button(
                    loc.localize("segments-tab-select-audio-input-directory-button-label")
                )
                selected_audio_directory_tb = gr.Textbox(show_label=False, interactive=False)
                select_audio_directory_btn.click(
                    partial(select_directory_to_state_and_tb, state_key="segments-audio-dir"),
                    outputs=[selected_audio_directory_tb, audio_directory_state],
                    show_progress=False,
                )

            with gr.Row():
                select_result_directory_btn = gr.Button(
                    loc.localize("segments-tab-select-results-input-directory-button-label")
                )
                selected_result_directory_tb = gr.Textbox(
                    show_label=False,
                    interactive=False,
                    placeholder=loc.localize("segments-tab-results-input-textbox-placeholder"),
                )
                select_result_directory_btn.click(
                    partial(select_directory_to_state_and_tb, state_key="segments-result-dir"),
                    outputs=[result_directory_state, selected_result_directory_tb],
                    show_progress=False,
                )

            with gr.Row():
                select_output_directory_btn = gr.Button(loc.localize("segments-tab-output-selection-button-label"))
                selected_output_directory_tb = gr.Textbox(
                    show_label=False,
                    interactive=False,
                    placeholder=loc.localize("segments-tab-output-selection-textbox-placeholder"),
                )
                select_output_directory_btn.click(
                    partial(select_directory_to_state_and_tb, state_key="segments-output-dir"),
                    outputs=[selected_output_directory_tb, output_directory_state],
                    show_progress=False,
                )

            min_conf_slider = gr.Slider(
                minimum=0.1,
                maximum=0.99,
                step=0.01,
                label=loc.localize("segments-tab-min-confidence-slider-label"),
                info=loc.localize("segments-tab-min-confidence-slider-info"),
            )
            num_seq_number = gr.Number(
                100,
                label=loc.localize("segments-tab-max-seq-number-label"),
                info=loc.localize("segments-tab-max-seq-number-info"),
                minimum=1,
            )
            seq_length_number = gr.Number(
                3.0,
                label=loc.localize("segments-tab-seq-length-number-label"),
                info=loc.localize("segments-tab-seq-length-number-info"),
                minimum=0.1,
            )
            threads_number = gr.Number(
                4,
                label=loc.localize("segments-tab-threads-number-label"),
                info=loc.localize("segments-tab-threads-number-info"),
                minimum=1,
            )

            extract_segments_btn = gr.Button(loc.localize("segments-tab-extract-button-label"))

            result_grid = gr.Matrix(
                headers=[
                    loc.localize("segments-tab-result-dataframe-column-file-header"),
                    loc.localize("segments-tab-result-dataframe-column-execution-header"),
                ],
                elem_classes="matrix-mh-200",
            )

            extract_segments_btn.click(
                extract_segments,
                inputs=[
                    audio_directory_state,
                    result_directory_state,
                    output_directory_state,
                    min_conf_slider,
                    num_seq_number,
                    seq_length_number,
                    threads_number,
                ],
                outputs=result_grid,
            )

    def build_review_tab():
        def collect_segments(directory, shuffle=False):
            import random

            segments = (
                [
                    entry.path
                    for entry in os.scandir(directory)
                    if entry.is_file() and entry.name.rsplit(".", 1)[-1] in cfg.ALLOWED_FILETYPES
                ]
                if os.path.isdir(directory)
                else []
            )

            if shuffle:
                random.shuffle(segments)

            return segments

        def collect_files(directory):
            return (
                collect_segments(directory, shuffle=True),
                collect_segments(os.path.join(directory, POSITIVE_LABEL_DIR)),
                collect_segments(os.path.join(directory, NEGATIVE_LABEL_DIR)),
            )

        def create_log_plot(positives, negatives, fig_num=None):
            import matplotlib.pyplot as plt
            from sklearn import linear_model
            import numpy as np
            from scipy.special import expit

            f = plt.figure(fig_num, figsize=(12, 6))
            f.set_dpi(300)
            f.clf()

            ax = f.add_subplot(111)
            ax.set_xlim(0, 1)
            ax.set_yticks([0, 1])
            ax.set_ylabel(
                f"{loc.localize('review-tab-regression-plot-y-label-false')}/{loc.localize('review-tab-regression-plot-y-label-true')}"
            )
            ax.set_xlabel(loc.localize("review-tab-regression-plot-x-label"))

            x_vals = []
            y_val = []

            for fl in positives + negatives:
                try:
                    x_val = float(os.path.basename(fl).split("_", 1)[0])

                    if 0 > x_val > 1:
                        continue

                    x_vals.append([x_val])
                    y_val.append(1 if fl in positives else 0)
                except ValueError:
                    pass

            if (len(positives) + len(negatives)) >= 2 and len(set(y_val)) > 1:
                log_model = linear_model.LogisticRegression(C=55)
                log_model.fit(x_vals, y_val)
                Xs = np.linspace(0, 10, 200)
                Ys = expit(Xs * log_model.coef_ + log_model.intercept_).ravel()
                target_ps = [0.85, 0.9, 0.95, 0.99]
                thresholds = [
                    (np.log(target_p / (1 - target_p)) - log_model.intercept_[0]) / log_model.coef_[0][0]
                    for target_p in target_ps
                ]
                p_colors = ["blue", "purple", "orange", "green"]

                for target_p, p_color, threshold in zip(target_ps, p_colors, thresholds):
                    ax.vlines(
                        threshold,
                        0,
                        target_p,
                        color=p_color,
                        linestyle="--",
                        linewidth=0.5,
                        label=f"p={target_p:.2f} treshold>={threshold:.2f}",
                    )
                    ax.hlines(target_p, 0, threshold, color=p_color, linestyle="--", linewidth=0.5)

                ax.plot(Xs, Ys, color="red")
                ax.scatter(thresholds, target_ps, color=p_colors, marker="x")

                box = ax.get_position()
                ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])
                ax.legend(loc="center left", bbox_to_anchor=(1, 0.5))

            if len(y_val) > 0:
                ax.scatter(x_vals, y_val, 2)

            return gr.Plot(value=f, visible=bool(y_val))

        with gr.Tab(loc.localize("review-tab-title")):
            review_state = gr.State(
                {
                    "input_directory": "",
                    "spcies_list": [],
                    "current_species": "",
                    "files": [],
                    "current": 0,
                    POSITIVE_LABEL_DIR: [],
                    NEGATIVE_LABEL_DIR: [],
                }
            )

            select_directory_btn = gr.Button(loc.localize("review-tab-input-directory-button-label"))

            with gr.Column(visible=False) as review_col:
                with gr.Row():
                    species_dropdown = gr.Dropdown(label=loc.localize("review-tab-species-dropdown-label"))
                    file_count_matrix = gr.Matrix(
                        headers=[
                            loc.localize("review-tab-file-matrix-todo-header"),
                            loc.localize("review-tab-file-matrix-pos-header"),
                            loc.localize("review-tab-file-matrix-neg-header"),
                        ],
                        interactive=False,
                    )

                with gr.Column() as review_item_col:
                    with gr.Row():
                        spectrogram_image = gr.Plot(label=loc.localize("review-tab-spectrogram-plot-label"))

                        with gr.Column():
                            positive_btn = gr.Button(loc.localize("review-tab-pos-button-label"))
                            negative_btn = gr.Button(loc.localize("review-tab-neg-button-label"))
                            review_audio = gr.Audio(
                                type="filepath", sources=[], show_download_button=False, autoplay=True
                            )
                            autoplay_checkbox = gr.Checkbox(
                                True, label=loc.localize("review-tab-autoplay-checkbox-label")
                            )

                no_samles_label = gr.Label(loc.localize("review-tab-no-files-label"), visible=False)
                species_regression_plot = gr.Plot(label=loc.localize("review-tab-regression-plot-label"))

            def next_review(next_review_state: dict, target_dir: str = None):
                current_file = next_review_state["files"][next_review_state["current"]]

                update_dict = {review_state: next_review_state}

                if target_dir:
                    selected_dir = os.path.join(
                        next_review_state["input_directory"], next_review_state["current_species"], target_dir
                    )

                    os.makedirs(selected_dir, exist_ok=True)

                    os.rename(
                        current_file,
                        os.path.join(selected_dir, os.path.basename(current_file)),
                    )

                    next_review_state[target_dir] += [current_file]
                    next_review_state["files"].remove(current_file)

                    update_dict |= {
                        species_regression_plot: create_log_plot(
                            next_review_state[POSITIVE_LABEL_DIR], next_review_state[NEGATIVE_LABEL_DIR], 2
                        ),
                    }

                    if not next_review_state["files"]:
                        update_dict |= {
                            no_samles_label: gr.Label(visible=True),
                            review_item_col: gr.Column(visible=False),
                        }
                    else:
                        next_file = next_review_state["files"][next_review_state["current"]]
                        update_dict |= {
                            review_audio: gr.Audio(next_file, label=os.path.basename(next_file)),
                            spectrogram_image: utils.spectrogram_from_file(next_file),
                        }

                    update_dict |= {
                        file_count_matrix: [
                            [
                                len(next_review_state["files"]),
                                len(next_review_state[POSITIVE_LABEL_DIR]),
                                len(next_review_state[NEGATIVE_LABEL_DIR]),
                            ],
                        ],
                    }
                else:
                    if next_review_state["current"] + 1 < len(next_review_state["files"]):
                        next_review_state["current"] += 1
                        next_file = next_review_state["files"][next_review_state["current"]]

                        update_dict |= {
                            review_audio: gr.Audio(next_file, label=os.path.basename(next_file)),
                            spectrogram_image: utils.spectrogram_from_file(next_file),
                        }

                return update_dict

            def select_subdir(new_value: str, next_review_state: dict):
                if new_value != next_review_state["current_species"]:
                    return update_review(next_review_state, selected_species=new_value)
                else:
                    return {review_state: next_review_state}

            def start_review(next_review_state):
                initial_dir = loc.get_state("review-input-dir", "")
                dir_name = _WINDOW.create_file_dialog(webview.FOLDER_DIALOG, directory=initial_dir)

                if dir_name:
                    loc.set_state("review-input-dir", dir_name[0])
                    next_review_state["input_directory"] = dir_name[0]
                    specieslist = [e.name for e in os.scandir(next_review_state["input_directory"]) if e.is_dir()]

                    if not specieslist:
                        raise gr.Error(loc.localize("review-tab-no-species-found-error"))

                    next_review_state["species_list"] = specieslist

                    return update_review(next_review_state)

                else:
                    return {review_state: next_review_state}

            def update_review(next_review_state: dict, selected_species: str = None):
                if selected_species:
                    next_review_state["current_species"] = selected_species
                else:
                    next_review_state["current_species"] = next_review_state["species_list"][0]

                todo_files, positives, negatives = collect_files(
                    os.path.join(next_review_state["input_directory"], next_review_state["current_species"])
                )

                next_review_state |= {
                    "files": todo_files,
                    POSITIVE_LABEL_DIR: positives,
                    NEGATIVE_LABEL_DIR: negatives,
                }

                update_dict = {
                    review_col: gr.Column(visible=True),
                    review_state: next_review_state,
                    file_count_matrix: [
                        [
                            len(next_review_state["files"]),
                            len(next_review_state[POSITIVE_LABEL_DIR]),
                            len(next_review_state[NEGATIVE_LABEL_DIR]),
                        ],
                    ],
                    species_regression_plot: create_log_plot(
                        next_review_state[POSITIVE_LABEL_DIR], next_review_state[NEGATIVE_LABEL_DIR], 2
                    ),
                }

                if not selected_species:
                    update_dict |= {
                        species_dropdown: gr.Dropdown(
                            choices=next_review_state["species_list"], value=next_review_state["current_species"]
                        )
                    }

                if todo_files:
                    update_dict |= {
                        review_item_col: gr.Column(visible=True),
                        review_audio: gr.Audio(value=todo_files[0], label=os.path.basename(todo_files[0])),
                        spectrogram_image: utils.spectrogram_from_file(todo_files[0], 1),
                        no_samles_label: gr.Label(visible=False),
                    }
                else:
                    update_dict |= {review_item_col: gr.Column(visible=False), no_samles_label: gr.Label(visible=True)}

                return update_dict

            def toggle_autoplay(value):
                return gr.Audio(autoplay=value)

            autoplay_checkbox.change(toggle_autoplay, inputs=autoplay_checkbox, outputs=review_audio)

            review_change_output = [
                review_col,
                review_item_col,
                review_audio,
                spectrogram_image,
                species_dropdown,
                no_samles_label,
                review_state,
                file_count_matrix,
                species_regression_plot,
            ]

            species_dropdown.change(
                select_subdir,
                show_progress=True,
                inputs=[species_dropdown, review_state],
                outputs=review_change_output,
            )

            review_btn_output = [
                review_audio,
                spectrogram_image,
                review_state,
                review_item_col,
                no_samles_label,
                file_count_matrix,
                species_regression_plot,
            ]

            positive_btn.click(
                partial(next_review, target_dir=POSITIVE_LABEL_DIR),
                inputs=review_state,
                outputs=review_btn_output,
                show_progress=True,
            )

            negative_btn.click(
                partial(next_review, target_dir=NEGATIVE_LABEL_DIR),
                inputs=review_state,
                outputs=review_btn_output,
                show_progress=True,
            )

            select_directory_btn.click(
                start_review,
                inputs=review_state,
                outputs=review_change_output,
                show_progress=True,
            )

    def build_species_tab():
        with gr.Tab(loc.localize("species-tab-title")):
            output_directory_state = gr.State()
            select_directory_btn = gr.Button(loc.localize("species-tab-select-output-directory-button-label"))
            classifier_name = gr.Textbox(
                "species_list.txt",
                visible=False,
                info=loc.localize("species-tab-filename-textbox-label"),
            )

            def select_directory_and_update_tb(name_tb):
                initial_dir = loc.get_state("species-output-dir", "")
                dir_name = _WINDOW.create_file_dialog(webview.FOLDER_DIALOG, directory=initial_dir)

                if dir_name:
                    loc.set_state("species-output-dir", dir_name[0])
                    return (
                        dir_name[0],
                        gr.Textbox(label=dir_name[0] + "\\", visible=True, value=name_tb),
                    )

                return None, name_tb

            select_directory_btn.click(
                select_directory_and_update_tb,
                inputs=classifier_name,
                outputs=[output_directory_state, classifier_name],
                show_progress=False,
            )

            lat_number, lon_number, week_number, sf_thresh_number, yearlong_checkbox = species_list_coordinates()

            sortby = gr.Radio(
                [
                    (loc.localize("species-tab-sort-radio-option-frequency"), "freq"),
                    (loc.localize("species-tab-sort-radio-option-alphabetically"), "alpha"),
                ],
                value="freq",
                label=loc.localize("species-tab-sort-radio-label"),
                info=loc.localize("species-tab-sort-radio-info"),
            )

            start_btn = gr.Button(loc.localize("species-tab-start-button-label"))
            start_btn.click(
                run_species_list,
                inputs=[
                    output_directory_state,
                    classifier_name,
                    lat_number,
                    lon_number,
                    week_number,
                    yearlong_checkbox,
                    sf_thresh_number,
                    sortby,
                ],
            )

    def build_settings():
        with gr.Tab(loc.localize("settings-tab-title")) as settings_tab:
            with gr.Row():
                options = [
                    lang.rsplit(".", 1)[0]
                    for lang in os.listdir(os.path.join(SCRIPT_DIR, "lang"))
                    if lang.endswith(".json")
                ]
                languages_dropdown = gr.Dropdown(
                    options,
                    value=loc.TARGET_LANGUAGE,
                    label=loc.localize("settings-tab-language-dropdown-label"),
                    info=loc.localize("settings-tab-language-dropdown-info"),
                    interactive=True,
                )

            gr.Markdown(
                """
                If you encounter a bug or error, please provide the error log.\n
                You can submit an issue on our [GitHub](https://github.com/kahst/BirdNET-Analyzer/issues).
                """,
                label=loc.localize("settings-tab-error-log-textbox-label"),
                elem_classes="mh-200",
            )

            error_log_tb = gr.TextArea(
                label=loc.localize("settings-tab-error-log-textbox-label"),
                info=f"{loc.localize('settings-tab-error-log-textbox-info-path')}: {cfg.ERROR_LOG_FILE}",
                interactive=False,
                placeholder=loc.localize("settings-tab-error-log-textbox-placeholder"),
                show_copy_button=True,
            )

            def on_language_change(value):
                loc.set_language(value)

            def on_tab_select(value: gr.SelectData):
                if value.selected and os.path.exists(cfg.ERROR_LOG_FILE):
                    with open(cfg.ERROR_LOG_FILE, "r", encoding="utf-8") as f:
                        lines = f.readlines()
                        last_100_lines = lines[-100:]
                        return "".join(last_100_lines)

                return ""

            languages_dropdown.input(on_language_change, inputs=languages_dropdown, show_progress=False)

            settings_tab.select(on_tab_select, outputs=error_log_tb, show_progress=False)

    with gr.Blocks(
        css=os.path.join(SCRIPT_DIR, "gui/gui.css"),
        js=os.path.join(SCRIPT_DIR, "gui/gui.js"),
        theme=gr.themes.Default(),
        analytics_enabled=False,
    ) as demo:
        build_header()
        build_single_analysis_tab()
        build_multi_analysis_tab()
        build_train_tab()
        build_segments_tab()
        build_review_tab()
        build_species_tab()
        build_settings()
        build_footer()

    url = demo.queue(api_open=False).launch(prevent_thread_lock=True, quiet=True)[1]
    _WINDOW = webview.create_window("BirdNET-Analyzer", url.rstrip("/") + "?__theme=light", min_size=(1024, 768))

    with suppress(ModuleNotFoundError):
        import pyi_splash  # type: ignore

        pyi_splash.close()

    webview.start(private_mode=False)
