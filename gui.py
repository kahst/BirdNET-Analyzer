import concurrent.futures
import os
import sys
from pathlib import Path

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

    logsdir = userdir / "BirdNET-Analyzer-GUI"

    logsdir.mkdir(parents=True, exist_ok=True)

    sys.stderr = sys.stdout = open(str(logsdir / "logs.txt"), "w")
    cfg.ERROR_LOG_FILE = str(logsdir / cfg.ERROR_LOG_FILE)


import multiprocessing

import gradio as gr
import librosa
import webview

import analyze
import segments
import species
import utils
from train import trainModel

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
    validate(out_path, "Please select a directory for the species list.")

    species.run(
        os.path.join(out_path, filename if filename else "species_list.txt"),
        lat,
        lon,
        -1 if use_yearlong else week,
        sf_thresh,
        sortby,
    )

    gr.Info(f"Species list saved at {cfg.OUTPUT_PATH}")


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
    validate(input_path, "Please select a file.")

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
    output_filename,
    combine_tables,
    locale,
    batch_size,
    threads,
    input_dir,
    skip_existing,
    progress=gr.Progress(),
):
    validate(input_dir, "Please select a directory.")
    batch_size = int(batch_size)
    threads = int(threads)

    if species_list_choice == _CUSTOM_SPECIES:
        validate(species_list_file, "Please select a species list.")

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
        output_filename if combine_tables else None,
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
    output_type: str,
    output_filename: str | None,
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
        progress(0, desc="Preparing ...")

    locale = locale.lower()
    # Load eBird codes, labels
    cfg.CODES = analyze.loadCodes()
    cfg.LABELS = utils.readLines(ORIGINAL_LABELS_FILE)
    cfg.LATITUDE, cfg.LONGITUDE, cfg.WEEK = lat, lon, -1 if use_yearlong else week
    cfg.LOCATION_FILTER_THRESHOLD = sf_thresh
    cfg.SKIP_EXISTING_RESULTS = skip_existing

    if species_list_choice == _CUSTOM_SPECIES:
        if not species_list_file or not species_list_file.name:
            cfg.SPECIES_LIST_FILE = None
        else:
            cfg.SPECIES_LIST_FILE = os.path.join(os.path.dirname(os.path.abspath(sys.argv[0])), species_list_file.name)

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
            raise gr.Error("No custom classifier selected.")

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
        cfg.TRANSLATED_LABELS_PATH, os.path.basename(cfg.LABELS_FILE).replace(".txt", f"_{locale}.txt")
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
        cfg.OUTPUT_PATH = output_path if output_path else input_path.split(".", 1)[0] + ".csv"

    # Parse input files
    if input_dir:
        cfg.FILE_LIST = utils.collect_audio_files(input_dir)
        cfg.INPUT_PATH = input_dir
    elif os.path.isdir(cfg.INPUT_PATH):
        cfg.FILE_LIST = utils.collect_audio_files(cfg.INPUT_PATH)
    else:
        cfg.FILE_LIST = [cfg.INPUT_PATH]

    validate(cfg.FILE_LIST, "No audio files found.")

    # Set confidence threshold
    cfg.MIN_CONFIDENCE = confidence

    # Set sensitivity
    cfg.SIGMOID_SENSITIVITY = sensitivity

    # Set overlap
    cfg.SIG_OVERLAP = overlap

    # Set frequency range
    cfg.BANDPASS_FMIN = max(0, min(cfg.SIG_FMAX, int(fmin)))
    cfg.BANDPASS_FMAX = max(cfg.SIG_FMIN, min(cfg.SIG_FMAX, int(fmax)))

    # Set result type
    cfg.RESULT_TYPE = OUTPUT_TYPE_MAP[output_type] if output_type in OUTPUT_TYPE_MAP else output_type.lower()

    if not cfg.RESULT_TYPE in ["table", "audacity", "r", "csv", "kaleidoscope"]:
        cfg.RESULT_TYPE = "table"

    # Set output filename
    if output_filename is not None and cfg.RESULT_TYPE == "table":
        cfg.OUTPUT_FILE = output_filename
    else:
        cfg.OUTPUT_FILE = None

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
        progress(0, desc="Starting ...")

    # Analyze files
    if cfg.CPU_THREADS < 2:
        for entry in flist:
            result = analyzeFile_wrapper(entry)

            result_list.append(result)
    else:
        with concurrent.futures.ProcessPoolExecutor(max_workers=cfg.CPU_THREADS) as executor:
            futures = (executor.submit(analyzeFile_wrapper, arg) for arg in flist)
            for i, f in enumerate(concurrent.futures.as_completed(futures), start=1):
                if progress is not None:
                    progress((i, len(flist)), total=len(flist), unit="files")
                result = f.result()

                result_list.append(result)

    # Combine results?
    if not cfg.OUTPUT_FILE is None:
        print(f"Combining results into {cfg.OUTPUT_FILE}...", end="", flush=True)
        analyze.combineResults(cfg.OUTPUT_PATH, cfg.OUTPUT_FILE)
        print("done!", flush=True)

    return [[os.path.relpath(r[0], input_dir), r[1]] for r in result_list] if input_dir else cfg.OUTPUT_PATH


_CUSTOM_SPECIES = "Custom species list"
_PREDICT_SPECIES = "Species by location"
_CUSTOM_CLASSIFIER = "Custom classifier"
_ALL_SPECIES = "all species"


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


def select_subdirectories():
    """Creates a directory selection dialog.

    Returns:
        A tuples of (directory, list of subdirectories) or (None, None) if the dialog was canceled.
    """
    dir_name = _WINDOW.create_file_dialog(webview.FOLDER_DIALOG)

    if dir_name:
        subdirs = utils.list_subdirectories(dir_name[0])
        labels = []

        for folder in subdirs:
            labels_in_folder = folder.split(",")

            for label in labels_in_folder:
                if not label in labels:
                    labels.append(label)

        return dir_name[0], [[label] for label in sorted(labels)]

    return None, None


def select_file(filetypes=()):
    """Creates a file selection dialog.

    Args:
        filetypes: List of filetypes to be filtered in the dialog.

    Returns:
        The selected file or None of the dialog was canceled.
    """
    files = _WINDOW.create_file_dialog(webview.OPEN_DIALOG, file_types=filetypes)

    return files[0] if files else None


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


def select_directory(collect_files=True):
    """Shows a directory selection system dialog.

    Uses the pywebview to create a system dialog.

    Args:
        collect_files: If True, also lists a files inside the directory.

    Returns:
        If collect_files==True, returns (directory path, list of (relative file path, audio length))
        else just the directory path.
        All values will be None of the dialog is cancelled.
    """
    dir_name = _WINDOW.create_file_dialog(webview.FOLDER_DIALOG)

    if collect_files:
        if not dir_name:
            return None, None

        files = utils.collect_audio_files(dir_name[0])

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
    validate(data_dir, "Please select your Training data.")
    validate(output_dir, "Please select a directory for the classifier.")
    validate(classifier_name, "Please enter a valid name for the classifier.")

    if not epochs or epochs < 0:
        raise gr.Error("Please enter a valid number of epochs.")

    if not batch_size or batch_size < 0:
        raise gr.Error("Please enter a valid batch size.")

    if not learning_rate or learning_rate < 0:
        raise gr.Error("Please enter a valid learning rate.")

    if fmin < cfg.SIG_FMIN or fmax > cfg.SIG_FMAX or fmin > fmax:
        raise gr.Error(f"Please enter valid frequency range in [{cfg.SIG_FMIN}, {cfg.SIG_FMAX}]")

    if not hidden_units or hidden_units < 0:
        hidden_units = 0

    if progress is not None:
        progress((0, epochs), desc="Loading data & building classifier", unit="epochs")

    cfg.TRAIN_DATA_PATH = data_dir
    cfg.SAMPLE_CROP_MODE = crop_mode
    cfg.SIG_OVERLAP = crop_overlap
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
                (num_files, num_total_files), total=num_total_files, unit="files", desc=f"Loading data for '{label}'"
            )

    def epochProgression(epoch, logs=None):
        if progress is not None:
            if epoch + 1 == epochs:
                progress((epoch + 1, epochs), total=epochs, unit="epochs", desc=f"Saving at {cfg.CUSTOM_CLASSIFIER}")
            else:
                progress((epoch + 1, epochs), total=epochs, unit="epochs", desc=f"Training model")

    def trialProgression(trial):
        if progress is not None:
            progress((trial, autotune_trials), total=autotune_trials, unit="trials", desc=f"Autotune in progress")

    history = trainModel(
        on_epoch_end=epochProgression, on_trial_result=trialProgression, on_data_load_end=dataLoadProgression
    )

    if len(history.epoch) < epochs:
        gr.Info("Stopped early - validation metric not improving.")

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
    validate(audio_dir, "No audio directory selected")

    if not result_dir:
        result_dir = audio_dir

    if not output_dir:
        output_dir = audio_dir

    if progress is not None:
        progress(0, desc="Searching files ...")

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
    with gr.Accordion("Inference settings", open=opened):
        with gr.Row():
            confidence_slider = gr.Slider(
                minimum=0,
                maximum=1,
                value=0.5,
                step=0.01,
                label="Minimum Confidence",
                info="Minimum confidence threshold.",
            )
            sensitivity_slider = gr.Slider(
                minimum=0.5,
                maximum=1.5,
                value=1,
                step=0.01,
                label="Sensitivity",
                info="Detection sensitivity; Higher values result in higher sensitivity.",
            )
            overlap_slider = gr.Slider(
                minimum=0, maximum=2.99, value=0, step=0.01, label="Overlap", info="Overlap of prediction segments."
            )

        with gr.Row():
            fmin_number = gr.Number(
                cfg.SIG_FMIN,
                label="Minimum bandpass frequency in Hz.",
                info="Note that frequency cut-offs should also be used during training in order to be effective here.",
            )

            fmax_number = gr.Number(
                cfg.SIG_FMAX,
                label="Maximum bandpass frequency in Hz.",
                info="Note that frequency cut-offs should also be used during training in order to be effective here.",
            )

        return confidence_slider, sensitivity_slider, overlap_slider, fmin_number, fmax_number


def locale():
    """Creates the gradio elements for locale selection

    Reads the translated labels inside the checkpoints directory.

    Returns:
        The dropdown element.
    """
    label_files = os.listdir(os.path.join(os.path.dirname(sys.argv[0]), ORIGINAL_TRANSLATED_LABELS_PATH))
    options = ["EN"] + [label_file.rsplit("_", 1)[-1].split(".")[0].upper() for label_file in label_files]

    return gr.Dropdown(options, value="EN", label="Locale", info="Locale for the translated species common names.")


def species_list_coordinates():
    lat_number = gr.Slider(
        minimum=-90, maximum=90, value=0, step=1, label="Latitude", info="Recording location latitude."
    )
    lon_number = gr.Slider(
        minimum=-180, maximum=180, value=0, step=1, label="Longitude", info="Recording location longitude."
    )
    with gr.Row():
        yearlong_checkbox = gr.Checkbox(True, label="Year-round")
        week_number = gr.Slider(
            minimum=1,
            maximum=48,
            value=1,
            step=1,
            interactive=False,
            label="Week",
            info="Week of the year when the recording was made. Values in [1, 48] (4 weeks per month).",
        )

        def onChange(use_yearlong):
            return gr.Slider(interactive=(not use_yearlong))

        yearlong_checkbox.change(onChange, inputs=yearlong_checkbox, outputs=week_number, show_progress=False)
    sf_thresh_number = gr.Slider(
        minimum=0.01,
        maximum=0.99,
        value=0.03,
        step=0.01,
        label="Location filter threshold",
        info="Minimum species occurrence frequency threshold for location filter.",
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
    with gr.Accordion("Species selection", open=opened):
        with gr.Row():
            species_list_radio = gr.Radio(
                [_CUSTOM_SPECIES, _PREDICT_SPECIES, _CUSTOM_CLASSIFIER, _ALL_SPECIES],
                value=_ALL_SPECIES,
                label="Species list",
                info="List of all possible species",
                elem_classes="d-block",
            )

            with gr.Column(visible=False) as position_row:
                lat_number, lon_number, week_number, sf_thresh_number, yearlong_checkbox = species_list_coordinates()

            species_file_input = gr.File(file_types=[".txt"], visible=False)
            empty_col = gr.Column()

            with gr.Column(visible=False) as custom_classifier_selector:
                classifier_selection_button = gr.Button("Select classifier")
                classifier_file_input = gr.Files(file_types=[".tflite"], visible=False, interactive=False)
                selected_classifier_state = gr.State()

                def on_custom_classifier_selection_click():
                    file = select_file(("TFLite classifier (*.tflite)",))

                    if file:
                        labels = os.path.splitext(file)[0] + "_Labels.txt"

                        return file, gr.File(value=[file, labels], visible=True)

                    return None

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
                    <img src='data:image/png;base64,{utils.img2base64("gui/img/birdnet_logo.png")}' style='width: 50px; height: 50px; margin-right: 10px;'>
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
                        <div style="display: flex;flex-direction: row;">GUI version: <span id="current-version">{cfg.GUI_VERSION}</span><span style="display: none" id="update-available"><a>+</a></span></div>
                        <div>Model version: {cfg.MODEL_VERSION}</div>
                    </div>
                    <div>K. Lisa Yang Center for Conservation Bioacoustics<br>Chemnitz University of Technology</div>
                    <div>For docs and support visit:<br><a href='https://birdnet.cornell.edu/analyzer' target='_blank'>birdnet.cornell.edu/analyzer</a></div>
                </div>
                """
            )

    def build_single_analysis_tab():
        with gr.Tab("Single file"):
            audio_input = gr.Audio(type="filepath", label="file", sources=["upload"])
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
                return i["path"] if i else None

            audio_input.change(get_audio_path, inputs=audio_input, outputs=audio_path_state, preprocess=False)

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
                headers=["Start (s)", "End (s)", "Scientific name", "Common name", "Confidence"],
                elem_classes="mh-200",
            )

            single_file_analyze = gr.Button("Analyze")

            single_file_analyze.click(runSingleFileAnalysis, inputs=inputs, outputs=output_dataframe)

    def build_multi_analysis_tab():
        with gr.Tab("Multiple files"):
            input_directory_state = gr.State()
            output_directory_predict_state = gr.State()

            with gr.Row():
                with gr.Column():
                    select_directory_btn = gr.Button("Select directory (recursive)")
                    directory_input = gr.Matrix(interactive=False, elem_classes="mh-200", headers=["Subpath", "Length"])

                    def select_directory_on_empty():
                        res = select_directory()

                        if res[1]:
                            if len(res[1]) > 100:
                                return [res[0], res[1][:100] + [["..."]]]  # hopefully fixes issue#272

                            return res

                        return [res[0], [["No files found"]]]

                    select_directory_btn.click(
                        select_directory_on_empty, outputs=[input_directory_state, directory_input], show_progress=False
                    )

                with gr.Column():
                    select_out_directory_btn = gr.Button("Select output directory.")
                    selected_out_textbox = gr.Textbox(
                        label="Output directory",
                        interactive=False,
                        placeholder="If not selected, the input directory will be used.",
                    )

                    def select_directory_wrapper():
                        return (select_directory(collect_files=False),) * 2

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

            with gr.Accordion("Output type", open=True):

                output_type_radio = gr.Radio(
                    list(OUTPUT_TYPE_MAP.keys()),
                    value="Raven selection table",
                    label="Result type",
                    info="Specifies output format.",
                )

                with gr.Row():
                    with gr.Column():
                        combine_tables_checkbox = gr.Checkbox(
                            False,
                            label="Combine selection tables",
                            info="If checked, all selection tables are combined into one.",
                        )

                    with gr.Column(visible=False) as output_filename_col:
                        output_filename = gr.Textbox(
                            "BirdNET_Results_Selection_Table.txt",
                            label="Output filename",
                            info="Name of the combined selection table.",
                        )

                    def on_output_type_change(value, check):
                        return gr.Checkbox(visible=value == "Raven selection table"), gr.Textbox(visible=check)

                    output_type_radio.change(
                        on_output_type_change,
                        inputs=[output_type_radio, combine_tables_checkbox],
                        outputs=[combine_tables_checkbox, output_filename],
                        show_progress=False,
                    )

                    def on_combine_tables_change(value):
                        return gr.Column(visible=value)

                    combine_tables_checkbox.change(
                        on_combine_tables_change,
                        inputs=combine_tables_checkbox,
                        outputs=output_filename_col,
                        show_progress=False,
                    )
                
                with gr.Row():
                    skip_existing_checkbox = gr.Checkbox(False, label="Skip existing results", info="Skip files that already have a result.")

            with gr.Row():
                batch_size_number = gr.Number(
                    precision=1, label="Batch size", value=1, info="Number of samples to process at the same time."
                )
                threads_number = gr.Number(precision=1, label="Threads", value=4, info="Number of CPU threads.")

            locale_radio = locale()

            start_batch_analysis_btn = gr.Button("Analyze")

            result_grid = gr.Matrix(headers=["File", "Execution"], elem_classes="mh-200")

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
                output_filename,
                combine_tables_checkbox,
                locale_radio,
                batch_size_number,
                threads_number,
                input_directory_state,
                skip_existing_checkbox,
            ]

            start_batch_analysis_btn.click(runBatchAnalysis, inputs=inputs, outputs=result_grid)

    def build_train_tab():
        with gr.Tab("Train"):
            input_directory_state = gr.State()
            output_directory_state = gr.State()

            with gr.Row():
                with gr.Column():
                    select_directory_btn = gr.Button("Training data")
                    directory_input = gr.List(headers=["Classes"], interactive=False, elem_classes="mh-200")
                    select_directory_btn.click(
                        select_subdirectories, outputs=[input_directory_state, directory_input], show_progress=False
                    )

                with gr.Column():
                    select_directory_btn = gr.Button("Classifier output")

                    with gr.Column():
                        classifier_name = gr.Textbox(
                            "CustomClassifier",
                            visible=False,
                            info="The name of the new classifier.",
                        )
                        output_format = gr.Radio(
                            ["tflite", "raven", "both"],
                            value="tflite",
                            label="Model output format",
                            info="Format for the trained classifier.",
                            visible=False,
                        )

                    def select_directory_and_update_tb():
                        dir_name = _WINDOW.create_file_dialog(webview.FOLDER_DIALOG)

                        if dir_name:
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
                False, label="Use autotune", info="Searches best params, this will take more time."
            )

            with gr.Column(visible=False) as autotune_params:
                with gr.Row():
                    autotune_trials = gr.Number(
                        50, label="Trials", info="Number of training runs for hyperparameter tuning."
                    )
                    autotune_executions_per_trials = gr.Number(
                        1,
                        label="Executions per trial",
                        info="The number of times a training run with a set of hyperparameters is repeated during hyperparameter tuning (this reduces the variance).",
                    )

            with gr.Column() as custom_params:
                with gr.Row():
                    epoch_number = gr.Number(50, label="Epochs", info="Number of training epochs.")
                    batch_size_number = gr.Number(32, label="Batch size", info="Batch size.")
                    learning_rate_number = gr.Number(0.001, label="Learning rate", info="Learning rate.")

                with gr.Row():
                    upsampling_mode = gr.Radio(
                        ["repeat", "mean", "smote"],
                        value="repeat",
                        label="Upsampling mode",
                        info="Balance data through upsampling.",
                    )
                    upsampling_ratio = gr.Slider(
                        0.0,
                        1.0,
                        0.0,
                        step=0.01,
                        label="Upsampling ratio",
                        info="Balance train data and upsample minority classes.",
                    )

                with gr.Row():
                    hidden_units_number = gr.Number(
                        0,
                        label="Hidden units",
                        info="Number of hidden units. If set to >0, a two-layer classifier is used.",
                    )
                    use_mixup = gr.Checkbox(
                        False, label="Use mixup", info="Whether to use mixup for training.", show_label=True
                    )

            def on_autotune_change(value):
                return gr.Column(visible=not value), gr.Column(visible=value)

            autotune_cb.change(
                on_autotune_change, inputs=autotune_cb, outputs=[custom_params, autotune_params], show_progress=False
            )

            with gr.Row():

                fmin_number = gr.Number(
                    cfg.SIG_FMIN,
                    label="Minimum bandpass frequency in Hz.",
                    info="Make sure that you apply the same frequency cut-off for inference.",
                )

                fmax_number = gr.Number(
                    cfg.SIG_FMAX,
                    label="Maximum bandpass frequency in Hz.",
                    info="Make sure that you apply the same frequency cut-off for inference.",
                )

            with gr.Row():
                crop_mode = gr.Radio(
                    ["center", "first", "segments"],
                    value="center",
                    label="Crop mode",
                    info="Crop mode for training data.",
                )
                crop_overlap = gr.Number(
                    0.0, label="Crop overlap", info="Overlap of training data segments", visible=False
                )

                def on_crop_select(new_crop_mode):
                    return gr.Number(visible=new_crop_mode == "segments", interactive=new_crop_mode == "segments")

                crop_mode.change(on_crop_select, inputs=crop_mode, outputs=crop_overlap)

            model_save_mode = gr.Radio(
                ["replace", "append"],
                value="replace",
                label="Model save mode",
                info="'replace' will overwrite the original classification layer and 'append' will combine the original classification layer with the new one.",
            )

            with gr.Row():
                cache_file_state = gr.State()
                cache_mode = gr.Radio(
                    ["none", "load", "save"], value="none", label="Cache mode", info="Cache mode for training files."
                )
                with gr.Column(visible=False) as new_cache_file_row:
                    select_cache_file_directory_btn = gr.Button("Cache file directory")

                    with gr.Column():
                        cache_file_name = gr.Textbox(
                            "train_cache.npz",
                            visible=False,
                            info="The name of the cache_file.",
                        )

                    def select_directory_and_update():
                        dir_name = _WINDOW.create_file_dialog(webview.FOLDER_DIALOG)

                        if dir_name:
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
                    selected_cache_file_btn = gr.Button("Select cache file")
                    cache_file_input = gr.File(file_types=[".npz"], visible=False, interactive=False)

                    def on_cache_file_selection_click():
                        file = select_file(("NPZ file (*.npz)",))

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

            start_training_button = gr.Button("Start training")

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
        with gr.Tab("Segments"):
            audio_directory_state = gr.State()
            result_directory_state = gr.State()
            output_directory_state = gr.State()

            def select_directory_to_state_and_tb():
                return (select_directory(collect_files=False),) * 2

            with gr.Row():
                select_audio_directory_btn = gr.Button("Select audio directory (recursive)")
                selected_audio_directory_tb = gr.Textbox(show_label=False, interactive=False)
                select_audio_directory_btn.click(
                    select_directory_to_state_and_tb,
                    outputs=[selected_audio_directory_tb, audio_directory_state],
                    show_progress=False,
                )

            with gr.Row():
                select_result_directory_btn = gr.Button("Select result directory")
                selected_result_directory_tb = gr.Textbox(
                    show_label=False, interactive=False, placeholder="Same as audio directory if not selected"
                )
                select_result_directory_btn.click(
                    select_directory_to_state_and_tb,
                    outputs=[result_directory_state, selected_result_directory_tb],
                    show_progress=False,
                )

            with gr.Row():
                select_output_directory_btn = gr.Button("Select output directory")
                selected_output_directory_tb = gr.Textbox(
                    show_label=False, interactive=False, placeholder="Same as audio directory if not selected"
                )
                select_output_directory_btn.click(
                    select_directory_to_state_and_tb,
                    outputs=[selected_output_directory_tb, output_directory_state],
                    show_progress=False,
                )

            min_conf_slider = gr.Slider(
                minimum=0.1, maximum=0.99, step=0.01, label="Minimum confidence", info="Minimum confidence threshold."
            )
            num_seq_number = gr.Number(
                100, label="Max number of segments", info="Maximum number of randomly extracted segments per species."
            )
            seq_length_number = gr.Number(3.0, label="Sequence length", info="Length of extracted segments in seconds.")
            threads_number = gr.Number(4, label="Threads", info="Number of CPU threads.")

            extract_segments_btn = gr.Button("Extract segments")

            result_grid = gr.Matrix(headers=["File", "Execution"], elem_classes="mh-200")

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

    def build_species_tab():
        with gr.Tab("Species"):
            output_directory_state = gr.State()
            select_directory_btn = gr.Button("Output directory")

            classifier_name = gr.Textbox(
                "species_list.txt",
                visible=False,
                info="Name of the file, if not specified 'species_list.txt' will be used.",
            )

            def select_directory_and_update_tb():
                dir_name = _WINDOW.create_file_dialog(webview.FOLDER_DIALOG)

                if dir_name:
                    return (
                        dir_name[0],
                        gr.Textbox(label=dir_name[0] + "\\", visible=True),
                    )

                return None, None

            select_directory_btn.click(
                select_directory_and_update_tb,
                outputs=[output_directory_state, classifier_name],
                show_progress=False,
            )

            lat_number, lon_number, week_number, sf_thresh_number, yearlong_checkbox = species_list_coordinates()

            sortby = gr.Radio(
                ["freq", "alpha"],
                value="freq",
                label="Sort by",
                info="Sort species by occurrence frequency or alphabetically.",
            )

            start_btn = gr.Button("Generate species list")
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

    with gr.Blocks(
        css="gui/gui.css",
        js="gui/gui.js",
        theme=gr.themes.Default(),
        analytics_enabled=False,
    ) as demo:
        build_header()
        build_single_analysis_tab()
        build_multi_analysis_tab()
        build_train_tab()
        build_segments_tab()
        build_species_tab()
        build_footer()

    url = demo.queue(api_open=False).launch(prevent_thread_lock=True, quiet=True)[1]
    _WINDOW = webview.create_window("BirdNET-Analyzer", url.rstrip("/") + "?__theme=light", min_size=(1024, 768))

    webview.start(private_mode=False)
