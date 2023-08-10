from __future__ import annotations

import concurrent.futures
import os
import sys
from multiprocessing import freeze_support
from pathlib import Path

import gradio as gr
import librosa
import webview

from birdnet._paths import ROOT_PATH
from birdnet.analysis.codes_loading import load_codes
from birdnet.configuration import config
from birdnet.embeddings.file_analysing import analyze_file
from birdnet.segments.folders_parsing import parse_folders
from birdnet.species.species_list_getting import get_species_list
from birdnet.utils.audio_file_collecting import collect_audio_files
from birdnet.utils.lines_reading import read_lines
from birdnet.segments.files_parsing import parse_files
from birdnet.segments.segments_extracting import extract_segments
from birdnet.train.train_model import train_model
from birdnet.utils.subdirectories_listing import list_subdirectories

_WINDOW: webview.Window
OUTPUT_TYPE_MAP = {"Raven selection table": "table", "Audacity": "audacity", "R": "r", "CSV": "csv"}
ORIGINAL_MODEL_PATH = config.MODEL_PATH
ORIGINAL_MDATA_MODEL_PATH = config.MDATA_MODEL_PATH
ORIGINAL_LABELS_FILE = config.LABELS_FILE
ORIGINAL_TRANSLATED_LABELS_PATH = config.TRANSLATED_LABELS_PATH


def analyze_file_wrapper(entry):
    return (entry[0], analyze_file(entry))


def extract_segments_wrapper(entry):
    return (entry[0][0], extract_segments(entry))


def validate(value, msg):
    """Checks if the value ist not falsy.

    If the value is falsy, an error will be raised.

    Args:
        value: Value to be tested.
        msg: Message in case of an error.
    """
    if not value:
        raise gr.Error(msg)


def run_single_file_analysis(
    input_path,
    confidence,
    sensitivity,
    overlap,
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

    return run_analysis(
        input_path,
        None,
        confidence,
        sensitivity,
        overlap,
        species_list_choice,
        species_list_file,
        lat,
        lon,
        week,
        use_yearlong,
        sf_thresh,
        custom_classifier_file,
        "csv",
        "en" if not locale else locale,
        1,
        4,
        None,
        progress=None,
    )


def run_batch_analysis(
    output_path,
    confidence,
    sensitivity,
    overlap,
    species_list_choice,
    species_list_file,
    lat,
    lon,
    week,
    use_yearlong,
    sf_thresh,
    custom_classifier_file,
    output_type,
    locale,
    batch_size,
    threads,
    input_dir,
    progress=gr.Progress(),
):
    validate(input_dir, "Please select a directory.")
    batch_size = int(batch_size)
    threads = int(threads)

    if species_list_choice == _CUSTOM_SPECIES:
        validate(species_list_file, "Please select a species list.")

    return run_analysis(
        None,
        output_path,
        confidence,
        sensitivity,
        overlap,
        species_list_choice,
        species_list_file,
        lat,
        lon,
        week,
        use_yearlong,
        sf_thresh,
        custom_classifier_file,
        output_type,
        "en" if not locale else locale,
        batch_size if batch_size and batch_size > 0 else 1,
        threads if threads and threads > 0 else 4,
        input_dir,
        progress,
    )


def run_analysis(
    input_path: str,
    output_path: str | None,
    confidence: float,
    sensitivity: float,
    overlap: float,
    species_list_choice: str,
    species_list_file,
    lat: float,
    lon: float,
    week: int,
    use_yearlong: bool,
    sf_thresh: float,
    custom_classifier_file,
    output_type: str,
    locale: str,
    batch_size: int,
    threads: int,
    input_dir: str,
    progress: gr.Progress | None,
):
    """Starts the analysis.

    Args:
        input_path: Either a file or directory.
        output_path: The output path for the result, if None the input_path is used
        confidence: The selected minimum confidence.
        sensitivity: The selected sensitivity.
        overlap: The selected segment overlap.
        species_list_choice: The choice for the species list.
        species_list_file: The selected custom species list file.
        lat: The selected latitude.
        lon: The selected longitude.
        week: The selected week of the year.
        use_yearlong: Use yearlong instead of week.
        sf_thresh: The threshold for the predicted species list.
        custom_classifier_file: Custom classifier to be used.
        output_type: The type of result to be generated.
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
    config.CODES = load_codes()
    config.LABELS = read_lines(ORIGINAL_LABELS_FILE)
    config.LATITUDE, config.LONGITUDE, config.WEEK = lat, lon, -1 if use_yearlong else week
    config.LOCATION_FILTER_THRESHOLD = sf_thresh

    if species_list_choice == _CUSTOM_SPECIES:
        if not species_list_file or not species_list_file.name:
            config.SPECIES_LIST_FILE = None
        else:
            config.SPECIES_LIST_FILE = os.path.join(os.path.dirname(os.path.abspath(sys.argv[0])), species_list_file.name)

            if os.path.isdir(config.SPECIES_LIST_FILE):
                config.SPECIES_LIST_FILE = os.path.join(config.SPECIES_LIST_FILE, "species_list.txt")

        config.SPECIES_LIST = read_lines(config.SPECIES_LIST_FILE)
        config.CUSTOM_CLASSIFIER = None
    elif species_list_choice == _PREDICT_SPECIES:
        config.SPECIES_LIST_FILE = None
        config.CUSTOM_CLASSIFIER = None
        config.SPECIES_LIST = get_species_list(config.LATITUDE, config.LONGITUDE, config.WEEK, config.LOCATION_FILTER_THRESHOLD)
    elif species_list_choice == _CUSTOM_CLASSIFIER:
        if custom_classifier_file is None:
            raise gr.Error("No custom classifier selected.")

        # Set custom classifier?
        config.CUSTOM_CLASSIFIER = custom_classifier_file  # we treat this as absolute path, so no need to join with dirname
        config.LABELS_FILE = custom_classifier_file.replace(".tflite", "_Labels.txt")  # same for labels file
        config.LABELS = read_lines(config.LABELS_FILE)
        config.LATITUDE = -1
        config.LONGITUDE = -1
        config.SPECIES_LIST_FILE = None
        config.SPECIES_LIST = []
        locale = "en"
    else:
        config.SPECIES_LIST_FILE = None
        config.SPECIES_LIST = []
        config.CUSTOM_CLASSIFIER = None

    # Load translated labels
    lfile = os.path.join(config.TRANSLATED_LABELS_PATH, os.path.basename(config.LABELS_FILE).replace(".txt", f"_{locale}.txt"))
    if not locale in ["en"] and os.path.isfile(lfile):
        config.TRANSLATED_LABELS = read_lines(lfile)
    else:
        config.TRANSLATED_LABELS = config.LABELS

    if len(config.SPECIES_LIST) == 0:
        print(f"Species list contains {len(config.LABELS)} species")
    else:
        print(f"Species list contains {len(config.SPECIES_LIST)} species")

    # Set input and output path
    config.INPUT_PATH = input_path

    if input_dir:
        config.OUTPUT_PATH = output_path if output_path else input_dir
    else:
        config.OUTPUT_PATH = output_path if output_path else input_path.split(".", 1)[0] + ".csv"

    # Parse input files
    if input_dir:
        config.FILE_LIST = collect_audio_files(input_dir)
        config.INPUT_PATH = input_dir
    elif os.path.isdir(config.INPUT_PATH):
        config.FILE_LIST = collect_audio_files(config.INPUT_PATH)
    else:
        config.FILE_LIST = [config.INPUT_PATH]

    validate(config.FILE_LIST, "No audio files found.")

    # Set confidence threshold
    config.MIN_CONFIDENCE = confidence

    # Set sensitivity
    config.SIGMOID_SENSITIVITY = sensitivity

    # Set overlap
    config.SIG_OVERLAP = overlap

    # Set result type
    config.RESULT_TYPE = OUTPUT_TYPE_MAP[output_type] if output_type in OUTPUT_TYPE_MAP else output_type.lower()

    if not config.RESULT_TYPE in ["table", "audacity", "r", "csv"]:
        config.RESULT_TYPE = "table"

    # Set number of threads
    if input_dir:
        config.CPU_THREADS = max(1, int(threads))
        config.TFLITE_THREADS = 1
    else:
        config.CPU_THREADS = 1
        config.TFLITE_THREADS = max(1, int(threads))

    # Set batch size
    config.BATCH_SIZE = max(1, int(batch_size))

    flist = []

    for f in config.FILE_LIST:
        flist.append((f, config.get_config()))

    result_list = []

    if progress is not None:
        progress(0, desc="Starting ...")

    # Analyze files
    if config.CPU_THREADS < 2:
        for entry in flist:
            result = analyze_file_wrapper(entry)

            result_list.append(result)
    else:
        with concurrent.futures.ProcessPoolExecutor(max_workers=config.CPU_THREADS) as executor:
            futures = (executor.submit(analyze_file_wrapper, arg) for arg in flist)
            for i, f in enumerate(concurrent.futures.as_completed(futures), start=1):
                if progress is not None:
                    progress((i, len(flist)), total=len(flist), unit="files")
                result = f.result()

                result_list.append(result)

    result_list_or_output_path = [
        [
            os.path.relpath(result[0], input_dir),
            result[1],
        ]
        for result in result_list] \
        if input_dir else config.OUTPUT_PATH

    return result_list_or_output_path


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
            gr.Row.update(visible=False),
            gr.File.update(visible=True),
            gr.Column.update(visible=False),
            gr.Column.update(visible=False),
        ]
    elif choice == _PREDICT_SPECIES:
        return [
            gr.Row.update(visible=True),
            gr.File.update(visible=False),
            gr.Column.update(visible=False),
            gr.Column.update(visible=False),
        ]
    elif choice == _CUSTOM_CLASSIFIER:
        return [
            gr.Row.update(visible=False),
            gr.File.update(visible=False),
            gr.Column.update(visible=True),
            gr.Column.update(visible=False),
        ]

    return [
        gr.Row.update(visible=False),
        gr.File.update(visible=False),
        gr.Column.update(visible=False),
        gr.Column.update(visible=True),
    ]


def select_subdirectories():
    """Creates a directory selection dialog.

    Returns:
        A tuples of (directory, list of subdirectories) or (None, None) if the dialog was canceled.
    """
    dir_name = _WINDOW.create_file_dialog(webview.FOLDER_DIALOG)

    if dir_name:
        subdirs = list_subdirectories(dir_name[0])

        return dir_name[0], [[d] for d in subdirs]

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

    return "{:2.0f}:{:02.0f}:{:06.3f}".format(hours, minutes, secs)


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

        files = collect_audio_files(dir_name[0])

        return dir_name[0], [
            [os.path.relpath(file, dir_name[0]), format_seconds(librosa.get_duration(filename=file))] for file in files
        ]

    return dir_name[0] if dir_name else None


def start_training(
    data_dir, output_dir, classifier_name, epochs, batch_size, learning_rate, hidden_units, progress=gr.Progress()
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

    if not hidden_units or hidden_units < 0:
        hidden_units = 0

    if progress is not None:
        progress((0, epochs), desc="Loading data & building classifier", unit="epoch")

    if not classifier_name.endswith(".tflite"):
        classifier_name += ".tflite"

    config.TRAIN_DATA_PATH = data_dir
    config.CUSTOM_CLASSIFIER = str(Path(output_dir) / classifier_name)
    config.TRAIN_EPOCHS = int(epochs)
    config.TRAIN_BATCH_SIZE = int(batch_size)
    config.TRAIN_LEARNING_RATE = learning_rate
    config.TRAIN_HIDDEN_UNITS = int(hidden_units)

    def progression(epoch, logs=None):
        if progress is not None:
            if epoch + 1 == epochs:
                progress((epoch + 1, epochs), total=epochs, unit="epoch", desc=f"Saving at {config.CUSTOM_CLASSIFIER}")
            else:
                progress((epoch + 1, epochs), total=epochs, unit="epoch")

    history = train_model(on_epoch_end=progression)

    precision = history.history["val_prec"]

    import matplotlib.pyplot as plt

    fig = plt.figure()
    plt.plot(precision)
    plt.ylabel("Precision")
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
    config.FILE_LIST = parse_folders(audio_dir, result_dir)

    # Set output folder
    config.OUTPUT_PATH = output_dir

    # Set number of threads
    config.CPU_THREADS = int(threads)

    # Set confidence threshold
    config.MIN_CONFIDENCE = max(0.01, min(0.99, min_conf))

    # Parse file list and make list of segments
    config.FILE_LIST = parse_files(config.FILE_LIST, max(1, int(num_seq)))

    # Add config items to each file list entry.
    # We have to do this for Windows which does not
    # support fork() and thus each process has to
    # have its own config. USE LINUX!
    flist = [(entry, max(config.SIG_LENGTH, float(seq_length)), config.get_config()) for entry in config.FILE_LIST]

    result_list = []

    # Extract segments
    if config.CPU_THREADS < 2:
        for i, entry in enumerate(flist):
            result = extract_segments_wrapper(entry)
            result_list.append(result)

            if progress is not None:
                progress((i, len(flist)), total=len(flist), unit="files")
    else:
        with concurrent.futures.ProcessPoolExecutor(max_workers=config.CPU_THREADS) as executor:
            futures = (executor.submit(extract_segments_wrapper, arg) for arg in flist)
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
                minimum=0, maximum=1, value=0.5, step=0.01, label="Minimum Confidence", info="Minimum confidence threshold."
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

        return confidence_slider, sensitivity_slider, overlap_slider


def locale():
    """Creates the gradio elements for locale selection

    Reads the translated labels inside the checkpoints directory.

    Returns:
        The dropdown element.
    """
    label_files = os.listdir(ROOT_PATH / ORIGINAL_TRANSLATED_LABELS_PATH)
    options = ["EN"] + [label_file.rsplit("_", 1)[-1].split(".")[0].upper() for label_file in label_files]

    return gr.Dropdown(options, value="EN", label="Locale", info="Locale for the translated species common names.")


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
                [_CUSTOM_SPECIES, _PREDICT_SPECIES, _CUSTOM_CLASSIFIER, "all species"],
                value="all species",
                label="Species list",
                info="List of all possible species",
                elem_classes="d-block",
            )

            with gr.Column(visible=False) as position_row:
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
                        return gr.Slider.update(interactive=(not use_yearlong))

                    yearlong_checkbox.change(onChange, inputs=yearlong_checkbox, outputs=week_number, show_progress=False)
                sf_thresh_number = gr.Slider(
                    minimum=0.01,
                    maximum=0.99,
                    value=0.03,
                    step=0.01,
                    label="Location filter threshold",
                    info="Minimum species occurrence frequency threshold for location filter.",
                )

            species_file_input = gr.File(file_types=[".txt"], info="Path to species list file or folder.", visible=False)
            empty_col = gr.Column()

            with gr.Column(visible=False) as custom_classifier_selector:
                classifier_selection_button = gr.Button("Select classifier")
                classifier_file_input = gr.Files(
                    file_types=[".tflite"], info="Path to the custom classifier.", visible=False, interactive=False
                )
                selected_classifier_state = gr.State()

                def on_custom_classifier_selection_click():
                    file = select_file(("TFLite classifier (*.tflite)",))

                    if file:
                        labels = os.path.splitext(file)[0] + "_Labels.txt"

                        return file, gr.File.update(value=[file, labels], visible=True)

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
    freeze_support()

    def build_single_analysis_tab():
        with gr.Tab("Single file"):
            audio_input = gr.Audio(type="filepath", label="file", elem_id="single_file_audio")

            confidence_slider, sensitivity_slider, overlap_slider = sample_sliders(False)
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

            inputs = [
                audio_input,
                confidence_slider,
                sensitivity_slider,
                overlap_slider,
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

            single_file_analyze.click(run_single_file_analysis, inputs=inputs, outputs=output_dataframe)

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

                        return res if res[1] else [res[0], [["No files found"]]]

                    select_directory_btn.click(
                        select_directory_on_empty, outputs=[input_directory_state, directory_input], show_progress=True
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

            confidence_slider, sensitivity_slider, overlap_slider = sample_sliders()

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

            output_type_radio = gr.Radio(
                list(OUTPUT_TYPE_MAP.keys()),
                value="Raven selection table",
                label="Result type",
                info="Specifies output format.",
            )

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
                species_list_radio,
                species_file_input,
                lat_number,
                lon_number,
                week_number,
                yearlong_checkbox,
                sf_thresh_number,
                selected_classifier_state,
                output_type_radio,
                locale_radio,
                batch_size_number,
                threads_number,
                input_directory_state,
            ]

            start_batch_analysis_btn.click(run_batch_analysis, inputs=inputs, outputs=result_grid)

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

                    with gr.Row():
                        classifier_name = gr.Textbox(
                            "CustomClassifier.tflite",
                            visible=False,
                            info="The filename of the new classifier.",
                        )

                    def select_directory_and_update_tb():
                        dir_name = _WINDOW.create_file_dialog(webview.FOLDER_DIALOG)

                        if dir_name:
                            return dir_name[0], gr.Textbox.update(label=dir_name[0] + "\\", visible=True)

                        return None, None

                    select_directory_btn.click(
                        select_directory_and_update_tb, outputs=[output_directory_state, classifier_name], show_progress=False
                    )

            with gr.Row():
                epoch_number = gr.Number(100, label="Epochs", info="Number of training epochs.")
                batch_size_number = gr.Number(32, label="Batch size", info="Batch size.")
                learning_rate_number = gr.Number(0.01, label="Learning rate", info="Learning rate.")

            hidden_units_number = gr.Number(
                0, label="Hidden units", info="Number of hidden units. If set to >0, a two-layer classifier is used."
            )

            train_history_plot = gr.Plot()

            start_training_button = gr.Button("Start training")

            start_training_button.click(
                start_training,
                inputs=[
                    input_directory_state,
                    output_directory_state,
                    classifier_name,
                    epoch_number,
                    batch_size_number,
                    learning_rate_number,
                    hidden_units_number,
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

    with gr.Blocks(
        css=r".d-block .wrap {display: block !important;} .mh-200 {max-height: 300px; overflow-y: auto !important;} footer {display: none !important;} #single_file_audio, #single_file_audio * {max-height: 81.6px; min-height: 0;}",
        theme=gr.themes.Default(),
        analytics_enabled=False,
    ) as demo:
        build_single_analysis_tab()
        build_multi_analysis_tab()
        build_train_tab()
        build_segments_tab()

    url = demo.queue(api_open=False).launch(prevent_thread_lock=True, quiet=True)[1]
    _WINDOW = webview.create_window("BirdNET-Analyzer", url.rstrip("/") + "?__theme=light", min_size=(1024, 768))

    webview.start(private_mode=False)
