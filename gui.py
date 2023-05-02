import gradio as gr
import analyze
import config as cfg
import os
import concurrent.futures
from multiprocessing import freeze_support
import webview
import model
import sys
from pathlib import Path
from train import trainModel

_WINDOW: webview.Window = None
OUTPUT_TYPE_MAP = {"Raven selection table": "table", "Audacity": "audacity", "R": "r", "CSV": "csv"}


def analyzeFile_wrapper(entry):
    return (entry[0], analyze.analyzeFile(entry))


def loadSpeciesList(fpath):
    slist = []

    if not fpath == None:
        with open(fpath, "r", encoding="utf-8") as sfile:
            for line in sfile.readlines():
                species = line.replace("\r", "").replace("\n", "")
                slist.append(species)

    return slist


def predictSpeciesList():
    l_filter = model.explore(cfg.LATITUDE, cfg.LONGITUDE, cfg.WEEK)
    cfg.SPECIES_LIST_FILE = None
    cfg.SPECIES_LIST = []

    for s in l_filter:
        if s[0] >= cfg.LOCATION_FILTER_THRESHOLD:
            cfg.SPECIES_LIST.append(s[1])


def validate(value, msg):
    if not value:
        raise gr.Error(msg)


def runSingleFileAnalysis(
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

    return runAnalysis(
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
        "csv",
        "en" if not locale else locale,
        1,
        4,
        None,
        progress=None,
    )


def runBatchAnalysis(
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

    return runAnalysis(
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
        output_type,
        "en" if not locale else locale,
        batch_size if batch_size and batch_size > 0 else 1,
        threads if threads and threads > 0 else 4,
        input_dir,
        progress,
    )


def runAnalysis(
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
    output_type,
    locale,
    batch_size,
    threads,
    input_dir,
    progress,
):
    if progress is not None:
        progress(0, desc="Preparing ...")

    locale = locale.lower()
    # Load eBird codes, labels
    cfg.CODES = analyze.loadCodes()
    cfg.LABELS = analyze.loadLabels(cfg.LABELS_FILE)
    # Load translated labels
    lfile = os.path.join(
        cfg.TRANSLATED_LABELS_PATH, os.path.basename(cfg.LABELS_FILE).replace(".txt", "_{}.txt".format(locale))
    )
    if not locale in ["en"] and os.path.isfile(lfile):
        cfg.TRANSLATED_LABELS = analyze.loadLabels(lfile)
    else:
        cfg.TRANSLATED_LABELS = cfg.LABELS

    cfg.LATITUDE, cfg.LONGITUDE, cfg.WEEK = lat, lon, -1 if use_yearlong else week
    cfg.LOCATION_FILTER_THRESHOLD = sf_thresh

    if species_list_choice == _CUSTOM_SPECIES:
        if not species_list_file or not species_list_file.name:
            cfg.SPECIES_LIST_FILE = None
        else:
            cfg.SPECIES_LIST_FILE = os.path.join(os.path.dirname(os.path.abspath(sys.argv[0])), species_list_file.name)

            if os.path.isdir(cfg.SPECIES_LIST_FILE):
                cfg.SPECIES_LIST_FILE = os.path.join(cfg.SPECIES_LIST_FILE, "species_list.txt")

        cfg.SPECIES_LIST = loadSpeciesList(cfg.SPECIES_LIST_FILE)
    elif species_list_choice == _PREDICT_SPECIES:
        predictSpeciesList()
    elif species_list_choice == _CUSTOM_CLASSIFIER:
        if custom_classifier_file is None:
            raise gr.Error("No custom classifier selected.")

        # Set custom classifier?
        cfg.CUSTOM_CLASSIFIER = custom_classifier_file  # we treat this as absolute path, so no need to join with dirname
        cfg.LABELS_FILE = custom_classifier_file.replace(".tflite", "_Labels.txt")  # same for labels file
        cfg.LABELS = analyze.loadLabels(cfg.LABELS_FILE)
        cfg.LATITUDE = -1
        cfg.LONGITUDE = -1

    if len(cfg.SPECIES_LIST) == 0:
        print("Species list contains {} species".format(len(cfg.LABELS)))
    else:
        print("Species list contains {} species".format(len(cfg.SPECIES_LIST)))

    # Set input and output path
    cfg.INPUT_PATH = input_path

    if input_dir:
        cfg.OUTPUT_PATH = input_dir
    else:
        cfg.OUTPUT_PATH = input_path.split(".", 1)[0] + ".csv"

    # Parse input files
    if input_dir:
        cfg.FILE_LIST = [
            str(p.resolve()) for p in Path(input_dir).glob("*") if p.suffix in {".wav", ".flac", ".mp3", ".ogg", ".m4a"}
        ]
        cfg.INPUT_PATH = input_dir
    elif os.path.isdir(cfg.INPUT_PATH):
        cfg.FILE_LIST = analyze.parseInputFiles(cfg.INPUT_PATH)
    else:
        cfg.FILE_LIST = [cfg.INPUT_PATH]

    # Set confidence threshold
    cfg.MIN_CONFIDENCE = confidence

    # Set sensitivity
    cfg.SIGMOID_SENSITIVITY = sensitivity

    # Set overlap
    cfg.SIG_OVERLAP = overlap

    # Set result type
    cfg.RESULT_TYPE = OUTPUT_TYPE_MAP[output_type] if output_type in OUTPUT_TYPE_MAP else output_type.lower()

    if not cfg.RESULT_TYPE in ["table", "audacity", "r", "csv"]:
        cfg.RESULT_TYPE = "table"

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

    return result_list if input_dir else cfg.OUTPUT_PATH


_CUSTOM_SPECIES = "Custom species list"
_PREDICT_SPECIES = "Species by location"
_CUSTOM_CLASSIFIER = "Custom classifier"


def show_species_choice(choice: str):
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
    dir_name = _WINDOW.create_file_dialog(webview.FOLDER_DIALOG)

    if dir_name:
        subdirs = os.listdir(dir_name[0])

        return dir_name[0], [[d] for d in subdirs]

    return None, None


def select_file(filetypes=()):
    files = _WINDOW.create_file_dialog(webview.OPEN_DIALOG, file_types=filetypes)
    return files[0] if files else None


def select_directory():
    dir_name = _WINDOW.create_file_dialog(webview.FOLDER_DIALOG)

    if dir_name:
        files = [
            str(p.resolve()) for p in Path(dir_name[0]).glob("**/*") if p.suffix in {".wav", ".flac", ".mp3", ".ogg", ".m4a"}
        ]
        return dir_name[0], files

    return None, None


def start_training(
    data_dir, output_dir, classifier_name, epochs, batch_size, learning_rate, hidden_units, progress=gr.Progress()
):
    if not data_dir:
        raise gr.Error("Please select your Training data.")

    if not output_dir:
        raise gr.Error("Please select a directory for the classifier.")

    if not classifier_name:
        raise gr.Error("Please enter a valid name for the classifier.")

    if not epochs or epochs < 0:
        raise gr.Error("Please enter a valid number of epochs.")

    if not batch_size or batch_size < 0:
        raise gr.Error("Please enter a valid batchsize.")

    if not learning_rate or learning_rate < 0:
        raise gr.Error("Please aenter a valid learning rate.")

    if not hidden_units or hidden_units < 0:
        hidden_units = 0

    if progress is not None:
        progress((0, epochs), desc="Loading data & building classifier", unit="epoch")

    if not classifier_name.endswith(".tflite"):
        classifier_name += ".tflite"

    cfg.TRAIN_DATA_PATH = data_dir
    cfg.CUSTOM_CLASSIFIER = str(Path(output_dir) / classifier_name)
    cfg.TRAIN_EPOCHS = int(epochs)
    cfg.TRAIN_BATCH_SIZE = int(batch_size)
    cfg.TRAIN_LEARNING_RATE = learning_rate
    cfg.TRAIN_HIDDEN_UNITS = int(hidden_units)

    def progression(epoch, logs=None):
        if progress is not None:
            if epoch + 1 == epochs:
                progress((epoch + 1, epochs), total=epochs, unit="epoch", desc=f"Saving at {cfg.CUSTOM_CLASSIFIER}")
            else:
                progress((epoch + 1, epochs), total=epochs, unit="epoch")

    history = trainModel(on_epoch_end=progression)

    precision = history.history["val_prec"]

    import matplotlib.pyplot as plt

    fig = plt.figure()
    plt.plot(precision)
    plt.ylabel("Precision")
    plt.xlabel("Epoch")

    return fig


if __name__ == "__main__":
    freeze_support()

    def sample_sliders(opened=True):
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
        label_files = os.listdir(os.path.join(os.path.dirname(sys.argv[0]), cfg.TRANSLATED_LABELS_PATH))
        options = ["EN"] + [label_file.rsplit("_", 1)[-1].split(".")[0].upper() for label_file in label_files]

        # return gr.Radio(options, value="EN", label="Locale", info="Locale for translated species common names.")
        return gr.Dropdown(options, value="EN", label="Locale", info="Locale for the translated species common names.")

    def species_lists(opened=True):
        with gr.Accordion("Species selection", open=opened):
            with gr.Row():
                species_list_radio = gr.Radio(
                    [_CUSTOM_SPECIES, _PREDICT_SPECIES, _CUSTOM_CLASSIFIER, "all species"],
                    value=_CUSTOM_SPECIES,
                    label="Species list",
                    info="List of all possible species",
                    elem_classes="d-block",
                )

                with gr.Column(visible=False) as position_row:
                    lat_number = gr.Slider(
                        minimum=-180, maximum=180, value=0, step=1, label="Latitude", info="Recording location latitude."
                    )
                    lon_number = gr.Slider(
                        minimum=-90, maximum=90, value=0, step=1, label="Longitude", info="Recording location longitude."
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
                        label="Location filter threshhold",
                        info="Minimum species occurrence frequency threshold for location filter.",
                    )

                species_file_input = gr.File(file_types=[".txt"], info="Path to species list file or folder.")

                empty_col = gr.Column(visible=False)

                with gr.Column(visible=False) as costom_classifier_selector:
                    classifier_selection_button = gr.Button("Select classifier")
                    classifier_file_input = gr.File(
                        file_types=[".tflite"], info="Path to the custom classifier.", visible=False, interactive=False
                    )
                    selected_classifier_state = gr.State()

                    def on_custom_classifier_selection_click():
                        file = select_file(("TFLite classifier (*.tflite)",))

                        if file:
                            return file, gr.File.update(value=file, visible=True)

                        return None

                    classifier_selection_button.click(on_custom_classifier_selection_click, outputs=[selected_classifier_state, classifier_file_input], show_progress=False)

                species_list_radio.change(
                    show_species_choice,
                    inputs=[species_list_radio],
                    outputs=[position_row, species_file_input, costom_classifier_selector, empty_col],
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
                    selected_classifier_state
                )

    with gr.Blocks(
        css=r".d-block .wrap {display: block !important;} .mh-200 {max-height: 300px; overflow-y: auto !important;} footer {display: none !important;} #single_file_audio, #single_file_audio > * {max-height: 81.6px}",
        theme=gr.themes.Default(),
    ) as demo:
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
                selected_classifier_state
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

            single_file_analyze.click(runSingleFileAnalysis, inputs=inputs, outputs=output_dataframe)

        with gr.Tab("Multiple files"):
            input_directory_state = gr.State()

            with gr.Column():
                select_directory_btn = gr.Button("Select directory (recursive)")
                directory_input = gr.File(label="directory", file_count="directory", interactive=False, elem_classes="mh-200")
                select_directory_btn.click(
                    select_directory, outputs=[input_directory_state, directory_input], show_progress=False
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
                selected_classifier_state
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

            start_batch_analysis_btn.click(runBatchAnalysis, inputs=inputs, outputs=result_grid)

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

                    def select_directory_wrapper():
                        dir_name = _WINDOW.create_file_dialog(webview.FOLDER_DIALOG)

                        if dir_name:
                            return dir_name[0], gr.Textbox.update(label=dir_name[0] + "\\", visible=True)

                        return None, None

                    select_directory_btn.click(
                        select_directory_wrapper, outputs=[output_directory_state, classifier_name], show_progress=False
                    )

            with gr.Row():
                epich_number = gr.Number(100, label="Epochs", info="Number of training epochs.")
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
                    epich_number,
                    batch_size_number,
                    learning_rate_number,
                    hidden_units_number,
                ],
                outputs=[train_history_plot],
            )

    api, url, _ = demo.launch(server_port=4200, prevent_thread_lock=True, enable_queue=True)
    _WINDOW = webview.create_window("BirdNET-Analyzer", url.rstrip("/") + "?__theme=light", min_size=(500, 500))

    webview.start(private_mode=False)
