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
    species_file,
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
    locale,
):
    validate(input_path, "Please select a file.")

    return runAnalysis(
        input_path,
        species_file,
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
        "csv",
        "en" if not locale else locale,
        1,
        4,
        None,
        progress=None,
    )


def runBatchAnalysis(
    species_file,
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
        species_file,
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
        output_type,
        "en" if not locale else locale,
        batch_size if batch_size and batch_size > 0 else 1,
        threads if threads and threads > 0 else 4,
        input_dir,
        progress,
    )


def runAnalysis(
    input_path,
    species_file,
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

    if len(cfg.SPECIES_LIST) == 0:
        print("Species list contains {} species".format(len(cfg.LABELS)))
    else:
        print("Species list contains {} species".format(len(cfg.SPECIES_LIST)))

    # Generate species list
    cfg.SPECIES_LIST_FILE = species_file.name if species_file and species_file.name else None
    cfg.SPECIES_LIST = analyze.loadSpeciesList(cfg.SPECIES_LIST_FILE)

    # Set input and output path
    cfg.INPUT_PATH = input_path

    if input_dir:
        cfg.OUTPUT_PATH = input_dir[0]
    else:
        cfg.OUTPUT_PATH = input_path.split(".", 1)[0] + ".csv"

    # Parse input files
    if input_dir:
        cfg.FILE_LIST = [
            str(p.resolve()) for p in Path(input_dir[0]).glob("*") if p.suffix in {".wav", ".flac", ".mp3", ".ogg", ".m4a"}
        ]
        cfg.INPUT_PATH = input_dir[0]
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


def show_species_choice(choice: str):
    if choice == _CUSTOM_SPECIES:
        return [gr.Row.update(visible=False), gr.File.update(visible=True), gr.Column.update(visible=False)]
    elif choice == _PREDICT_SPECIES:
        return [gr.Row.update(visible=True), gr.File.update(visible=False), gr.Column.update(visible=False)]

    return [gr.Row.update(visible=False), gr.File.update(visible=False), gr.Column.update(visible=True)]


def select_directory():
    dir_name = _WINDOW.create_file_dialog(webview.FOLDER_DIALOG)

    if dir_name:
        files = [
            str(p.resolve()) for p in Path(dir_name[0]).glob("**/*") if p.suffix in {".wav", ".flac", ".mp3", ".ogg", ".m4a"}
        ]
        return dir_name, files

    return None, None


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
                    [_CUSTOM_SPECIES, _PREDICT_SPECIES, "all species"],
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

                species_list_radio.change(
                    show_species_choice,
                    inputs=[species_list_radio],
                    outputs=[position_row, species_file_input, empty_col],
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
                )

    with gr.Blocks(
        css=r".d-block .wrap {display: block !important;} .mh-200 {max-height: 300px; overflow-y: auto !important;} footer {display: none !important;}",
        theme=gr.themes.Default(),
    ) as demo:
        with gr.Tab("Single file"):
            audio_input = gr.Audio(type="filepath", label="file")

            confidence_slider, sensitivity_slider, overlap_slider = sample_sliders(False)
            (
                species_list_radio,
                species_file_input,
                lat_number,
                lon_number,
                week_number,
                sf_thresh_number,
                yearlong_checkbox,
            ) = species_lists(False)
            locale_radio = locale()

            inputs = [
                audio_input,
                species_file_input,
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

            with gr.Column() as directory_selection_column:
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
                species_file_input,
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
                output_type_radio,
                locale_radio,
                batch_size_number,
                threads_number,
                input_directory_state,
            ]

            start_batch_analysis_btn.click(runBatchAnalysis, inputs=inputs, outputs=result_grid)

    api, url, _ = demo.launch(server_port=4200, prevent_thread_lock=True, enable_queue=True)
    _WINDOW = webview.create_window("BirdNET-Analyzer", url.rstrip("/") + "?__theme=light", min_size=(500, 500))

    webview.start(private_mode=False)
