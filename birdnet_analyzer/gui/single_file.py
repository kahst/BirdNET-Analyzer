import os

import gradio as gr

import birdnet_analyzer.localization as loc
import birdnet_analyzer.gui.utils as gu
import birdnet_analyzer.gui.analysis as ga


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
    import csv
    from datetime import timedelta

    gu.validate(input_path, loc.localize("validation-no-file-selected"))

    result_filepath = ga.runAnalysis(
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

    # read the result file to return the data to be displayed.
    with open(result_filepath, "r", encoding="utf-8") as f:
        reader = csv.reader(f)
        data = list(reader)
        data = [l[0:-1] for l in data[1:]]  # remove last column (file path) and first row (header)

        for row in data:
            for col_idx in range(2):
                seconds = float(row[col_idx])
                time_str = str(timedelta(seconds=seconds))
                row[col_idx] = time_str

    return data


def build_single_analysis_tab():
    with gr.Tab(loc.localize("single-tab-title")):
        audio_input = gr.Audio(type="filepath", label=loc.localize("single-audio-label"), sources=["upload"])
        audio_path_state = gr.State()

        confidence_slider, sensitivity_slider, overlap_slider, fmin_number, fmax_number = gu.sample_sliders(False)

        (
            species_list_radio,
            species_file_input,
            lat_number,
            lon_number,
            week_number,
            sf_thresh_number,
            yearlong_checkbox,
            selected_classifier_state,
        ) = gu.species_lists(False)
        locale_radio = gu.locale()

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


if __name__ == "__main__":
    gu.open_window(build_single_analysis_tab)