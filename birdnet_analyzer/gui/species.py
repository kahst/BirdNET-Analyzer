import os
import gradio as gr
import webview

import birdnet_analyzer.localization as loc
import birdnet_analyzer.gui.utils as gu
import birdnet_analyzer.species as species
import birdnet_analyzer.config as cfg


def run_species_list(out_path, filename, lat, lon, week, use_yearlong, sf_thresh, sortby):
    gu.validate(out_path, loc.localize("validation-no-directory-selected"))

    species.run(
        os.path.join(out_path, filename if filename else "species_list.txt"),
        lat,
        lon,
        -1 if use_yearlong else week,
        sf_thresh,
        sortby,
    )

    gr.Info(f"{loc.localize('species-tab-finish-info')} {cfg.OUTPUT_PATH}")


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
            dir_name = gu._WINDOW.create_file_dialog(webview.FOLDER_DIALOG, directory=initial_dir)

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

        lat_number, lon_number, week_number, sf_thresh_number, yearlong_checkbox = gu.species_list_coordinates()

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
