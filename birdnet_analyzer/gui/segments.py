from functools import partial
import concurrent.futures
import os

import gradio as gr

import birdnet_analyzer.localization as loc
import birdnet_analyzer.gui.utils as gu
import birdnet_analyzer.config as cfg
import birdnet_analyzer.segments as segments
import birdnet_analyzer.training_data_segments as training_data_segments

def extractSegments_wrapper(entry):
    return (entry[0][0], segments.extractSegments(entry))

def extractTrainingDataSegments_wrapper(entry):
    return (entry[0][0], training_data_segments.extractSegments(entry))

def extract_segments(audio_dir, result_dir, output_dir, min_conf, num_seq, seq_length, threads, training_data_mode, annotation_files_suffix, species_column_name, progress=gr.Progress()):
    gu.validate(audio_dir, loc.localize("validation-no-audio-directory-selected"))

    if training_data_mode:
        gu.validate(annotation_files_suffix, loc.localize("validation-no-annotation-files-suffix"))
        gu.validate(species_column_name, loc.localize("validation-no-species-column-name"))

    if not result_dir:
        result_dir = audio_dir

    if not output_dir:
        output_dir = audio_dir

    if progress is not None:
        progress(0, desc=f"{loc.localize('progress-search')} ...")

    # Add config items to each file list entry.
    # We have to do this for Windows which does not
    # support fork() and thus each process has to
    # have its own config. USE LINUX!
    if training_data_mode:
        try:
            flist = training_data_segments.getFileList(audio_dir, result_dir, output_dir, num_seq, seq_length, threads, annotation_files_suffix, species_column_name)
        except Exception as e:  
            if e.args and len(e.args) > 2:
                raise gr.Error(f"{loc.localize(e.args[1])} {e.args[2]}")
            else:
                raise gr.Error(f"{e}")
        extract_fn = extractTrainingDataSegments_wrapper
    else:
        flist = segments.getFileList(audio_dir, result_dir, output_dir, min_conf, num_seq, seq_length, threads)
        extract_fn = extractSegments_wrapper

    result_list = []

    # Extract segments
    if cfg.CPU_THREADS < 2:
        for i, entry in enumerate(flist):
            result = extract_fn(entry)
            result_list.append(result)

            if progress is not None:
                progress((i, len(flist)), total=len(flist), unit="files")
    else:
        with concurrent.futures.ProcessPoolExecutor(max_workers=cfg.CPU_THREADS) as executor:
            futures = (executor.submit(extract_fn, arg) for arg in flist)
            for i, f in enumerate(concurrent.futures.as_completed(futures), start=1):
                if progress is not None:
                    progress((i, len(flist)), total=len(flist), unit="files")
                result = f.result()

                result_list.append(result)

    return [[os.path.relpath(r[0], audio_dir), r[1]] for r in result_list]


def build_segments_tab():
    with gr.Tab(loc.localize("segments-tab-title")):
        audio_directory_state = gr.State()
        result_directory_state = gr.State()
        output_directory_state = gr.State()

        def select_directory_to_state_and_tb(state_key):
            return (gu.select_directory(collect_files=False, state_key=state_key),) * 2

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
            interactive=True,
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

        with gr.Row():
            training_data_cb = gr.Checkbox(
                False,
                label=loc.localize("segments-training-data-checkbox-label"),
                info=loc.localize("segments-training-data-checkbox-info"),
            )

        annotation_files_suffix_tb = gr.Textbox(
            "annotation",
            label=loc.localize("segments-annotation-files-suffix-label"),
            info=loc.localize("segments-annotation-files-suffix-info"),
            visible=False,
            interactive=True
        )
        species_column_name_tb = gr.Textbox(
            "species",
            label=loc.localize("segments-species-column-name-label"),
            info=loc.localize("segments-species-column-name-info"),
            visible=False,
            interactive=True
        )

        extract_segments_btn = gr.Button(loc.localize("segments-tab-extract-button-label"))

        result_grid = gr.Matrix(
            headers=[
                loc.localize("segments-tab-result-dataframe-column-file-header"),
                loc.localize("segments-tab-result-dataframe-column-execution-header"),
            ],
            elem_classes="matrix-mh-200",
        )

        def on_training_data_change(value):
            btn_label = loc.localize("segments-tab-select-annotation-input-directory-button-label") if value else loc.localize("segments-tab-select-results-input-directory-button-label")
            return gr.Button(btn_label), gr.Slider(visible=not value), gr.Textbox(visible=value), gr.Textbox(visible=value)

        training_data_cb.change(
            on_training_data_change, inputs=training_data_cb, outputs=[select_result_directory_btn, min_conf_slider, annotation_files_suffix_tb, species_column_name_tb], show_progress=False
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
                training_data_cb,
                annotation_files_suffix_tb,
                species_column_name_tb,
            ],
            outputs=result_grid,
        )


if __name__ == "__main__":
    gu.open_window(build_segments_tab)
