import os
from pathlib import Path

import PIL.Image
import gradio as gr

import birdnet_analyzer.embeddings as embeddings
import birdnet_analyzer.config as cfg
import birdnet_analyzer.localization as loc
import birdnet_analyzer.utils as utils
import birdnet_analyzer.gui.utils as gu
import birdnet_analyzer.search as search
import birdnet_analyzer.audio as audio
from functools import partial
import PIL

SCRIPT_DIR = os.path.abspath(os.path.dirname(__file__))

def play_audio(audio_infos):
    arr, sr = audio.openAudioFile(audio_infos[0], offset=audio_infos[1], duration=3)
    return sr, arr

def update_export_state(audio_infos, checkbox_value, export_state: dict):
    if checkbox_value:
        export_state[audio_infos[2]] = audio_infos
    else:
        export_state.pop(audio_infos[2], None)

    return export_state

def run_embeddings(input_path, db_directory, db_name, dataset, overlap, threads, batch_size, fmin, fmax, progress=gr.Progress(track_tqdm=True)):
    gu.validate(input_path, loc.localize("embeddings-input-dir-validation-message"))
    gu.validate(db_directory, loc.localize("embeddings-db-dir-validation-message"))
    gu.validate(db_name, loc.localize("embeddings-db-name-validation-message"))
    db_path = os.path.join(db_directory, db_name)

    embeddings.run(input_path, db_path, dataset, overlap, threads, batch_size, fmin, fmax)
    gr.Info(f"{loc.localize('embeddings-tab-finish-info')} {db_path}")
    return gr.Plot()

def run_search(db_path, query_path, max_samples):
    gu.validate(db_path, loc.localize("embeddings-search-db-validation-message"))
    gu.validate(query_path, loc.localize("embeddings-search-query-validation-message"))
    gu.validate(max_samples, loc.localize("embeddings-search-max-samples-validation-message"))

    db = search.getDatabase(db_path)
    results, scores = search.getSearchResults(query_path, db, max_samples, cfg.SIG_FMIN, cfg.SIG_FMAX)
    db.db.close() # Close the database connection to avoid having wal/shm files

    return results, gr.Button(interactive=True)

def run_export(export_state):
    if len(export_state.items()) > 0:
        export_folder = gu.select_folder(state_key="embeddings-search-export-folder")
        if export_folder:
            for index, file in export_state.items():
                dest = os.path.join(export_folder, f"result_{index+1}.wav")
                sig, _ = audio.openAudioFile(file[0], offset=file[1], duration=3)
                audio.saveSignal(sig, dest)
        gr.Info(f"{loc.localize('embeddings-search-export-finish-info')} {export_folder}")
    else:
        gr.Info(loc.localize("embeddings-search-export-no-results-info"))
    
    


def build_embeddings_tab():
    with gr.Tab(loc.localize("embeddings-tab-title")):
        with gr.Tab(loc.localize("embeddings-extract-tab-title")):
            input_directory_state = gr.State()
            db_directory_state = gr.State()
            def select_directory_to_state_and_tb(state_key):
                return (gu.select_directory(collect_files=False, state_key=state_key),) * 2
            with gr.Row():
                select_audio_directory_btn = gr.Button(
                    loc.localize("embeddings-tab-select-input-directory-button-label")
                )
                selected_audio_directory_tb = gr.Textbox(show_label=False, interactive=False)
                select_audio_directory_btn.click(
                    partial(select_directory_to_state_and_tb, state_key="embeddings-input-dir"),
                    outputs=[selected_audio_directory_tb, input_directory_state],
                    show_progress=False,
                )
            with gr.Row():
                select_db_directory_btn = gr.Button(
                    loc.localize("embeddings-tab-select-db-directory-button-label")
                )
            with gr.Row():
                db_name = gr.Textbox(
                    "embeddings.sqlite",
                    visible=False,
                    interactive=True,
                    info=loc.localize("embeddings-tab-db-info"),
                )
            
            def select_directory_and_update_tb():
                dir_name = gu.select_directory(state_key="embeddings-db-dir", collect_files=False)
                if dir_name:
                    loc.set_state("embeddings-db-dir", dir_name)
                    return (
                        dir_name,
                        gr.Textbox(label=dir_name, visible=True),
                    )
                return None, None
            select_db_directory_btn.click(
                select_directory_and_update_tb,
                outputs=[db_directory_state, db_name],
                show_progress=False,
            )
            with gr.Row():
                dataset_input = gr.Textbox(
                    "Dataset",
                    visible=True,
                    interactive=True,
                    label=loc.localize("embeddings-tab-dataset-label"),
                    info=loc.localize("embeddings-tab-dataset-info"),
                )
            with gr.Accordion(loc.localize("embedding-settings-accordion-label"), open=False):
                with gr.Row():
                    overlap_slider = gr.Slider(
                        minimum=0,
                        maximum=2.99,
                        value=0,
                        step=0.01,
                        label=loc.localize("embedding-settings-overlap-slider-label"),
                        info=loc.localize("embedding-settings-overlap-slider-info"),
                    )
                    batch_size_number = gr.Number(
                        precision=1,
                        label=loc.localize("embedding-settings-batchsize-number-label"),
                        value=1,
                        info=loc.localize("embedding-settings-batchsize-number-info"),
                        minimum=1,
                        interactive=True,
                    )
                    threads_number = gr.Number(
                        precision=1,
                        label=loc.localize("embedding-settings-threads-number-label"),
                        value=4,
                        info=loc.localize("embedding-settings-threads-number-info"),
                        minimum=1,
                        interactive=True,
                    )
                with gr.Row():
                    fmin_number = gr.Number(
                        cfg.SIG_FMIN,
                        minimum=0,
                        label=loc.localize("embedding-settings-fmin-number-label"),
                        info=loc.localize("embedding-settings-fmin-number-info"),
                        interactive=True,
                    )
                    fmax_number = gr.Number(
                        cfg.SIG_FMAX,
                        minimum=0,
                        label=loc.localize("embedding-settings-fmax-number-label"),
                        info=loc.localize("embedding-settings-fmax-number-info"),
                        interactive=True,
                    )
            
            progress_plot = gr.Plot()

            start_btn = gr.Button(loc.localize("embeddings-tab-start-button-label"))
            start_btn.click(
                run_embeddings,
                inputs=[
                    input_directory_state,
                    db_directory_state,
                    db_name,
                    dataset_input,
                    overlap_slider,
                    batch_size_number,
                    threads_number,
                    fmin_number,
                    fmax_number
                ],
                outputs=[progress_plot],
            )
        with gr.Tab(loc.localize("embeddings-search-tab-title")):            
            results_state = gr.State([])
            export_state = gr.State({})

            with gr.Row():
                with gr.Column():
                    db_selection_button = gr.Button(
                        loc.localize("embeddings-search-db-selection-button-label")
                    )
                    db_selection_tb = gr.Textbox(
                        label=loc.localize("embeddings-search-db-selection-textbox-label"),
                        interactive=False,
                        visible=False,
                    )
                    def on_db_selection_click():
                        file = gu.select_file(("Embeddings Database (*.sqlite)",), state_key="embeddings_search_db")

                        if file:
                            return gr.Textbox(value=file, visible=True), [], {}

                        return None, [], {}
                    db_selection_button.click(
                        on_db_selection_click,
                        outputs=[db_selection_tb, results_state, export_state],
                        show_progress=False,
                    )
                with gr.Column():
                    max_samples_number = gr.Number(
                        label=loc.localize("embeddings-search-max-samples-number-label"),
                        value=10,
                        interactive=True,
                    )

            hidden_audio = gr.Audio(visible=False, autoplay=True, type="numpy")

            with gr.Row():
                with gr.Column():
                    query_spectrogram = gr.Plot(label='')
                    query_input = gr.Audio(type="filepath", label=loc.localize("embeddings-search-query-label"), sources=["upload"])

                    def update_query_spectrogram(audiofilepath):
                        if audiofilepath:
                            spec = utils.spectrogram_from_file(audiofilepath['path'])
                            return spec, [], {}
                        else:
                            return None, [], {}

                    query_input.change(update_query_spectrogram, inputs=[query_input], outputs=[query_spectrogram, results_state, export_state], preprocess=False)
                
                with gr.Column(elem_id="embeddings-search-results"):
                    @gr.render(inputs=[results_state, db_selection_tb])
                    def render_results(results, db_path):
                        checkboxes = []

                        with gr.Row():
                            select_all_button = gr.Button("Toggle all")
                            def select_all(*checkboxes):
                                if False in checkboxes:
                                    return [True] * len(checkboxes)
                                return [False] * len(checkboxes)
                            select_all_button.click(select_all, inputs=checkboxes, outputs=checkboxes)
                        
                        with gr.Row():
                            if db_path != None:
                                db = search.getDatabase(db_path)
                                for index, r in enumerate(results):
                                    with gr.Column():
                                        embedding_source = db.get_embedding_source(r.embedding_id)
                                        file = embedding_source.source_id
                                        spec = utils.spectrogram_from_file(file, offset=embedding_source.offsets[0], duration=3)
                                        plot_audio_state = gr.State([file, embedding_source.offsets[0], index])
                                        with gr.Row():
                                            gr.Plot(spec, label=index+1)

                                        with gr.Row():    
                                            play_btn = gr.Button("▶")
                                            play_btn.click(play_audio, inputs=plot_audio_state, outputs=hidden_audio)
                                            checkbox = gr.Checkbox(label="Export")
                                            checkbox.change(update_export_state, inputs=[plot_audio_state, checkbox, export_state], outputs=export_state)
                                            checkboxes.append(checkbox)
                                db.db.close() # Close the database connection to avoid having wal/shm files



            with gr.Row():
                search_btn = gr.Button(loc.localize("embeddings-search-start-button-label"))
                export_btn = gr.Button(loc.localize("embeddings-search-export-button-label"), interactive=False)
                search_btn.click(
                    run_search,
                    inputs=[db_selection_tb, query_input, max_samples_number],
                    outputs=[results_state, export_btn],
                )

                export_btn.click(
                    run_export,
                    inputs=[export_state],
                )

                    


                
            