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

def run_embeddings(input_path, db_directory, db_name, dataset, overlap, threads, batch_size, fmin, fmax):
    #TODO: Add validation
    db_path = os.path.join(db_directory, db_name)

    embeddings.run(input_path, db_path, dataset, overlap, threads, batch_size, fmin, fmax)
    gr.Info(f"{loc.localize('embeddings-tab-finish-info')} {db_path}")

def run_search(db_path, query_path):
    db = search.getDatabase(db_path)
    results, scores = search.getSearchResults(query_path, db, 4, cfg.SIG_FMIN, cfg.SIG_FMAX)
    specs = []
    for i, r in enumerate(results):
        embedding_source = db.get_embedding_source(r.embedding_id)
        file = embedding_source.source_id
        spec = utils.spectrogram_from_file(file, offset=embedding_source.offsets[0], duration=3)
        specs.append(spec)
    
    return specs

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
            )
        with gr.Tab(loc.localize("embeddings-search-tab-title")):
            db_selection_button = gr.Button(
                loc.localize("embedding-search-db-selection-button-label")
            )
            db_selection_tb = gr.Textbox(
                label=loc.localize("embedding-search-db-selection-textbox-label"),
                interactive=False,
                visible=False,
            )
            def on_db_selection_click():
                file = gu.select_file(("Embeddings Database (*.sqlite)",), state_key="embeddings_search_db")

                if file:
                    return gr.Textbox(value=file, visible=True)

                return None, None 
            db_selection_button.click(
                on_db_selection_click,
                outputs=db_selection_tb,
                show_progress=False,
            )
            with gr.Row():
                with gr.Column():
                    query_spectrogram = gr.Plot()
                    query_input = gr.Audio(type="filepath", label=loc.localize("embeddings-search-query-label"), sources=["upload"])

                    def update_query_spectrogram(audiofilepath):
                        if audiofilepath:
                            spec = utils.spectrogram_from_file(audiofilepath['path'])
                            return spec

                    query_input.change(update_query_spectrogram, inputs=[query_input], outputs=[query_spectrogram], preprocess=False)
                with gr.Column():
                    with gr.Row():
                        result_plot1 = gr.Plot()
                        result_plot2 = gr.Plot()
                    with gr.Row():
                        result_plot3 = gr.Plot()
                        result_plot4 = gr.Plot()

            with gr.Row():
                search_btn = gr.Button(loc.localize("embeddings-search-start-button-label"))
                search_btn.click(
                    run_search,
                    inputs=[db_selection_tb, query_input],
                    outputs=[result_plot1, result_plot2, result_plot3, result_plot4],
                )

                    


                
            