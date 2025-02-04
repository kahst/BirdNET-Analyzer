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
import birdnet_analyzer.analyze as analyze
from functools import partial
import PIL
import json

SCRIPT_DIR = os.path.abspath(os.path.dirname(__file__))
PAGE_SIZE = 4

def play_audio(audio_infos):
    arr, sr = audio.openAudioFile(audio_infos[0], offset=audio_infos[1], duration=audio_infos[2], speed=cfg.AUDIO_SPEED, fmin=cfg.BANDPASS_FMIN, fmax=cfg.BANDPASS_FMAX)
    return sr, arr

def update_export_state(audio_infos, checkbox_value, export_state: dict):
    if checkbox_value:
        export_state[audio_infos[3]] = audio_infos
    else:
        export_state.pop(audio_infos[3], None)

    return export_state

def run_embeddings(input_path, db_directory, db_name, overlap, threads, batch_size, audio_speed, fmin, fmax, progress=gr.Progress(track_tqdm=True)):
    gu.validate(input_path, loc.localize("embeddings-input-dir-validation-message"))
    gu.validate(db_directory, loc.localize("embeddings-db-dir-validation-message"))
    gu.validate(db_name, loc.localize("embeddings-db-name-validation-message"))
    db_path = os.path.join(db_directory, db_name)

    settings_path = os.path.join(db_path, "birdnet_analyzer_settings.json")

    # Check if the settings file exists and load bandpass and speed settings from the file
    if (os.path.exists(settings_path)):
        with open(settings_path, "r") as f:
            settings = json.load(f)

        embeddings.run(input_path, db_path, overlap, threads, batch_size, settings["AUDIO_SPEED"], settings["BANDPASS_FMIN"], settings["BANDPASS_FMAX"]) 
    else:
        # Transform audiospeed from slider to float
        audio_speed = max(0.1, 1.0 / (audio_speed * -1)) if audio_speed < 0 else max(1.0, float(audio_speed))
        
        if fmin is None or fmax is None or fmin < cfg.SIG_FMIN or fmax > cfg.SIG_FMAX or fmin > fmax:
            raise gr.Error(f"{loc.localize('validation-no-valid-frequency')} [{cfg.SIG_FMIN}, {cfg.SIG_FMAX}]")

        embeddings.run(input_path, db_path, overlap, threads, batch_size, audio_speed, fmin, fmax)

    gr.Info(f"{loc.localize('embeddings-tab-finish-info')} {db_path}")
    return gr.Plot(), gr.Slider(visible=False), gr.Number(visible=False), gr.Number(visible=False)

def run_search(db_path, query_path, max_samples, score_fn):
    gu.validate(db_path, loc.localize("embeddings-search-db-validation-message"))
    gu.validate(query_path, loc.localize("embeddings-search-query-validation-message"))
    gu.validate(max_samples, loc.localize("embeddings-search-max-samples-validation-message"))

    db = search.getDatabase(db_path)
    settings_path = os.path.join(db_path, "birdnet_analyzer_settings.json")
    with open(settings_path, 'r') as f:
        settings = json.load(f)

    results = search.getSearchResults(query_path, db, max_samples, settings["AUDIO_SPEED"], settings["BANDPASS_FMIN"], settings["BANDPASS_FMAX"], score_fn)
    db.db.close() # Close the database connection to avoid having wal/shm files

    chunks = [results[i:i + PAGE_SIZE] for i in range(0, len(results), PAGE_SIZE)]

    return chunks, 0, gr.Button(interactive=True), {}

def run_export(export_state):
    if len(export_state.items()) > 0:
        export_folder = gu.select_folder(state_key="embeddings-search-export-folder")
        if export_folder:
            for index, file in export_state.items():
                dest = os.path.join(export_folder, f"result_{index+1}_score_{file[4]:.5f}.wav")
                sig, _ = audio.openAudioFile(file[0], offset=file[1]/cfg.AUDIO_SPEED, duration=file[2]/cfg.AUDIO_SPEED, sample_rate=None)
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
                db_name_tb = gr.Textbox(
                    "embeddings_database",
                    visible=False,
                    interactive=True,
                    info=loc.localize("embeddings-tab-db-info"),
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
                    audio_speed_slider = gr.Slider(
                        minimum=-10,
                        maximum=10,
                        value=0,
                        step=1,
                        label=loc.localize("embedding-settings-audio-speed-slider-label"),
                        info=loc.localize("embedding-settings-audio-speed-slider-info"),
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

            def select_directory_and_update_tb():
                dir_name = gu.select_directory(state_key="embeddings-db-dir", collect_files=False)
                if dir_name:
                    loc.set_state("embeddings-db-dir", dir_name)
                    settings_path = os.path.join(dir_name, db_name_tb.value, "birdnet_analyzer_settings.json")
                    if os.path.exists(settings_path):
                        return (
                            dir_name,
                            gr.Textbox(label=dir_name, visible=True),
                            gr.Slider(visible=False),
                            gr.Number(visible=False),
                            gr.Number(visible=False),
                        )

                    return (
                        dir_name, gr.Textbox(label=dir_name, visible=True), gr.Slider(visible=True), gr.Number(visible=True), gr.Number(visible=True)
                    )
                return None, None, gr.Slider(visible=True), gr.Number(visible=True), gr.Number(visible=True)
            
            select_db_directory_btn.click(
                select_directory_and_update_tb,
                outputs=[db_directory_state, db_name_tb, audio_speed_slider, fmin_number, fmax_number],
                show_progress=False,
            )

            def check_settings_file(dir_name, db_name):
                settings_path = os.path.join(dir_name, db_name, "birdnet_analyzer_settings.json")
                if db_name and os.path.exists(settings_path):
                        return (
                            gr.Slider(visible=False),
                            gr.Number(visible=False),
                            gr.Number(visible=False),
                        )
                else:
                    return gr.Slider(visible=True), gr.Number(visible=True), gr.Number(visible=True)

            db_name_tb.change(
                check_settings_file,
                inputs=[db_directory_state, db_name_tb],
                outputs=[audio_speed_slider, fmin_number, fmax_number],
                show_progress=False,
            )
            
            progress_plot = gr.Plot()

            start_btn = gr.Button(loc.localize("embeddings-tab-start-button-label"), variant="huggingface")
            start_btn.click(
                run_embeddings,
                inputs=[
                    input_directory_state,
                    db_directory_state,
                    db_name_tb,
                    overlap_slider,
                    batch_size_number,
                    threads_number,
                    audio_speed_slider,
                    fmin_number,
                    fmax_number
                ],
                outputs=[progress_plot, audio_speed_slider, fmin_number, fmax_number],
            )
        with gr.Tab(loc.localize("embeddings-search-tab-title")):            
            results_state = gr.State([])
            page_state = gr.State(0)
            export_state = gr.State({})

            with gr.Row():
                with gr.Column():
                    with gr.Row():
                        db_selection_tb = gr.Textbox(
                            label=loc.localize("embeddings-search-db-selection-textbox-label"),
                            max_lines=3,
                            interactive=False,
                            visible=False,
                        )
                        db_embedding_count_number = gr.Number(
                            interactive=False,
                            visible=False,
                            label=loc.localize("embeddings-search-db-embedding-count-number-label")
                        )
                    
                    with gr.Row():       
                        db_bandpass_frequencies_tb = gr.Textbox(
                            label=loc.localize("embeddings-search-db-bandpass-frequencies-label"),
                            interactive=False,
                            visible=False,
                        )
                        db_audio_speed_number = gr.Number(
                            interactive=False,
                            visible=False,
                            label=loc.localize("embeddings-search-db-audio-speed-number-label")
                        )

                    db_selection_button = gr.Button(
                        loc.localize("embeddings-search-db-selection-button-label")
                    )
                    def on_db_selection_click():
                        file = gu.select_folder(state_key="embeddings_search_db")

                        db = embeddings.getDatabase(file)
                        embedding_count = db.count_embeddings()
                        db.db.close()

                        settings_path = os.path.join(file, "birdnet_analyzer_settings.json")
                        with open(settings_path, 'r') as f:
                            settings = json.load(f)

                        frequencies = f"{settings['BANDPASS_FMIN']} - {settings['BANDPASS_FMAX']} Hz"
                        speed = settings['AUDIO_SPEED']

                        if file:
                            return gr.Textbox(value=file, visible=True), gr.Number(value=embedding_count, visible=True), gr.Textbox(visible=True, value=frequencies), gr.Number(visible=True, value=speed), [], {}

                        return None, None, None, None, [], {}
                    db_selection_button.click(
                        on_db_selection_click,
                        outputs=[db_selection_tb, db_embedding_count_number, db_bandpass_frequencies_tb, db_audio_speed_number, results_state, export_state],
                        show_progress=False,
                    )
                with gr.Column():
                    max_samples_number = gr.Number(
                        label=loc.localize("embeddings-search-max-samples-number-label"),
                        value=10,
                        interactive=True,
                    )
                    score_fn_select = gr.Radio(
                        label=loc.localize("embeddings-search-score-fn-select-label"),
                        choices=["cosine", "dot", "euclidean"],
                        value="cosine",
                        interactive=True,
                    )

            hidden_audio = gr.Audio(visible=False, autoplay=True, type="numpy")

            with gr.Row():
                with gr.Column():
                    query_spectrogram = gr.Plot(label='')
                    query_input = gr.Audio(type="filepath", label=loc.localize("embeddings-search-query-label"), sources=["upload"])
                    gr.HTML(f"<p>{loc.localize('embeddings-search-query-hint')}</p>")
                    def update_query_spectrogram(audiofilepath):
                        if audiofilepath:
                            spec = utils.spectrogram_from_file(audiofilepath['path'])
                            return spec, [], {}
                        else:
                            return None, [], {}

                    query_input.change(update_query_spectrogram, inputs=[query_input], outputs=[query_spectrogram, results_state, export_state], preprocess=False)

                with gr.Column(elem_id="embeddings-search-results"):
                    @gr.render(inputs=[results_state, page_state, db_selection_tb, export_state], triggers=[results_state.change, page_state.change, db_selection_tb.change])
                    def render_results(results, page, db_path, exports):
                        with gr.Row():
                            if db_path != None and len(results) > 0:
                                db = search.getDatabase(db_path)

                                for i, r in enumerate(results[page]):
                                    with gr.Column():
                                        index = i + page * PAGE_SIZE
                                        embedding_source = db.get_embedding_source(r.embedding_id)
                                        file = embedding_source.source_id
                                        offset = embedding_source.offsets[0]
                                        duration = 3
                                        spec = utils.spectrogram_from_file(file, offset=offset, duration=duration, speed=cfg.AUDIO_SPEED, fmin=cfg.BANDPASS_FMIN, fmax=cfg.BANDPASS_FMAX)
                                        plot_audio_state = gr.State([file, offset, duration, index, r.sort_score])
                                        with gr.Row():
                                            gr.Plot(spec, label=f"{index+1}_score: {r.sort_score:.2f}")

                                        with gr.Row():    
                                            play_btn = gr.Button("▶")
                                            play_btn.click(play_audio, inputs=plot_audio_state, outputs=hidden_audio)
                                            checkbox = gr.Checkbox(label="Export", value=(index in exports.keys()))
                                            checkbox.change(update_export_state, inputs=[plot_audio_state, checkbox, export_state], outputs=export_state)
                                db.db.close() # Close the database connection to avoid having wal/shm files

                        with gr.Row():
                            prev_btn = gr.Button("Previous Page", interactive=page>0)
                            next_btn = gr.Button("Next Page", interactive=page < len(results) - 1)

                            def prev_page(page):
                                return page -1 if page > 0 else 0
                            def next_page(page):
                                return page +1
                            
                            prev_btn.click(prev_page, inputs=[page_state], outputs=[page_state])
                            next_btn.click(next_page, inputs=[page_state], outputs=[page_state])
                     


            with gr.Row():
                search_btn = gr.Button(loc.localize("embeddings-search-start-button-label"), variant="huggingface")
                export_btn = gr.Button(loc.localize("embeddings-search-export-button-label"), variant="huggingface", interactive=False)
                search_btn.click(
                    run_search,
                    inputs=[db_selection_tb, query_input, max_samples_number, score_fn_select],
                    outputs=[results_state, page_state, export_btn, export_state],
                )

                export_btn.click(
                    run_export,
                    inputs=[export_state],
                )

                    


                
            