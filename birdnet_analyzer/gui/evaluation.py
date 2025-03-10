import os
import json
import gradio as gr
import pandas as pd
import matplotlib.pyplot as plt
import tempfile
import shutil

from birdnet_analyzer.evaluation.core import process_data
from birdnet_analyzer.evaluation.preprocessing.data_processor import DataProcessor


def build_evaluation_tab():
    # Default columns for annotations
    annotation_default_columns = {
        "Start Time": "Begin Time (s)",
        "End Time": "End Time (s)",
        "Class": "Class",
        "Recording": "Begin File",
        "Duration": "File Duration (s)",
    }

    # Default columns for predictions
    prediction_default_columns = {
        "Start Time": "Begin Time (s)",
        "End Time": "End Time (s)",
        "Class": "Common Name",
        "Recording": "Begin File",
        "Duration": "File Duration (s)",
        "Confidence": "Confidence",
    }

    def download_class_mapping_template():
        template_mapping = {
            "Predicted Class Name 1": "Annotation Class Name 1",
            "Predicted Class Name 2": "Annotation Class Name 2",
            "Predicted Class Name 3": "Annotation Class Name 3",
            "Predicted Class Name 4": "Annotation Class Name 4",
            "Predicted Class Name 5": "Annotation Class Name 5",
        }
        fd, temp_path = tempfile.mkstemp(suffix=".json")
        with os.fdopen(fd, 'w') as f:
            json.dump(template_mapping, f, indent=4)
        desired_path = os.path.join(os.path.dirname(temp_path), "class_mapping_template.json")
        if os.path.exists(desired_path):
            os.remove(desired_path)
        os.rename(temp_path, desired_path)
        return gr.update(value=desired_path, visible=True)

    def download_results_table(pa, predictions, labels, class_wise_value):
        if pa is None or predictions is None or labels is None:
            return None
        try:
            metrics_df = pa.calculate_metrics(predictions, labels, per_class_metrics=class_wise_value)
            temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".csv")
            metrics_df.to_csv(temp_file.name, index=True)
            temp_file.close()
            desired_path = os.path.join(os.path.dirname(temp_file.name), "results_table.csv")
            if os.path.exists(desired_path):
                os.remove(desired_path)
            os.rename(temp_file.name, desired_path)
            return gr.update(value=desired_path)
        except Exception as e:
            print(f"Error saving results table: {e}")
            return None

    def download_data_table(processor_state):
        if processor_state is None:
            return None
        try:
            # If processor_state is a dict, extract the actual processor instance.
            proc = processor_state.get("processor") if isinstance(processor_state, dict) else processor_state
            data_df = proc.get_sample_data()
            temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".csv")
            data_df.to_csv(temp_file.name, index=False)
            temp_file.close()
            desired_path = os.path.join(os.path.dirname(temp_file.name), "data_table.csv")
            if os.path.exists(desired_path):
                os.remove(desired_path)
            os.rename(temp_file.name, desired_path)
            return gr.update(value=desired_path)
        except Exception as e:
            print(f"Error saving data table: {e}")
            return None

    def get_columns_from_uploaded_files(files):
        columns = set()
        if files:
            for file_obj in files:
                try:
                    df = pd.read_csv(file_obj.name, sep=None, engine="python", nrows=0)
                    columns.update(df.columns)
                except Exception as e:
                    print(f"Error reading file {file_obj.name}: {e}")
        return sorted(list(columns))

    def save_uploaded_files(files):
        if not files:
            return None
        temp_dir = tempfile.mkdtemp()
        for file_obj in files:
            dest_path = os.path.join(temp_dir, os.path.basename(file_obj.name))
            shutil.copy(file_obj.name, dest_path)
        return temp_dir

    # Single initialize_processor that can reuse given directories.
    def initialize_processor(annotation_files, prediction_files, mapping_file_obj,
                             sample_duration_value, min_overlap_value, recording_duration,
                             ann_start_time, ann_end_time, ann_class, ann_recording, ann_duration,
                             pred_start_time, pred_end_time, pred_class, pred_confidence, pred_recording, pred_duration,
                             annotation_dir=None, prediction_dir=None):
        if not annotation_files or not prediction_files:
            return [], [], None, None, None
        if annotation_dir is None:
            annotation_dir = save_uploaded_files(annotation_files)
        if prediction_dir is None:
            prediction_dir = save_uploaded_files(prediction_files)
        # Fallback for annotation columns.
        ann_start_time = ann_start_time if ann_start_time else annotation_default_columns["Start Time"]
        ann_end_time = ann_end_time if ann_end_time else annotation_default_columns["End Time"]
        ann_class = ann_class if ann_class else annotation_default_columns["Class"]
        ann_recording = ann_recording if ann_recording else annotation_default_columns["Recording"]
        ann_duration = ann_duration if ann_duration else annotation_default_columns["Duration"]

        # Fallback for prediction columns.
        pred_start_time = pred_start_time if pred_start_time else prediction_default_columns["Start Time"]
        pred_end_time = pred_end_time if pred_end_time else prediction_default_columns["End Time"]
        pred_class = pred_class if pred_class else prediction_default_columns["Class"]
        pred_confidence = pred_confidence if pred_confidence else prediction_default_columns["Confidence"]
        pred_recording = pred_recording if pred_recording else prediction_default_columns["Recording"]
        pred_duration = pred_duration if pred_duration else prediction_default_columns["Duration"]

        cols_ann = {
            "Start Time": ann_start_time,
            "End Time": ann_end_time,
            "Class": ann_class,
            "Recording": ann_recording,
            "Duration": ann_duration,
        }
        cols_pred = {
            "Start Time": pred_start_time,
            "End Time": pred_end_time,
            "Class": pred_class,
            "Confidence": pred_confidence,
            "Recording": pred_recording,
            "Duration": pred_duration,
        }
        # Handle mapping file: if it has a temp_files attribute use that, otherwise assume it's a filepath.
        if mapping_file_obj and hasattr(mapping_file_obj, "temp_files"):
            mapping_path = list(mapping_file_obj.temp_files)[0]
        else:
            mapping_path = mapping_file_obj if mapping_file_obj else None
        if mapping_path:
            with open(mapping_path, 'r') as f:
                class_mapping = json.load(f)
        else:
            class_mapping = None
        try:
            proc = DataProcessor(
                prediction_directory_path=prediction_dir,
                prediction_file_name=None,
                annotation_directory_path=annotation_dir,
                annotation_file_name=None,
                class_mapping=class_mapping,
                sample_duration=sample_duration_value,
                min_overlap=min_overlap_value,
                columns_predictions=cols_pred,
                columns_annotations=cols_ann,
                recording_duration=recording_duration,
            )
            avail_classes = list(proc.classes)  # Ensure it's a list
            avail_recordings = proc.samples_df["filename"].unique().tolist()
            return avail_classes, avail_recordings, proc, annotation_dir, prediction_dir
        except Exception as e:
            print(f"Error initializing processor: {e}")
            return [], [], None, None, None

    # update_selections is triggered when files or mapping file change.
    # It creates the temporary directories once and stores them along with the processor.
    # It now also receives the current selection values so that user selections are preserved.
    def update_selections(annotation_files, prediction_files, mapping_file_obj,
                          sample_duration_value, min_overlap_value, recording_duration_value,
                          ann_start_time, ann_end_time, ann_class, ann_recording, ann_duration,
                          pred_start_time, pred_end_time, pred_class, pred_confidence, pred_recording, pred_duration,
                          current_classes, current_recordings):
        if recording_duration_value.strip() == "":
            rec_dur = None
        else:
            try:
                rec_dur = float(recording_duration_value)
            except ValueError:
                rec_dur = None
        # Create temporary directories once.
        annotation_dir = save_uploaded_files(annotation_files)
        prediction_dir = save_uploaded_files(prediction_files)
        avail_classes, avail_recordings, proc, annotation_dir, prediction_dir = initialize_processor(
            annotation_files, prediction_files, mapping_file_obj,
            sample_duration_value, min_overlap_value, rec_dur,
            ann_start_time, ann_end_time, ann_class, ann_recording, ann_duration,
            pred_start_time, pred_end_time, pred_class, pred_confidence, pred_recording, pred_duration,
            annotation_dir, prediction_dir
        )
        # Build a state dictionary to store the processor and the directories.
        state = {"processor": proc, "annotation_dir": annotation_dir, "prediction_dir": prediction_dir}
        # If no current selection exists, default to all available classes/recordings;
        # otherwise, preserve any selections that are still valid.
        new_classes = avail_classes if not current_classes else [c for c in current_classes if c in avail_classes] or avail_classes
        new_recordings = avail_recordings if not current_recordings else [r for r in current_recordings if r in avail_recordings] or avail_recordings
        return (gr.update(choices=avail_classes, value=new_classes),
                gr.update(choices=avail_recordings, value=new_recordings),
                state)

    with gr.TabItem("Evaluation"):
        # Custom CSS to match the layout style of other files and remove gray backgrounds.
        gr.Markdown(
            """
            <style>
            body { background-color: #fff; font-family: "Helvetica Neue", Helvetica, Arial, sans-serif; }
            .gradio-container { 
                border: 1px solid #ccc; 
                border-radius: 8px; 
                padding: 16px; 
                background-color: transparent; 
            }
            /* Override any group styles */
            .gradio-group { 
                background-color: transparent !important; 
                border: none !important; 
                box-shadow: none !important;
            }
            h2, h3 { color: #333; }
            .custom-button { border-radius: 6px; }
            /* Grid layout for checkbox groups */
            .custom-checkbox-group { 
                display: grid; 
                grid-template-columns: repeat(auto-fit, minmax(150px, 1fr)); 
                grid-gap: 8px; 
            }
            </style>
            """
        )

        processor_state = gr.State()
        pa_state = gr.State()
        predictions_state = gr.State()
        labels_state = gr.State()

        # ----------------------- File Selection Box -----------------------
        with gr.Group():
            gr.Markdown("### File Selection")
            with gr.Row():
                with gr.Column():
                    gr.Markdown("**Annotations Files:**")
                    gr.Markdown("<small>Select the files containing the true labels for evaluation.</small>")
                    annotation_files = gr.File(label="Select Annotation Files", file_count="multiple", file_types=[".csv", ".txt"])
                with gr.Column():
                    gr.Markdown("**Predictions Files:**")
                    gr.Markdown("<small>Select the files containing the model's predictions.</small>")
                    prediction_files = gr.File(label="Select Prediction Files", file_count="multiple", file_types=[".csv", ".txt"])

        # ----------------------- Annotations Columns Box -----------------------
        with gr.Group():
            gr.Markdown("### Annotations Columns")
            with gr.Row():
                for label_text in ["Start Time", "End Time", "Class", "Recording", "Duration"]:
                    gr.Markdown(f"**{label_text}**")
            with gr.Row():
                annotation_columns = {}
                for label_text in ["Start Time", "End Time", "Class", "Recording", "Duration"]:
                    annotation_columns[label_text] = gr.Dropdown(choices=[], label="", show_label=False)

        # ----------------------- Predictions Columns Box -----------------------
        with gr.Group():
            gr.Markdown("### Predictions Columns")
            with gr.Row():
                for label_text in ["Start Time", "End Time", "Class", "Confidence", "Recording", "Duration"]:
                    gr.Markdown(f"**{label_text}**")
            with gr.Row():
                prediction_columns = {}
                for label_text in ["Start Time", "End Time", "Class", "Confidence", "Recording", "Duration"]:
                    prediction_columns[label_text] = gr.Dropdown(choices=[], label="", show_label=False)

        # ----------------------- Class Mapping Box -----------------------
        with gr.Group():
            gr.Markdown("### Class Mapping (Optional)")
            gr.Markdown("<small>If class names differ between prediction and annotation files, use a class mapping JSON file.</small>")
            gr.Markdown(
                "<small>Click 'Download Template' to get a JSON template for mapping class names between prediction and annotation files.</small>")
            with gr.Row():
                mapping_file = gr.File(label="Upload Mapping File", file_count="single", file_types=[".json"])
                download_mapping_button = gr.DownloadButton(label="Download Template", visible=True, variant="huggingface")
            download_mapping_button.click(fn=download_class_mapping_template, inputs=[], outputs=download_mapping_button)

        # ----------------------- Classes and Recordings Selection Box -----------------------
        with gr.Group():
            with gr.Row():
                with gr.Column():
                    gr.Markdown("### Select Classes:")
                    gr.Markdown(
                        "<small>Select the classes to calculate the metrics.</small>")
                    select_classes_checkboxgroup = gr.CheckboxGroup(choices=[], value=[], label="", interactive=True, elem_classes="custom-checkbox-group")
                with gr.Column():
                    gr.Markdown("### Select Recordings:")
                    gr.Markdown(
                        "<small>Select the recordings to calculate the metrics.</small>")
                    select_recordings_checkboxgroup = gr.CheckboxGroup(choices=[], value=[], label="", interactive=True, elem_classes="custom-checkbox-group")

        # ----------------------- Parameters Box -----------------------
        gr.Markdown("### Parameters")
        with gr.Row():
            sample_duration = gr.Number(value=3, label="Sample Duration (s)", precision=0, info="Audio sample length (in seconds).")
            recording_duration = gr.Textbox(label="Recording Duration (s)", placeholder="Determined from files", info="Inferred from the data if not specified.")
            min_overlap = gr.Number(value=0.5, label="Minimum Overlap (s)", info="Overlap needed to assign an annotation to a sample.")
            threshold = gr.Slider(minimum=0.01, maximum=0.99, value=0.1, label="Threshold", info="Threshold for classifying a prediction as positive.")
            class_wise = gr.Checkbox(label="Class-wise Metrics", value=False, info="Calculate metrics separately for each class.")

        # ----------------------- Metrics Box -----------------------
        gr.Markdown("### Metrics")
        with gr.Row():
            metric_info = {
                "AUROC": "AUROC measures the likelihood that the model ranks a random positive case higher than a random negative case.",
                "Precision": "Precision measures how often the model's positive predictions are actually correct.",
                "Recall": "Recall measures the percentage of actual positive cases correctly identified by the model for each class.",
                "F1 Score": "The F1 score is the harmonic mean of precision and recall, balancing both metrics.",
                "Average Precision (AP)": "Average Precision summarizes the precision-recall curve by averaging precision across all recall levels.",
                "Accuracy": "Accuracy measures the percentage of correct predictions made by the model.",
            }
            metrics_checkboxes = {}
            for metric_name, description in metric_info.items():
                metrics_checkboxes[metric_name.lower()] = gr.Checkbox(label=metric_name, value=True, info=description)

        # ----------------------- Actions Box -----------------------
        gr.Markdown("### Actions")
        with gr.Row():
            calculate_button = gr.Button("Calculate Metrics", variant="huggingface")
            plot_metrics_button = gr.Button("Plot Metrics", variant="huggingface")
            plot_confusion_button = gr.Button("Plot Confusion Matrix", variant="huggingface")
            plot_metrics_all_thresholds_button = gr.Button("Plot Metrics All Thresholds", variant="huggingface")
        with gr.Row():
            download_results_button = gr.DownloadButton(label="Download Results Table", visible=True, variant="huggingface")
            download_data_button = gr.DownloadButton(label="Download Data Table", visible=True, variant="huggingface")
        download_results_button.click(fn=download_results_table,
                                      inputs=[pa_state, predictions_state, labels_state, class_wise],
                                      outputs=download_results_button)
        download_data_button.click(fn=download_data_table,
                                   inputs=[processor_state],
                                   outputs=download_data_button)
        results_text = gr.Textbox(label="Results", lines=10, visible=False)
        plot_output = gr.Plot(label="Plot", visible=False)

        # Update column dropdowns when files are uploaded.
        def update_annotation_columns(uploaded_files):
            cols = get_columns_from_uploaded_files(uploaded_files)
            cols = [""] + cols
            updates = []
            for label in ["Start Time", "End Time", "Class", "Recording", "Duration"]:
                default_val = annotation_default_columns.get(label)
                val = default_val if default_val in cols else None
                updates.append(gr.update(choices=cols, value=val))
            return updates

        def update_prediction_columns(uploaded_files):
            cols = get_columns_from_uploaded_files(uploaded_files)
            cols = [""] + cols
            updates = []
            for label in ["Start Time", "End Time", "Class", "Confidence", "Recording", "Duration"]:
                default_val = prediction_default_columns.get(label)
                val = default_val if default_val in cols else None
                updates.append(gr.update(choices=cols, value=val))
            return updates

        annotation_files.change(fn=update_annotation_columns,
                                inputs=annotation_files,
                                outputs=[annotation_columns[label] for label in ["Start Time", "End Time", "Class", "Recording", "Duration"]])
        prediction_files.change(fn=update_prediction_columns,
                                inputs=prediction_files,
                                outputs=[prediction_columns[label] for label in ["Start Time", "End Time", "Class", "Confidence", "Recording", "Duration"]])
        # Update available selections (classes and recordings) and the processor state when files or mapping file change.
        # Also pass the current selection values so that user selections are preserved.
        for comp in list(annotation_columns.values()) + list(prediction_columns.values()) + [mapping_file]:
            comp.change(
                fn=update_selections,
                inputs=[
                    annotation_files, prediction_files, mapping_file, sample_duration, min_overlap, recording_duration,
                    annotation_columns["Start Time"], annotation_columns["End Time"],
                    annotation_columns["Class"],
                    annotation_columns["Recording"], annotation_columns["Duration"],
                    prediction_columns["Start Time"], prediction_columns["End Time"],
                    prediction_columns["Class"],
                    prediction_columns["Confidence"], prediction_columns["Recording"],
                    prediction_columns["Duration"],
                    select_classes_checkboxgroup, select_recordings_checkboxgroup
                ],
                outputs=[select_classes_checkboxgroup, select_recordings_checkboxgroup, processor_state]
            )

        # calculate_metrics now uses the stored temporary directories from processor_state.
        # The function now accepts selected_classes and selected_recordings as inputs.
        def calculate_metrics(annotation_files, prediction_files, mapping_file_obj, sample_duration_value,
                              min_overlap_value, recording_duration_value,
                              ann_start_time, ann_end_time, ann_class, ann_recording, ann_duration,
                              pred_start_time, pred_end_time, pred_class, pred_confidence, pred_recording, pred_duration,
                              threshold_value, class_wise_value, *extra_inputs):
            # Expect extra_inputs to contain:
            # [metrics checkbox values..., selected_classes, selected_recordings, proc_state]
            if len(extra_inputs) < 3:
                return ("Missing processor state or selection values.", None, None, None, gr.update(), gr.update(), None, gr.update(visible=True))
            # Extract metrics checkbox values (assume there are 6 metrics)
            metrics_checkbox_values = extra_inputs[:-3]
            selected_classes_list = extra_inputs[-3] or []
            selected_recordings_list = extra_inputs[-2] or None
            proc_state = extra_inputs[-1]
            selected_metrics = []
            for value, (m_lower, _) in zip(metrics_checkbox_values, metrics_checkboxes.items()):
                if value:
                    selected_metrics.append(m_lower)
            valid_metrics = {
                "accuracy": "accuracy",
                "recall": "recall",
                "precision": "precision",
                "f1 score": "f1",
                "average precision (ap)": "ap",
                "auroc": "auroc",
            }
            metrics = tuple([valid_metrics[m] for m in selected_metrics if m in valid_metrics])
            # Fall back to available classes from processor state if none selected.
            if not selected_classes_list and proc_state and proc_state.get("processor"):
                selected_classes_list = list(proc_state.get("processor").classes)
            if not selected_classes_list:
                return ("Error: At least one class must be selected.", None, None, None, gr.update(), gr.update(), proc_state, gr.update(visible=True))
            if recording_duration_value.strip() == "":
                rec_dur = None
            else:
                try:
                    rec_dur = float(recording_duration_value)
                except ValueError:
                    return ("Please enter a valid number for Recording Duration.",
                            None, None, None, gr.update(), gr.update(), proc_state, gr.update(visible=True))
            if mapping_file_obj and hasattr(mapping_file_obj, "temp_files"):
                mapping_path = list(mapping_file_obj.temp_files)[0]
            else:
                mapping_path = mapping_file_obj if mapping_file_obj else None
            # Use the stored temporary directories from the processor state.
            annotation_dir = proc_state.get("annotation_dir")
            prediction_dir = proc_state.get("prediction_dir")
            try:
                metrics_df, pa, preds, labs = process_data(
                    annotation_path=annotation_dir,
                    prediction_path=prediction_dir,
                    mapping_path=mapping_path,
                    sample_duration=sample_duration_value,
                    min_overlap=min_overlap_value,
                    recording_duration=rec_dur,
                    columns_annotations={
                        "Start Time": ann_start_time,
                        "End Time": ann_end_time,
                        "Class": ann_class,
                        "Recording": ann_recording,
                        "Duration": ann_duration,
                    },
                    columns_predictions={
                        "Start Time": pred_start_time,
                        "End Time": pred_end_time,
                        "Class": pred_class,
                        "Confidence": pred_confidence,
                        "Recording": pred_recording,
                        "Duration": pred_duration,
                    },
                    selected_classes=selected_classes_list,
                    selected_recordings=selected_recordings_list,
                    metrics_list=metrics,
                    threshold=threshold_value,
                    class_wise=class_wise_value,
                )
                result_text = metrics_df.to_string()
                return (result_text, pa, preds, labs,
                        gr.update(), gr.update(), proc_state, gr.update(visible=True))
            except Exception as e:
                result_text = f"Error processing data: {e}"
                return (result_text, None, None, None,
                        gr.update(), gr.update(), proc_state, gr.update(visible=True))

        # Updated calculate_button click now passes the selected classes and recordings.
        calculate_button.click(
            calculate_metrics,
            inputs=[
                annotation_files, prediction_files, mapping_file, sample_duration, min_overlap, recording_duration,
                annotation_columns["Start Time"], annotation_columns["End Time"], annotation_columns["Class"],
                annotation_columns["Recording"], annotation_columns["Duration"],
                prediction_columns["Start Time"], prediction_columns["End Time"], prediction_columns["Class"],
                prediction_columns["Confidence"], prediction_columns["Recording"], prediction_columns["Duration"],
                threshold, class_wise
            ] + [checkbox for checkbox in metrics_checkboxes.values()] + [select_classes_checkboxgroup, select_recordings_checkboxgroup, processor_state],
            outputs=[results_text, pa_state, predictions_state, labels_state,
                     select_classes_checkboxgroup, select_recordings_checkboxgroup, processor_state, results_text]
        )

        def plot_metrics(pa, predictions, labels, class_wise_value):
            if pa is None or predictions is None or labels is None:
                return None, "Please calculate metrics first.", gr.update(visible=False)
            try:
                fig = pa.plot_metrics(predictions, labels, per_class_metrics=class_wise_value)
                plt.close(fig)
                return fig, None, gr.update(visible=True)
            except Exception as e:
                return None, f"Error plotting metrics: {e}", gr.update(visible=False)

        plot_metrics_button.click(
            plot_metrics,
            inputs=[pa_state, predictions_state, labels_state, class_wise],
            outputs=[plot_output, results_text, plot_output]
        )

        def plot_confusion_matrix(pa, predictions, labels):
            if pa is None or predictions is None or labels is None:
                return None, "Please calculate metrics first.", gr.update(visible=False)
            try:
                fig = pa.plot_confusion_matrix(predictions, labels)
                plt.close(fig)
                return fig, None, gr.update(visible=True)
            except Exception as e:
                return None, f"Error plotting confusion matrix: {e}", gr.update(visible=False)

        plot_confusion_button.click(
            plot_confusion_matrix,
            inputs=[pa_state, predictions_state, labels_state],
            outputs=[plot_output, results_text, plot_output]
        )

        def plot_metrics_all_thresholds(pa, predictions, labels, class_wise_value):
            if pa is None or predictions is None or labels is None:
                return None, "Please calculate metrics first.", gr.update(visible=False)
            try:
                fig = pa.plot_metrics_all_thresholds(predictions, labels, per_class_metrics=class_wise_value)
                plt.close(fig)
                return fig, None, gr.update(visible=True)
            except Exception as e:
                return None, f"Error plotting metrics for all thresholds: {e}", gr.update(visible=False)

        plot_metrics_all_thresholds_button.click(
            plot_metrics_all_thresholds,
            inputs=[pa_state, predictions_state, labels_state, class_wise],
            outputs=[plot_output, results_text, plot_output]
        )


if __name__ == "__main__":
    import birdnet_analyzer.gui.utils as gu
    gu.open_window(build_evaluation_tab)
