# evaluation.py

import os
import json
import gradio as gr
import pandas as pd
import matplotlib.pyplot as plt
import tempfile
from tkinter import Tk, filedialog

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

    with gr.TabItem("Evaluation"):

        # States to hold processed data
        processor_state = gr.State()
        pa_state = gr.State()
        predictions_state = gr.State()
        labels_state = gr.State()

        gr.Markdown("## Performance Assessor")

        # File selection group
        with gr.Group():
            gr.Markdown("### File Selection")

            # File inputs layout
            with gr.Row():
                # Annotations Folder selection
                with gr.Column():
                    annotation_label = gr.Markdown("**Annotations Folder:**")
                    annotation_input = gr.Textbox(label="", placeholder="Enter path to annotations folder")
                    annotation_browse = gr.Button("Browse")
                # Predictions Folder selection
                with gr.Column():
                    prediction_label = gr.Markdown("**Predictions Folder:**")
                    prediction_input = gr.Textbox(label="", placeholder="Enter path to predictions folder")
                    prediction_browse = gr.Button("Browse")

            # Browse button functions
            def browse_annotation_folder():
                Tk().withdraw()
                folder_path = filedialog.askdirectory(title="Select Annotations Folder")
                return folder_path

            def browse_prediction_folder():
                Tk().withdraw()
                folder_path = filedialog.askdirectory(title="Select Predictions Folder")
                return folder_path

            annotation_browse.click(
                browse_annotation_folder,
                outputs=annotation_input
            )

            prediction_browse.click(
                browse_prediction_folder,
                outputs=prediction_input
            )

            gr.Row()  # Add empty row for spacing

            # Columns selection layout
            # Annotations columns selection
            gr.Markdown("### Annotations Columns")
            with gr.Row():
                # First row: labels
                for label_text in ["Start Time", "End Time", "Class", "Recording", "Duration"]:
                    gr.Markdown(f"**{label_text}**")
            with gr.Row():
                # Second row: dropdowns
                annotation_columns = {}
                for label_text in ["Start Time", "End Time", "Class", "Recording", "Duration"]:
                    annotation_columns[label_text] = gr.Dropdown(choices=[], label="", show_label=False)

            gr.Row()  # Add empty row for spacing

            # Predictions columns selection
            gr.Markdown("### Predictions Columns")
            with gr.Row():
                # First row: labels
                for label_text in ["Start Time", "End Time", "Class", "Confidence", "Recording", "Duration"]:
                    gr.Markdown(f"**{label_text}**")
            with gr.Row():
                # Second row: dropdowns
                prediction_columns = {}
                for label_text in ["Start Time", "End Time", "Class", "Confidence", "Recording", "Duration"]:
                    prediction_columns[label_text] = gr.Dropdown(choices=[], label="", show_label=False)

            gr.Row()  # Add empty row for spacing

            # Mapping and Select Classes/Recordings
            # Class mapping in one row
            gr.Markdown("### Class Mapping")
            with gr.Row():
                download_mapping_button = gr.Button("Download Template")
                mapping_label = gr.Markdown("**Class Mapping (Optional):**")
                mapping_input = gr.File(label="Drag and drop or click to select mapping file", file_types=['.json'])

            gr.Row()  # Add empty row for spacing

            # Select Classes and Recordings in another row
            with gr.Row():
                select_classes_label = gr.Markdown("**Select Classes:**")
                select_classes_combobox = gr.Dropdown(choices=[], multiselect=True, label="")
                select_recordings_label = gr.Markdown("**Select Recordings:**")
                select_recordings_combobox = gr.Dropdown(choices=[], multiselect=True, label="")

        gr.Row()  # Add empty row for spacing

        # Parameters and Metrics layout
        # Parameters
        gr.Markdown("### Parameters")
        with gr.Row():
            sample_duration = gr.Number(value=3, label="Sample Duration (s)", precision=0)
            recording_duration = gr.Textbox(label="Recording Duration (s)", placeholder="Determined from files")
            min_overlap = gr.Number(value=0.5, label="Minimum Overlap (s)")
            threshold = gr.Slider(minimum=0.01, maximum=0.99, value=0.1, label="Threshold")
            class_wise = gr.Checkbox(label="Class-wise Metrics", value=False)

        gr.Row()  # Add empty row for spacing

        # Metrics
        gr.Markdown("### Metrics")
        with gr.Row():
            # Metric checkboxes with hover text using 'info' parameter
            metric_info = {
                "AUROC": """AUROC measures the probability that the model will rank a random positive case higher than a random negative case.""",
                "Precision": """Precision measures how often the model's positive predictions are actually correct.""",
                "Recall": """Recall measures the percentage of positive cases that the model successfully identifies for each class.""",
                "F1 Score": """The F1 score is the harmonic mean of precision and recall, providing a balance between the two.""",
                "Average Precision (AP)": """Average Precision summarizes the precision-recall curve by averaging the precision values across all recall levels.""",
                "Accuracy": """Accuracy measures the percentage of times the model correctly predicts the correct class.""",
            }
            metrics_checkboxes = {}
            for metric_name, description in metric_info.items():
                metrics_checkboxes[metric_name.lower()] = gr.Checkbox(label=metric_name, value=True, info=description)

        gr.Row()  # Add empty row for spacing

        # Actions
        gr.Markdown("### Actions")
        with gr.Row():
            calculate_button = gr.Button("Calculate Metrics")
            plot_metrics_button = gr.Button("Plot Metrics")
            plot_confusion_button = gr.Button("Plot Confusion Matrix")
            plot_metrics_all_thresholds_button = gr.Button("Plot Metrics All Thresholds")

        gr.Row()  # Add empty row for spacing

        # Download buttons layout
        with gr.Row():
            download_results_button = gr.Button("Download Results Table")
            download_data_button = gr.Button("Download Data Table")

        # Results area
        results_text = gr.Textbox(label="Results", lines=10, visible=False)
        plot_output = gr.Plot(label="Plot", visible=False)
        download_results_file = gr.File(label="Download Results Table", visible=False)
        download_data_file = gr.File(label="Download Data Table", visible=False)
        mapping_template_file = gr.File(label="Download Mapping Template", visible=False)

        # Functions and event handlers

        # Function to download the mapping template
        def download_class_mapping_template():
            # Create a descriptive template dictionary
            template_mapping = {
                "Predicted Class Name 1": "Annotation Class Name 1",
                "Predicted Class Name 2": "Annotation Class Name 2",
                "Predicted Class Name 3": "Annotation Class Name 3",
                "Predicted Class Name 4": "Annotation Class Name 4",
                "Predicted Class Name 5": "Annotation Class Name 5",
            }
            temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".json")
            with open(temp_file.name, "w") as f:
                json.dump(template_mapping, f, indent=4)
            return temp_file.name, gr.update(visible=True)

        download_mapping_button.click(
            download_class_mapping_template,
            outputs=[mapping_template_file]
        )

        def get_columns_from_files(path):
            columns = set()
            if os.path.isfile(path):
                try:
                    df = pd.read_csv(path, sep=None, engine="python", nrows=0)
                    columns.update(df.columns)
                except Exception as e:
                    print(f"Error reading file {path}: {e}")
            elif os.path.isdir(path):
                for filename in os.listdir(path):
                    filepath = os.path.join(path, filename)
                    if os.path.isfile(filepath) and filename.endswith((".txt", ".csv")):
                        try:
                            df = pd.read_csv(filepath, sep=None, engine="python", nrows=0)
                            columns.update(df.columns)
                        except Exception as e:
                            print(f"Error reading file {filepath}: {e}")
            return sorted(list(columns))

        def update_annotation_columns(annotation_path):
            columns = get_columns_from_files(annotation_path)
            updates = []
            for label_text in ["Start Time", "End Time", "Class", "Recording", "Duration"]:
                default_value = annotation_default_columns.get(label_text)
                value = default_value if default_value in columns else None
                updates.append(gr.update(choices=columns, value=value))
            return updates

        def update_prediction_columns(prediction_path):
            columns = get_columns_from_files(prediction_path)
            updates = []
            for label_text in ["Start Time", "End Time", "Class", "Confidence", "Recording", "Duration"]:
                default_value = prediction_default_columns.get(label_text)
                value = default_value if default_value in columns else None
                updates.append(gr.update(choices=columns, value=value))
            return updates

        annotation_input.change(
            update_annotation_columns,
            inputs=annotation_input,
            outputs=[annotation_columns[label_text] for label_text in ["Start Time", "End Time", "Class", "Recording", "Duration"]]
        )

        prediction_input.change(
            update_prediction_columns,
            inputs=prediction_input,
            outputs=[prediction_columns[label_text] for label_text in ["Start Time", "End Time", "Class", "Confidence", "Recording", "Duration"]]
        )

        def initialize_processor(
            annotation_path, prediction_path, mapping_input, sample_duration_value, min_overlap_value, recording_duration,
            ann_start_time, ann_end_time, ann_class, ann_recording, ann_duration,
            pred_start_time, pred_end_time, pred_class, pred_confidence, pred_recording, pred_duration
        ):
            # Map columns
            columns_annotations = {
                "Start Time": ann_start_time,
                "End Time": ann_end_time,
                "Class": ann_class,
                "Recording": ann_recording,
                "Duration": ann_duration,
            }
            columns_predictions = {
                "Start Time": pred_start_time,
                "End Time": pred_end_time,
                "Class": pred_class,
                "Confidence": pred_confidence,
                "Recording": pred_recording,
                "Duration": pred_duration,
            }

            # Prepare mapping
            if mapping_input is not None:
                mapping_input.seek(0)
                class_mapping = json.load(mapping_input)
            else:
                class_mapping = None

            try:
                # Initialize DataProcessor
                annotation_dir = annotation_path
                prediction_dir = prediction_path

                processor = DataProcessor(
                    prediction_directory_path=prediction_dir,
                    prediction_file_name=None,
                    annotation_directory_path=annotation_dir,
                    annotation_file_name=None,
                    class_mapping=class_mapping,
                    sample_duration=sample_duration_value,
                    min_overlap=min_overlap_value,
                    columns_predictions=columns_predictions,
                    columns_annotations=columns_annotations,
                    recording_duration=recording_duration,
                )

                # Get available classes and recordings
                available_classes = processor.classes
                available_recordings = processor.samples_df["filename"].unique().tolist()

                return (
                    gr.update(choices=available_classes, value=available_classes),
                    gr.update(choices=available_recordings, value=available_recordings),
                    processor
                )
            except Exception as e:
                print(f"Error initializing processor: {e}")
                return (
                    gr.update(choices=[], value=[]),
                    gr.update(choices=[], value=[]),
                    None
                )

        def calculate_metrics(
            annotation_path, prediction_path, mapping_input, sample_duration_value, min_overlap_value, recording_duration_value,
            ann_start_time, ann_end_time, ann_class, ann_recording, ann_duration,
            pred_start_time, pred_end_time, pred_class, pred_confidence, pred_recording, pred_duration,
            threshold_value, class_wise_value,
            *metrics_checkbox_values
        ):
            # Get selected metrics
            selected_metrics = []
            for value, (metric_name_lower, _) in zip(metrics_checkbox_values, metrics_checkboxes.items()):
                if value:
                    selected_metrics.append(metric_name_lower)
            valid_metrics = {
                "accuracy": "accuracy",
                "recall": "recall",
                "precision": "precision",
                "f1 score": "f1",
                "average precision (ap)": "ap",
                "auroc": "auroc",
            }
            metrics = tuple([valid_metrics[m] for m in selected_metrics if m in valid_metrics])

            # Get selected classes and recordings
            selected_classes_list = select_classes_combobox.value
            if not selected_classes_list:
                selected_classes_list = None  # Select all if none selected

            selected_recordings_list = select_recordings_combobox.value
            if not selected_recordings_list:
                selected_recordings_list = None  # Select all if none selected

            # Parse recording_duration_value
            if recording_duration_value.strip() == "":
                recording_duration = None
            else:
                try:
                    recording_duration = float(recording_duration_value)
                except ValueError:
                    return "Please enter a valid number for Recording Duration.", None, None, None, None, None, None, gr.update(visible=True)

            class_update, recording_update, processor = initialize_processor(
                annotation_path, prediction_path, mapping_input, sample_duration_value, min_overlap_value, recording_duration,
                ann_start_time, ann_end_time, ann_class, ann_recording, ann_duration,
                pred_start_time, pred_end_time, pred_class, pred_confidence, pred_recording, pred_duration
            )

            if processor is None:
                return "Error initializing processor.", None, None, None, class_update, recording_update, None, gr.update(visible=True)

            # Threshold
            threshold = threshold_value

            # Class-wise metrics
            class_wise = class_wise_value

            try:
                metrics_df, pa, predictions, labels = process_data(
                    annotation_path=annotation_path,
                    prediction_path=prediction_path,
                    mapping_path=mapping_input,
                    sample_duration=sample_duration_value,
                    min_overlap=min_overlap_value,
                    recording_duration=recording_duration,
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
                    threshold=threshold,
                    class_wise=class_wise,
                )
                result_text = metrics_df.to_string()
                return result_text, pa, predictions, labels, class_update, recording_update, processor, gr.update(visible=True)
            except Exception as e:
                result_text = f"Error processing data: {e}"
                return result_text, None, None, None, class_update, recording_update, None, gr.update(visible=True)

        calculate_button.click(
            calculate_metrics,
            inputs=[
                annotation_input, prediction_input, mapping_input, sample_duration, min_overlap, recording_duration,
                annotation_columns["Start Time"], annotation_columns["End Time"], annotation_columns["Class"],
                annotation_columns["Recording"], annotation_columns["Duration"],
                prediction_columns["Start Time"], prediction_columns["End Time"], prediction_columns["Class"],
                prediction_columns["Confidence"], prediction_columns["Recording"], prediction_columns["Duration"],
                threshold, class_wise
            ] + [checkbox for checkbox in metrics_checkboxes.values()],
            outputs=[results_text, pa_state, predictions_state, labels_state,
                     select_classes_combobox, select_recordings_combobox, processor_state, results_text]
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

        def download_results_table(pa, predictions, labels, class_wise_value):
            if pa is None or predictions is None or labels is None:
                return None, gr.update(visible=False)
            try:
                metrics_df = pa.calculate_metrics(predictions, labels, per_class_metrics=class_wise_value)
                temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".csv")
                metrics_df.to_csv(temp_file.name, index=True)
                return temp_file.name, gr.update(visible=True)
            except Exception as e:
                print(f"Error saving results table: {e}")
                return None, gr.update(visible=False)

        download_results_button.click(
            download_results_table,
            inputs=[pa_state, predictions_state, labels_state, class_wise],
            outputs=[download_results_file]
        )

        def download_data_table(processor):
            if processor is None:
                return None, gr.update(visible=False)
            try:
                data_df = processor.get_sample_data()
                temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".csv")
                data_df.to_csv(temp_file.name, index=False)
                return temp_file.name, gr.update(visible=True)
            except Exception as e:
                print(f"Error saving data table: {e}")
                return None, gr.update(visible=False)

        download_data_button.click(
            download_data_table,
            inputs=[processor_state],
            outputs=[download_data_file]
        )

    return build_evaluation_tab  # Return the function to be called in main.py
