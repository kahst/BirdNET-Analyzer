# evaluation.py

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
        # Step 1: Create a temporary file with mkstemp.
        fd, temp_path = tempfile.mkstemp(suffix=".json")
        # Step 2: Write the JSON data into the file.
        with os.fdopen(fd, 'w') as f:
            json.dump(template_mapping, f, indent=4)
        # Step 3: Define the desired file name in the same directory.
        desired_path = os.path.join(os.path.dirname(temp_path), "class_mapping_template.json")

        # If a file with this name already exists, remove it.
        if os.path.exists(desired_path):
            os.remove(desired_path)

        # Step 4: Rename the temporary file to the desired name.
        os.rename(temp_path, desired_path)
        # Step 5: Return the update with the new file path.
        return gr.update(value=desired_path, visible=True)

    def download_results_table(pa, predictions, labels, class_wise_value):
        if pa is None or predictions is None or labels is None:
            return None  # No file to return
        try:
            metrics_df = pa.calculate_metrics(predictions, labels, per_class_metrics=class_wise_value)
            temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".csv")
            metrics_df.to_csv(temp_file.name, index=True)
            temp_file.close()  # Make sure the file is closed

            # Rename the temporary file to a fixed name in the same directory.
            desired_path = os.path.join(os.path.dirname(temp_file.name), "results_table.csv")

            # If a file with this name already exists, remove it.
            if os.path.exists(desired_path):
                os.remove(desired_path)

            os.rename(temp_file.name, desired_path)

            return gr.update(value=desired_path)
        except Exception as e:
            print(f"Error saving results table: {e}")
            return None

    def download_data_table(processor):
        if processor is None:
            return None
        try:
            data_df = processor.get_sample_data()
            temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".csv")
            data_df.to_csv(temp_file.name, index=False)
            temp_file.close()

            # Rename the temporary file to a fixed name in the same directory.
            desired_path = os.path.join(os.path.dirname(temp_file.name), "data_table.csv")

            # If a file with this name already exists, remove it.
            if os.path.exists(desired_path):
                os.remove(desired_path)

            os.rename(temp_file.name, desired_path)

            return gr.update(value=desired_path)
        except Exception as e:
            print(f"Error saving data table: {e}")
            return None

    # Helper function to extract columns from uploaded files
    def get_columns_from_uploaded_files(files):
        columns = set()
        if files:
            for file_obj in files:
                try:
                    # Here we assume file_obj has a .name attribute (per Gradio docs)
                    df = pd.read_csv(file_obj.name, sep=None, engine="python", nrows=0)
                    columns.update(df.columns)
                except Exception as e:
                    print(f"Error reading file {file_obj.name}: {e}")
        return sorted(list(columns))

    # Helper function to save uploaded files into a temporary directory
    def save_uploaded_files(files):
        if not files:
            return None
        temp_dir = tempfile.mkdtemp()
        for file_obj in files:
            dest_path = os.path.join(temp_dir, os.path.basename(file_obj.name))
            shutil.copy(file_obj.name, dest_path)
        return temp_dir

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
            with gr.Row():
                # Annotations Files selection
                with gr.Column():
                    annotation_label = gr.Markdown("**Annotations Files:**")
                    annotation_files = gr.File(
                        label="Select Annotation Files",
                        file_count="multiple",
                        file_types=[".csv", ".txt"]
                    )
                # Predictions Files selection
                with gr.Column():
                    prediction_label = gr.Markdown("**Predictions Files:**")
                    prediction_files = gr.File(
                        label="Select Prediction Files",
                        file_count="multiple",
                        file_types=[".csv", ".txt"]
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
                download_mapping_button = gr.DownloadButton(
                    label="Download Template",
                    visible=True
                )

                mapping_label = gr.Markdown("**Class Mapping (Optional):**")
                mapping_file = gr.File(label="Upload Mapping File", file_count="single", file_types=[".json"])
            download_mapping_button.click(
                fn=download_class_mapping_template,
                inputs=[],
                outputs=download_mapping_button
            )

            gr.Row()  # Add empty row for spacing

            # Select Classes and Recordings in another row
            with gr.Row():
                select_classes_label = gr.Markdown("**Select Classes:**")
                select_classes_combobox = gr.Dropdown(choices=[], multiselect=True, label="")
                select_recordings_label = gr.Markdown("**Select Recordings:**")
                select_recordings_combobox = gr.Dropdown(choices=[], multiselect=True, label="")

        gr.Row()  # Add empty row for spacing

        # Parameters and Metrics layout
        gr.Markdown("### Parameters")
        with gr.Row():
            sample_duration = gr.Number(value=3, label="Sample Duration (s)", precision=0)
            recording_duration = gr.Textbox(label="Recording Duration (s)", placeholder="Determined from files")
            min_overlap = gr.Number(value=0.5, label="Minimum Overlap (s)")
            threshold = gr.Slider(minimum=0.01, maximum=0.99, value=0.1, label="Threshold")
            class_wise = gr.Checkbox(label="Class-wise Metrics", value=False)

        gr.Row()  # Add empty row for spacing

        gr.Markdown("### Metrics")
        with gr.Row():
            # Metric checkboxes with hover text using 'info' parameter
            metric_info = {
                "AUROC": "AUROC measures the probability that the model will rank a random positive case higher than a random negative case.",
                "Precision": "Precision measures how often the model's positive predictions are actually correct.",
                "Recall": "Recall measures the percentage of positive cases that the model successfully identifies for each class.",
                "F1 Score": "The F1 score is the harmonic mean of precision and recall, providing a balance between the two.",
                "Average Precision (AP)": "Average Precision summarizes the precision-recall curve by averaging the precision values across all recall levels.",
                "Accuracy": "Accuracy measures the percentage of times the model correctly predicts the correct class.",
            }
            metrics_checkboxes = {}
            for metric_name, description in metric_info.items():
                metrics_checkboxes[metric_name.lower()] = gr.Checkbox(label=metric_name, value=True, info=description)

        gr.Row()  # Add empty row for spacing

        gr.Markdown("### Actions")
        with gr.Row():
            calculate_button = gr.Button("Calculate Metrics")
            plot_metrics_button = gr.Button("Plot Metrics")
            plot_confusion_button = gr.Button("Plot Confusion Matrix")
            plot_metrics_all_thresholds_button = gr.Button("Plot Metrics All Thresholds")

        gr.Row()  # Add empty row for spacing

        with gr.Row():
            download_results_button = gr.DownloadButton(
                label="Download Results Table",
                visible=True
            )

            download_data_button = gr.DownloadButton(
                label="Download Data Table",
                visible=True
            )

        download_results_button.click(
            fn=download_results_table,
            inputs=[pa_state, predictions_state, labels_state, class_wise],
            outputs=download_results_button
        )

        download_data_button.click(
            fn=download_data_table,
            inputs=[processor_state],
            outputs=download_data_button
        )

        results_text = gr.Textbox(label="Results", lines=10, visible=False)
        plot_output = gr.Plot(label="Plot", visible=False)

        # Functions and event handlers

        def update_annotation_columns(uploaded_files):
            columns = get_columns_from_uploaded_files(uploaded_files)
            updates = []
            for label_text in ["Start Time", "End Time", "Class", "Recording", "Duration"]:
                default_value = annotation_default_columns.get(label_text)
                value = default_value if default_value in columns else None
                updates.append(gr.update(choices=columns, value=value))
            return updates

        def update_prediction_columns(uploaded_files):
            columns = get_columns_from_uploaded_files(uploaded_files)
            updates = []
            for label_text in ["Start Time", "End Time", "Class", "Confidence", "Recording", "Duration"]:
                default_value = prediction_default_columns.get(label_text)
                value = default_value if default_value in columns else None
                updates.append(gr.update(choices=columns, value=value))
            return updates

        annotation_files.change(
            update_annotation_columns,
            inputs=annotation_files,
            outputs=[annotation_columns[label_text] for label_text in
                     ["Start Time", "End Time", "Class", "Recording", "Duration"]]
        )

        prediction_files.change(
            update_prediction_columns,
            inputs=prediction_files,
            outputs=[prediction_columns[label_text] for label_text in
                     ["Start Time", "End Time", "Class", "Confidence", "Recording", "Duration"]]
        )

        def initialize_processor(
                annotation_files, prediction_files, mapping_file_obj, sample_duration_value, min_overlap_value,
                recording_duration,
                ann_start_time, ann_end_time, ann_class, ann_recording, ann_duration,
                pred_start_time, pred_end_time, pred_class, pred_confidence, pred_recording, pred_duration
        ):
            # Save uploaded files to temporary directories
            annotation_dir = save_uploaded_files(annotation_files)
            prediction_dir = save_uploaded_files(prediction_files)

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

            # Prepare mapping using our helper
            if mapping_file_obj.temp_files:
                mapping_path = list(mapping_file_obj.temp_files)[0]
            else:
                mapping_path = None

            if mapping_path:
                with open(mapping_path, 'r') as f:
                    class_mapping = json.load(f)
            else:
                class_mapping = None

            try:
                # Initialize DataProcessor with temporary directories
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
                annotation_files, prediction_files, mapping_file_obj, sample_duration_value, min_overlap_value,
                recording_duration_value,
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
                annotation_files, prediction_files, mapping_file, sample_duration_value, min_overlap_value,
                recording_duration,
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
                    annotation_path=save_uploaded_files(annotation_files),
                    prediction_path=save_uploaded_files(prediction_files),
                    mapping_path=list(mapping_file.temp_files)[0] if mapping_file.temp_files else None,
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
                annotation_files, prediction_files, mapping_file, sample_duration, min_overlap, recording_duration,
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
