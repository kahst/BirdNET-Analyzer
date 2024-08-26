import os
import csv
import numpy as np
from tqdm import tqdm
from scipy.optimize import minimize_scalar
import argparse
import concurrent.futures
import multiprocessing

import audio
import config as cfg
import utils
import analyze

def aggregate_predictions(chunk_predictions, method='average'):
    """Aggregate predictions across all chunks of a file."""
    if method == 'average':
        return np.mean(chunk_predictions, axis=0)
    elif method == 'max':
        return np.max(chunk_predictions, axis=0)
    elif method == 'lme':
        return np.log(np.mean(np.exp(chunk_predictions), axis=0))
    else:
        raise ValueError("Invalid aggregation method. Choose from 'average', 'max', or 'lme'.")

def calculate_sparse_accuracy_on_positives(y_true, y_pred):
    """Calculate accuracy only on relevant (non-zero) labels in multilabel classification."""
    # Consider only the positions where y_true or y_pred is 1
    relevant_positions = (y_true == 1) | (y_pred == 1)
    
    # Calculate correct predictions at these relevant positions
    correct_predictions = np.sum((y_true == y_pred) & relevant_positions)
    
    # Calculate total relevant positions
    total_relevant = np.sum(relevant_positions)
    
    # Avoid division by zero
    if total_relevant == 0:
        return 1.0  # If no relevant positions, define accuracy as 1.0 (perfect accuracy by convention)
    
    # Calculate accuracy as the proportion of correct predictions in relevant positions
    return correct_predictions / total_relevant

def calculate_precision(y_true, y_pred):
    """Calculate precision for multilabel classification."""
    true_positives = np.sum(y_true * y_pred, axis=0)
    predicted_positives = np.sum(y_pred, axis=0)
    
    # Calculate precision, but only where predicted_positives is greater than 0
    precision_per_class = np.divide(true_positives, predicted_positives, where=predicted_positives != 0)
    
    # Only consider classes where there was at least one predicted positive
    valid_classes = predicted_positives > 0
    
    if np.sum(valid_classes) == 0:
        return 0.0  # If no valid classes, return zero precision by convention
    
    return np.mean(precision_per_class[valid_classes])

def calculate_recall(y_true, y_pred):
    """Calculate recall for multilabel classification."""
    true_positives = np.sum(y_true * y_pred, axis=0)
    actual_positives = np.sum(y_true, axis=0)
    
    # Calculate recall, but only where actual_positives is greater than 0
    recall_per_class = np.divide(true_positives, actual_positives, where=actual_positives != 0)
    
    # Only consider classes where there was at least one actual positive
    valid_classes = actual_positives > 0
    
    if np.sum(valid_classes) == 0:
        return 0.0  # If no valid classes, return zero recall by convention
    
    return np.mean(recall_per_class[valid_classes])

def calculate_f1_score(precision, recall):
    """Calculate F1 score based on precision and recall."""
    if (precision + recall) == 0:
        return 0
    return 2 * (precision * recall) / (precision + recall)

def roc_auc_single_class(y_true, y_scores):
    """Compute ROC AUC for a single class."""
    # Sort scores and corresponding true labels
    sorted_indices = np.argsort(y_scores)[::-1]
    y_true_sorted = y_true[sorted_indices]

    tps = np.cumsum(y_true_sorted)
    fps = np.cumsum(1 - y_true_sorted)
    
    if tps[-1] == 0:
        tpr = np.zeros_like(tps)
    else:
        tpr = tps / tps[-1]

    if fps[-1] == 0:
        fpr = np.zeros_like(fps)
    else:
        fpr = fps / fps[-1]

    auc = np.trapz(tpr, fpr)
    return auc

def roc_auc_multilabel(y_true, y_scores):
    """Compute ROC AUC for multilabel classification."""
    aucs = []
    for i in range(y_true.shape[1]):
        if np.sum(y_true[:, i]) == 0:
            continue  # Skip if there are no true positives for this class
        
        auc = roc_auc_single_class(y_true[:, i], y_scores[:, i])
        aucs.append(auc)
    
    if len(aucs) == 0:
        return 1.0  # If no valid classes, return perfect ROC-AUC by convention
    
    return np.mean(aucs)

def calculate_metrics(y_true, y_pred, y_pred_proba=None):
    """Calculate performance metrics manually using NumPy."""
    sparse_accuracy = calculate_sparse_accuracy_on_positives(y_true, y_pred)
    precision = calculate_precision(y_true, y_pred)
    recall = calculate_recall(y_true, y_pred)
    f1_score = calculate_f1_score(precision, recall)

    metrics = {
        'sparse_accuracy_on_positives': sparse_accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1_score
    }

    if y_pred_proba is not None:
        metrics['roc_auc'] = roc_auc_multilabel(y_true, y_pred_proba)
    
    return metrics

def optimize_threshold(y_true, y_pred_proba):
    """Optimize threshold to maximize F1-score."""
    
    def f1_for_threshold(threshold):
        """Calculate F1-score for a given threshold."""
        y_pred = (y_pred_proba >= threshold).astype(int)
        precision = calculate_precision(y_true, y_pred)
        recall = calculate_recall(y_true, y_pred)
        return -calculate_f1_score(precision, recall)  # Negate to maximize

    result = minimize_scalar(f1_for_threshold, bounds=(0, 1), method='bounded')
    return result.x  # Best threshold

def pretty_print_metrics(metrics):
    """Pretty print the calculated metrics."""
    print("\nPerformance Metrics:")
    for metric_name, metric_value in metrics.items():
        print(f"{metric_name.replace('_', ' ').capitalize()}: {metric_value:.4f}")

def process_file(afile, pooling_method, threshold):
    """Process a single audio file, apply a threshold, and return the true label, binary predictions, and raw probabilities."""
    # Extract the label from the file path
    label = os.path.basename(os.path.dirname(afile))
    label_idx = cfg.LABELS.index(label)
    
    # One-hot encode the true label
    true_label = np.zeros(len(cfg.LABELS))
    true_label[label_idx] = 1
    
    # Load and process the audio file to get chunk predictions
    chunks = analyze.getRawAudioFromFile(afile, 0, 60)
    chunk_predictions = analyze.predict(chunks)
    
    # Aggregate predictions for the whole file
    aggregated_prediction = aggregate_predictions(chunk_predictions, method=pooling_method)
    
    # Apply the threshold to convert probabilities to binary predictions
    pred_binary = (aggregated_prediction >= threshold).astype(int)
    
    return true_label, pred_binary, aggregated_prediction

def process_files(args):
    """Process audio files and return true labels and predicted probabilities."""
    # Set configuration parameters based on CLI arguments
    cfg.SIG_OVERLAP = args.overlap
    cfg.TFLITE_THREADS = 4 # We'll set this to 4 by default, seems to be faster for small files
    cfg.LABELS = utils.readLines(cfg.LABELS_FILE)
    
    if args.slist:
        cfg.SPECIES_LIST = utils.readLines(args.slist)
    else:
        cfg.SPECIES_LIST = cfg.LABELS
        
    print(f"Species list contains {len(cfg.SPECIES_LIST)} species.")
    
    # Set bandpass frequency range
    cfg.BANDPASS_FMIN = max(0, min(cfg.SIG_FMAX, int(args.fmin)))
    cfg.BANDPASS_FMAX = max(cfg.SIG_FMIN, min(cfg.SIG_FMAX, int(args.fmax)))

    # Set custom classifier?
    if args.classifier is not None:
        cfg.CUSTOM_CLASSIFIER = args.classifier  # we treat this as absolute path, so no need to join with dirname

        if args.classifier.endswith(".tflite"):
            cfg.LABELS_FILE = args.classifier.replace(".tflite", "_Labels.txt")  # same for labels file
            cfg.LABELS = utils.readLines(cfg.LABELS_FILE)
        else:
            cfg.APPLY_SIGMOID = False
            cfg.LABELS_FILE = os.path.join(args.classifier, "labels", "label_names.csv")
            cfg.LABELS = [line.split(",")[1] for line in utils.readLines(cfg.LABELS_FILE)]

    # Collect all audio file paths from the test directory
    afiles = utils.shuffle_list(utils.collect_audio_files(args.i, labels=cfg.SPECIES_LIST))[:args.num_files]

    y_true = []
    y_pred_proba = []

    # Use multiprocessing to process files in parallel
    with concurrent.futures.ProcessPoolExecutor(max_workers=args.threads) as executor:
        with tqdm(total=len(afiles), desc="Processing files") as pbar:
            futures = {executor.submit(process_file, afile, args.pooling_method, 0.5): afile for afile in afiles}  # No threshold applied yet
            for future in concurrent.futures.as_completed(futures):
                true_label, _, prediction = future.result()
                
                y_true.append(true_label)
                y_pred_proba.append(prediction)
                pbar.update(1)

    y_true = np.array(y_true)
    y_pred_proba = np.array(y_pred_proba)

    return y_true, y_pred_proba

def apply_threshold(y_pred_proba, threshold):
    """Apply the threshold to predicted probabilities to get binary predictions."""
    return (y_pred_proba >= threshold).astype(int)

def apply_thresholds_per_class(y_pred_proba, thresholds):
    """Apply class-specific thresholds to predicted probabilities."""
    y_pred = np.zeros_like(y_pred_proba)
    for i, threshold in enumerate(thresholds):
        y_pred[:, i] = (y_pred_proba[:, i] >= threshold).astype(int)
    return y_pred

def optimize_threshold_per_class(y_true, y_pred_proba):
    """Optimize threshold for each class to maximize F1-score."""
    num_classes = y_true.shape[1]
    best_thresholds = np.zeros(num_classes)

    for i in range(num_classes):
        def f1_for_threshold(threshold):
            """Calculate F1-score for a given threshold for a specific class."""
            y_pred_class = (y_pred_proba[:, i] >= threshold).astype(int)
            precision = calculate_precision(y_true[:, [i]], y_pred_class[:, np.newaxis])
            recall = calculate_recall(y_true[:, [i]], y_pred_class[:, np.newaxis])
            return -calculate_f1_score(precision, recall)  # Negate to maximize

        result = minimize_scalar(f1_for_threshold, bounds=(0, 1), method='bounded')
        best_thresholds[i] = result.x
        
    # Limit best thresholds to [0.01, 0.99]
    best_thresholds = np.clip(best_thresholds, 0.0001, 0.9999)
    
    return best_thresholds

def calculate_and_print_metrics(y_true, y_pred, y_pred_proba=None):
    """Calculate performance metrics and pretty print them."""
    metrics = calculate_metrics(y_true, y_pred, y_pred_proba)
    pretty_print_metrics(metrics)
    
def calculate_per_class_metrics(y_true, y_pred, thresholds, is_global=False):
    """Calculate per-class precision, recall, and F1-score."""
    num_classes = y_true.shape[1]
    per_class_metrics = []

    for i in range(num_classes):
        precision = calculate_precision(y_true[:, [i]], y_pred[:, [i]])
        recall = calculate_recall(y_true[:, [i]], y_pred[:, [i]])
        f1_score = calculate_f1_score(precision, recall)
        
        # Check if we have at least one true label for this class
        if np.sum(y_true[:, i]) > 0:        
            per_class_metrics.append({
                'Class': cfg.LABELS[i],
                'Samples': np.sum(y_true[:, i]),
                'Threshold': thresholds if is_global else thresholds[i],
                'Precision': precision,
                'Recall': recall,
                'F1-Score': f1_score
            })
    
    return per_class_metrics

def save_per_class_metrics(per_class_metrics, output_file='per_class_metrics.csv'):
    """Save per-class metrics to a CSV file using Python's built-in csv module."""
    with open(output_file, mode='w', newline='') as file:
        writer = csv.writer(file)
        # Write the header
        writer.writerow(['Class', 'Samples', 'Threshold', 'Precision', 'Recall', 'F1-Score'])
        # Write the data
        for metric in per_class_metrics:
            writer.writerow([
                metric['Class'], 
                metric['Samples'],
                f"{metric['Threshold']:.4f}", 
                f"{metric['Precision']:.4f}", 
                f"{metric['Recall']:.4f}", 
                f"{metric['F1-Score']:.4f}"
            ])
    print(f"Per-class metrics saved to {output_file}")

def parse_arguments():
    """Parse command-line arguments using argparse."""
    parser = argparse.ArgumentParser(description="Audio Classification and Analysis")

    parser.add_argument('--i', type=str, default='example/', help='Path to the test directory containing audio files with subfolders as label names. Defaults to "example/".')
    parser.add_argument('--classifier', type=str, default=None, help='Path to custom trained classifier. Defaults to None. If set, --lat, --lon and --locale are ignored.')
    parser.add_argument('--overlap', type=float, default=0.0, help='Overlap of prediction segments. Values in [0.0, 2.9]. Defaults to 0.0.')
    parser.add_argument('--pooling_method', type=str, default='lme', choices=['average', 'max', 'lme'], help='Method for aggregating chunk predictions.')
    parser.add_argument('--num_files', type=int, default=100, help='Number of files to process. Defaults to all files.')
    parser.add_argument('--threads', type=int, default=min(8, max(1, multiprocessing.cpu_count() // 2)), help='Number of parallel workers threads for multiprocessing.')
    parser.add_argument('--min_conf', type=str, default='0.1', help='Confidence threshold for converting probabilities to binary predictions. Use "auto" to optimize globally, "auto_class" to optimize per class.')
    parser.add_argument(
        "--fmin",
        type=int,
        default=cfg.SIG_FMIN,
        help=f"Minimum frequency for bandpass filter in Hz. Defaults to {cfg.SIG_FMIN} Hz.",
    )
    parser.add_argument(
        "--fmax",
        type=int,
        default=cfg.SIG_FMAX,
        help=f"Maximum frequency for bandpass filter in Hz. Defaults to {cfg.SIG_FMAX} Hz.",
    )
    parser.add_argument(
        "--slist",
        default="",
        help='Path to species list file or folder. If folder is provided, species list needs to be named "species_list.txt". If lat and lon are provided, this list will be ignored.',
    )
    parser.add_argument('--output', type=str, default='per_class_metrics.csv', help='Path to save per-class metrics and thresholds as a CSV file.')

    return parser.parse_args()

if __name__ == '__main__':
    
    args = parse_arguments()
    
    # Step 1: Process the files to get true labels and predicted probabilities
    y_true, y_pred_proba = process_files(args)
    
    # Step 2: Determine the best threshold(s)
    if args.min_conf == 'auto':
        best_threshold = optimize_threshold(y_true, y_pred_proba)
        y_pred = apply_threshold(y_pred_proba, best_threshold)
        print(f"Optimized Threshold: {best_threshold:.4f}")
    elif args.min_conf == 'auto_class':
        best_thresholds = optimize_threshold_per_class(y_true, y_pred_proba)
        y_pred = apply_thresholds_per_class(y_pred_proba, best_thresholds)
        
        # Print lowest and highest thresholds (if class has a true label)
        min_threshold = np.min(best_thresholds[y_true.sum(axis=0) > 0])
        max_threshold = np.max(best_thresholds[y_true.sum(axis=0) > 0])
        min_threshold_class = cfg.LABELS[np.argmin(best_thresholds[y_true.sum(axis=0) > 0])]
        max_threshold_class = cfg.LABELS[np.argmax(best_thresholds[y_true.sum(axis=0) > 0])]
        print(f"Optimized Thresholds: Min={min_threshold:.4f} ({min_threshold_class}), Max={max_threshold:.4f} ({max_threshold_class})")            
        
    else:
        best_threshold = float(args.min_conf)
        y_pred = apply_threshold(y_pred_proba, best_threshold)
    
    # Step 3: Calculate and print the metrics
    calculate_and_print_metrics(y_true, y_pred, y_pred_proba)
    
    # Step 4: Calculate and save per-class metrics
    if args.output:
        
        if args.min_conf == 'auto_class':
            per_class_metrics = calculate_per_class_metrics(y_true, y_pred, best_thresholds, is_global=False)
        else:
            per_class_metrics = calculate_per_class_metrics(y_true, y_pred, best_threshold, is_global=True)
        
        save_per_class_metrics(per_class_metrics, args.output)



