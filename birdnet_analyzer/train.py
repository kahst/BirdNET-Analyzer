"""Module for training a custom classifier.

Can be used to train a custom classifier with new training data.
"""

import argparse
import csv
import multiprocessing
import os
from functools import partial
from multiprocessing.pool import Pool

import numpy as np
import tqdm

import birdnet_analyzer.audio as audio
import birdnet_analyzer.config as cfg
import birdnet_analyzer.model as model
import birdnet_analyzer.utils as utils

SCRIPT_DIR = os.path.abspath(os.path.dirname(__file__))


def save_sample_counts(labels, y_train):
    samples_per_label = {}
    label_combinations = np.unique(y_train, axis=0)

    for label_combination in label_combinations:
        label = "+".join([labels[i] for i in range(len(label_combination)) if label_combination[i] == 1])
        samples_per_label[label] = np.sum(np.all(y_train == label_combination, axis=1))

    csv_file_path = cfg.CUSTOM_CLASSIFIER + "_sample_counts.csv"

    with open(csv_file_path, mode="w", newline="") as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(["Label", "Count"])

        for label, count in samples_per_label.items():
            writer.writerow([label, count])


def _loadAudioFile(f, label_vector, config):
    """Load an audio file and extract features.
    Args:
        f: Path to the audio file.
        label_vector: The label vector for the file.
    Returns:
        A tuple of (x_train, y_train).
    """

    x_train = []
    y_train = []

    # restore config in case we're on Windows to be thread save
    cfg.setConfig(config)

    # Try to load the audio file
    try:
        # Load audio
        sig, rate = audio.openAudioFile(
            f,
            duration=cfg.SIG_LENGTH if cfg.SAMPLE_CROP_MODE == "first" else None,
            fmin=cfg.BANDPASS_FMIN,
            fmax=cfg.BANDPASS_FMAX,
        )

    # if anything happens print the error and ignore the file
    except Exception as e:
        # Print Error
        print(f"\t Error when loading file {f}", flush=True)
        print(f"\t {e}", flush=True)
        return np.array([]), np.array([])

    # Crop training samples
    if cfg.SAMPLE_CROP_MODE == "center":
        sig_splits = [audio.cropCenter(sig, rate, cfg.SIG_LENGTH)]
    elif cfg.SAMPLE_CROP_MODE == "first":
        sig_splits = [audio.splitSignal(sig, rate, cfg.SIG_LENGTH, cfg.SIG_OVERLAP, cfg.SIG_MINLEN)[0]]
    else:
        sig_splits = audio.splitSignal(sig, rate, cfg.SIG_LENGTH, cfg.SIG_OVERLAP, cfg.SIG_MINLEN)

    # Get feature embeddings
    batch_size = 1  # turns out that batch size 1 is the fastest, probably because of having to resize the model input when the number of samples in a batch changes
    for i in range(0, len(sig_splits), batch_size):
        batch_sig = sig_splits[i : i + batch_size]
        batch_label = [label_vector] * len(batch_sig)
        embeddings = model.embeddings(batch_sig)

        # Add to training data
        x_train.extend(embeddings)
        y_train.extend(batch_label)

    return x_train, y_train


def _loadTrainingData(cache_mode="none", cache_file="", progress_callback=None):
    """Loads the data for training.

    Reads all subdirectories of "config.TRAIN_DATA_PATH" and uses their names as new labels.

    These directories should contain all the training data for each label.

    If a cache file is provided, the training data is loaded from there.

    Args:
        cache_mode: Cache mode. Can be 'none', 'load' or 'save'. Defaults to 'none'.
        cache_file: Path to cache file.

    Returns:
        A tuple of (x_train, y_train, labels).
    """
    # Load from cache
    if cache_mode == "load":
        if os.path.isfile(cache_file):
            print(f"\t...loading from cache: {cache_file}", flush=True)
            x_train, y_train, labels, cfg.BINARY_CLASSIFICATION, cfg.MULTI_LABEL = utils.loadFromCache(cache_file)
            return x_train, y_train, labels
        else:
            print(f"\t...cache file not found: {cache_file}", flush=True)

    # Get list of subfolders as labels
    folders = list(sorted(utils.list_subdirectories(cfg.TRAIN_DATA_PATH)))

    # Read all individual labels from the folder names
    labels = []

    for folder in folders:
        labels_in_folder = folder.split(",")
        for label in labels_in_folder:
            if label not in labels:
                labels.append(label)

    # Sort labels
    labels = list(sorted(labels))

    # Get valid labels
    valid_labels = [l for l in labels if l.lower() not in cfg.NON_EVENT_CLASSES and not l.startswith("-")]

    # Check if binary classification
    cfg.BINARY_CLASSIFICATION = len(valid_labels) == 1

    # Validate the classes for binary classification
    if cfg.BINARY_CLASSIFICATION:
        if len([l for l in folders if l.startswith("-")]) > 0:
            raise Exception(
                "Negative labels can't be used with binary classification",
                "validation-no-negative-samples-in-binary-classification",
            )
        if len([l for l in folders if l.lower() in cfg.NON_EVENT_CLASSES]) == 0:
            raise Exception(
                "Non-event samples are required for binary classification",
                "validation-non-event-samples-required-in-binary-classification",
            )

    # Check if multi label
    cfg.MULTI_LABEL = len(valid_labels) > 1 and any("," in f for f in folders)

    # Check if multi-label and binary classficication
    if cfg.BINARY_CLASSIFICATION and cfg.MULTI_LABEL:
        raise Exception("Error: Binary classfication and multi-label not possible at the same time")

    # Only allow repeat upsampling for multi-label setting
    if cfg.MULTI_LABEL and cfg.UPSAMPLING_RATIO > 0 and cfg.UPSAMPLING_MODE != "repeat":
        raise Exception(
            "Only repeat-upsampling ist available for multi-label", "validation-only-repeat-upsampling-for-multi-label"
        )

    # Load training data
    x_train = []
    y_train = []

    for folder in folders:
        # Get label vector
        label_vector = np.zeros((len(valid_labels),), dtype="float32")
        folder_labels = folder.split(",")

        for label in folder_labels:
            if label.lower() not in cfg.NON_EVENT_CLASSES and not label.startswith("-"):
                label_vector[valid_labels.index(label)] = 1
            elif (
                label.startswith("-") and label[1:] in valid_labels
            ):  # Negative labels need to be contained in the valid labels
                label_vector[valid_labels.index(label[1:])] = -1

        # Get list of files
        # Filter files that start with '.' because macOS seems to them for temp files.
        files = filter(
            os.path.isfile,
            (
                os.path.join(cfg.TRAIN_DATA_PATH, folder, f)
                for f in sorted(os.listdir(os.path.join(cfg.TRAIN_DATA_PATH, folder)))
                if not f.startswith(".") and f.rsplit(".", 1)[-1].lower() in cfg.ALLOWED_FILETYPES
            ),
        )

        # Load files using thread pool
        with Pool(cfg.CPU_THREADS) as p:
            tasks = []

            for f in files:
                task = p.apply_async(partial(_loadAudioFile, f=f, label_vector=label_vector, config=cfg.getConfig()))
                tasks.append(task)

            # Wait for tasks to complete and monitor progress with tqdm
            num_files_processed = 0

            with tqdm.tqdm(total=len(tasks), desc=f" - loading '{folder}'", unit="f") as progress_bar:
                for task in tasks:
                    result = task.get()
                    # Make sure result is not empty
                    # Empty results might be caused by errors when loading the audio file
                    # TODO: We should check for embeddings size in result, otherwise we can't add them to the training data
                    if len(result[0]) > 0:
                        x_train += result[0]
                        y_train += result[1]

                    num_files_processed += 1
                    progress_bar.update(1)

                    if progress_callback:
                        progress_callback(num_files_processed, len(tasks), folder)

    # Convert to numpy arrays
    x_train = np.array(x_train, dtype="float32")
    y_train = np.array(y_train, dtype="float32")

    # Save to cache?
    if cache_mode == "save":
        print(f"\t...saving training data to cache: {cache_file}", flush=True)
        try:
            # Only save the valid labels
            utils.saveToCache(cache_file, x_train, y_train, valid_labels)
        except Exception as e:
            print(f"\t...error saving cache: {e}", flush=True)

    # Return only the valid labels for further use
    return x_train, y_train, valid_labels


def trainModel(on_epoch_end=None, on_trial_result=None, on_data_load_end=None, autotune_directory="autotune"):
    """Trains a custom classifier.

    Args:
        on_epoch_end: A callback function that takes two arguments `epoch`, `logs`.

    Returns:
        A keras `History` object, whose `history` property contains all the metrics.
    """

    # Load training data
    print("Loading training data...", flush=True)
    x_train, y_train, labels = _loadTrainingData(cfg.TRAIN_CACHE_MODE, cfg.TRAIN_CACHE_FILE, on_data_load_end)
    print(f"...Done. Loaded {x_train.shape[0]} training samples and {y_train.shape[1]} labels.", flush=True)

    if cfg.AUTOTUNE:
        import gc

        import keras
        import keras_tuner

        # Call callback to initialize progress bar
        if on_trial_result:
            on_trial_result(0)

        class BirdNetTuner(keras_tuner.BayesianOptimization):
            def __init__(self, x_train, y_train, max_trials, executions_per_trial, on_trial_result):
                super().__init__(
                    max_trials=max_trials,
                    executions_per_trial=executions_per_trial,
                    overwrite=True,
                    directory=autotune_directory,
                    project_name="birdnet_analyzer",
                )
                self.x_train = x_train
                self.y_train = y_train
                self.on_trial_result = on_trial_result

            def run_trial(self, trial, *args, **kwargs):
                histories = []
                hp: keras_tuner.HyperParameters = trial.hyperparameters
                trial_number = len(self.oracle.trials)

                for execution in range(int(self.executions_per_trial)):
                    print(f"Running Trial #{trial_number} execution #{execution + 1}", flush=True)

                    # Build model
                    print("Building model...", flush=True)
                    classifier = model.buildLinearClassifier(
                        self.y_train.shape[1],
                        self.x_train.shape[1],
                        hidden_units=hp.Choice(
                            "hidden_units", [0, 128, 256, 512, 1024, 2048], default=cfg.TRAIN_HIDDEN_UNITS
                        ),
                        dropout=hp.Choice("dropout", [0.0, 0.25, 0.33, 0.5, 0.75, 0.9], default=cfg.TRAIN_DROPOUT),
                    )
                    print("...Done.", flush=True)

                    # Only allow repeat upsampling in multi-label setting
                    upsampling_choices = ["repeat", "mean", "linear"]  # SMOTE is too slow

                    if cfg.MULTI_LABEL:
                        upsampling_choices = ["repeat"]

                    # Train model
                    print("Training model...", flush=True)
                    classifier, history = model.trainLinearClassifier(
                        classifier,
                        self.x_train,
                        self.y_train,
                        epochs=cfg.TRAIN_EPOCHS,
                        batch_size=hp.Choice("batch_size", [8, 16, 32, 64, 128], default=cfg.TRAIN_BATCH_SIZE),
                        learning_rate=hp.Choice(
                            "learning_rate",
                            [0.1, 0.01, 0.005, 0.002, 0.001, 0.0005, 0.0002, 0.0001],
                            default=cfg.TRAIN_LEARNING_RATE,
                        ),
                        val_split=cfg.TRAIN_VAL_SPLIT,
                        upsampling_ratio=hp.Choice(
                            "upsampling_ratio", [0.0, 0.25, 0.33, 0.5, 0.75, 1.0], default=cfg.UPSAMPLING_RATIO
                        ),
                        upsampling_mode=hp.Choice("upsampling_mode", upsampling_choices, default=cfg.UPSAMPLING_MODE),
                        train_with_mixup=hp.Boolean("mixup", default=cfg.TRAIN_WITH_MIXUP),
                        train_with_label_smoothing=hp.Boolean(
                            "label_smoothing", default=cfg.TRAIN_WITH_LABEL_SMOOTHING
                        ),
                    )

                    # Get the best validation loss
                    # Is it maybe better to return the negative val_auprc??
                    best_val_loss = history.history["val_loss"][np.argmin(history.history["val_loss"])]
                    histories.append(best_val_loss)

                    print(
                        f"Finished Trial #{trial_number} execution #{execution + 1}. best validation loss: {best_val_loss}",
                        flush=True,
                    )

                keras.backend.clear_session()
                del classifier
                del history
                gc.collect()

                # Call the on_trial_result callback
                if self.on_trial_result:
                    self.on_trial_result(trial_number)

                return histories

        tuner = BirdNetTuner(
            x_train=x_train,
            y_train=y_train,
            max_trials=cfg.AUTOTUNE_TRIALS,
            executions_per_trial=cfg.AUTOTUNE_EXECUTIONS_PER_TRIAL,
            on_trial_result=on_trial_result,
        )
        try:
            tuner.search()
        except utils.EmptyClassException as e:
            e.message = f"Class with label {labels[e.index]} is empty. Please remove it from the training data."
            e.args = (e.message,)
            utils.writeErrorLog(e)
            raise e

        best_params = tuner.get_best_hyperparameters()[0]
        print("Best params: ")
        print("hidden_units: ", best_params["hidden_units"])
        print("dropout: ", best_params["dropout"])
        print("batch_size: ", best_params["batch_size"])
        print("learning_rate: ", best_params["learning_rate"])
        print("upsampling_mode: ", best_params["upsampling_mode"])
        print("upsampling_ratio: ", best_params["upsampling_ratio"])
        print("mixup: ", best_params["mixup"])
        print("label_smoothing: ", best_params["label_smoothing"])
        cfg.TRAIN_HIDDEN_UNITS = best_params["hidden_units"]
        cfg.TRAIN_DROPOUT = best_params["dropout"]
        cfg.TRAIN_BATCH_SIZE = best_params["batch_size"]
        cfg.TRAIN_LEARNING_RATE = best_params["learning_rate"]
        cfg.UPSAMPLING_MODE = best_params["upsampling_mode"]
        cfg.UPSAMPLING_RATIO = best_params["upsampling_ratio"]
        cfg.TRAIN_WITH_MIXUP = best_params["mixup"]
        cfg.TRAIN_WITH_LABEL_SMOOTHING = best_params["label_smoothing"]

    # Build model
    print("Building model...", flush=True)
    classifier = model.buildLinearClassifier(
        y_train.shape[1], x_train.shape[1], cfg.TRAIN_HIDDEN_UNITS, cfg.TRAIN_DROPOUT
    )
    print("...Done.", flush=True)

    # Train model
    print("Training model...", flush=True)
    try:
        classifier, history = model.trainLinearClassifier(
            classifier,
            x_train,
            y_train,
            epochs=cfg.TRAIN_EPOCHS,
            batch_size=cfg.TRAIN_BATCH_SIZE,
            learning_rate=cfg.TRAIN_LEARNING_RATE,
            val_split=cfg.TRAIN_VAL_SPLIT,
            upsampling_ratio=cfg.UPSAMPLING_RATIO,
            upsampling_mode=cfg.UPSAMPLING_MODE,
            train_with_mixup=cfg.TRAIN_WITH_MIXUP,
            train_with_label_smoothing=cfg.TRAIN_WITH_LABEL_SMOOTHING,
            on_epoch_end=on_epoch_end,
        )
    except utils.EmptyClassException as e:
        e.message = f"Class with label {labels[e.index]} is empty. Please remove it from the training data."
        e.args = (e.message,)
        utils.writeErrorLog(e)
        raise e
    except Exception as e:
        utils.writeErrorLog(e)
        raise Exception("Error training model")

    print("...Done.", flush=True)

    # Best validation AUPRC (at minimum validation loss)
    best_val_auprc = history.history["val_AUPRC"][np.argmin(history.history["val_loss"])]
    best_val_auroc = history.history["val_AUROC"][np.argmin(history.history["val_loss"])]

    print("Saving model...", flush=True)

    try:
        if cfg.TRAINED_MODEL_OUTPUT_FORMAT == "both":
            model.save_raven_model(classifier, cfg.CUSTOM_CLASSIFIER, labels)
            model.saveLinearClassifier(classifier, cfg.CUSTOM_CLASSIFIER, labels, mode=cfg.TRAINED_MODEL_SAVE_MODE)
        elif cfg.TRAINED_MODEL_OUTPUT_FORMAT == "tflite":
            model.saveLinearClassifier(classifier, cfg.CUSTOM_CLASSIFIER, labels, mode=cfg.TRAINED_MODEL_SAVE_MODE)
        elif cfg.TRAINED_MODEL_OUTPUT_FORMAT == "raven":
            model.save_raven_model(classifier, cfg.CUSTOM_CLASSIFIER, labels)
        else:
            raise ValueError(f"Unknown model output format: {cfg.TRAINED_MODEL_OUTPUT_FORMAT}")
    except Exception as e:
        utils.writeErrorLog(e)
        raise Exception("Error saving model")

    save_sample_counts(labels, y_train)

    print(f"...Done. Best AUPRC: {best_val_auprc}, Best AUROC: {best_val_auroc}", flush=True)

    return history


if __name__ == "__main__":
    # Parse arguments
    parser = argparse.ArgumentParser(description="Train a custom classifier with BirdNET")
    parser.add_argument(
        "--i",
        default=os.path.join(SCRIPT_DIR, "train_data/"),
        help="Path to training data folder. Subfolder names are used as labels.",
    )
    parser.add_argument(
        "--crop_mode",
        default="center",
        help="Crop mode for training data. Can be 'center', 'first' or 'segments'. Defaults to 'center'.",
    )
    parser.add_argument(
        "--crop_overlap",
        type=float,
        default=0.0,
        help="Overlap of training data segments in seconds if crop_mode is 'segments'. Defaults to 0.",
    )
    parser.add_argument(
        "--o", default="checkpoints/custom/Custom_Classifier", help="Path to trained classifier model output."
    )
    parser.add_argument("--epochs", type=int, default=50, help="Number of training epochs. Defaults to 50.")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size. Defaults to 32.")
    parser.add_argument("--val_split", type=float, default=0.2, help="Validation split ratio. Defaults to 0.2.")
    parser.add_argument("--learning_rate", type=float, default=0.001, help="Learning rate. Defaults to 0.001.")
    parser.add_argument(
        "--hidden_units",
        type=int,
        default=0,
        help="Number of hidden units. Defaults to 0. If set to >0, a two-layer classifier is used.",
    )
    parser.add_argument("--dropout", type=float, default=0.0, help="Dropout rate. Defaults to 0.")
    parser.add_argument("--mixup", action=argparse.BooleanOptionalAction, help="Whether to use mixup for training.")
    parser.add_argument(
        "--upsampling_ratio",
        type=float,
        default=0.0,
        help="Balance train data and upsample minority classes. Values between 0 and 1. Defaults to 0.",
    )
    parser.add_argument(
        "--upsampling_mode",
        default="repeat",
        help="Upsampling mode. Can be 'repeat', 'mean' or 'smote'. Defaults to 'repeat'.",
    )
    parser.add_argument(
        "--model_format",
        default="tflite",
        help="Model output format. Can be 'tflite', 'raven' or 'both'. Defaults to 'tflite'.",
    )
    parser.add_argument(
        "--model_save_mode",
        default="replace",
        help="Model save mode. Can be 'replace' or 'append', where 'replace' will overwrite the original classification layer and 'append' will combine the original classification layer with the new one. Defaults to 'replace'.",
    )
    parser.add_argument(
        "--cache_mode", default="none", help="Cache mode. Can be 'none', 'load' or 'save'. Defaults to 'none'."
    )
    parser.add_argument(
        "--cache_file", default="train_cache.npz", help="Path to cache file. Defaults to 'train_cache.npz'."
    )
    parser.add_argument(
        "--threads", type=int, default=min(8, max(1, multiprocessing.cpu_count() // 2)), help="Number of CPU threads."
    )

    parser.add_argument(
        "--fmin",
        type=int,
        default=cfg.SIG_FMIN,
        help="Minimum frequency for bandpass filter in Hz. Defaults to {} Hz.".format(cfg.SIG_FMIN),
    )
    parser.add_argument(
        "--fmax",
        type=int,
        default=cfg.SIG_FMAX,
        help="Maximum frequency for bandpass filter in Hz. Defaults to {} Hz.".format(cfg.SIG_FMAX),
    )

    parser.add_argument(
        "--autotune",
        action=argparse.BooleanOptionalAction,
        help="Whether to use automatic hyperparameter tuning (this will execute multiple training runs to search for optimal hyperparameters).",
    )
    parser.add_argument(
        "--autotune_trials",
        type=int,
        default=50,
        help="Number of training runs for hyperparameter tuning. Defaults to 50.",
    )
    parser.add_argument(
        "--autotune_executions_per_trial",
        type=int,
        default=1,
        help="The number of times a training run with a set of hyperparameters is repeated during hyperparameter tuning (this reduces the variance). Defaults to 1.",
    )

    args = parser.parse_args()

    # Config
    cfg.TRAIN_DATA_PATH = args.i
    cfg.SAMPLE_CROP_MODE = args.crop_mode
    cfg.SIG_OVERLAP = max(0.0, min(2.9, float(args.crop_overlap)))
    cfg.CUSTOM_CLASSIFIER = args.o
    cfg.TRAIN_EPOCHS = args.epochs
    cfg.TRAIN_BATCH_SIZE = args.batch_size
    cfg.TRAIN_VAL_SPLIT = args.val_split
    cfg.TRAIN_LEARNING_RATE = args.learning_rate
    cfg.TRAIN_HIDDEN_UNITS = args.hidden_units
    cfg.TRAIN_DROPOUT = min(max(0, args.dropout), 0.9)
    cfg.TRAIN_WITH_MIXUP = args.mixup if args.mixup is not None else cfg.TRAIN_WITH_MIXUP
    cfg.UPSAMPLING_RATIO = min(max(0, args.upsampling_ratio), 1)
    cfg.UPSAMPLING_MODE = args.upsampling_mode
    cfg.TRAINED_MODEL_OUTPUT_FORMAT = args.model_format
    cfg.TRAINED_MODEL_SAVE_MODE = args.model_save_mode
    cfg.TRAIN_CACHE_MODE = args.cache_mode.lower()
    cfg.TRAIN_CACHE_FILE = args.cache_file
    cfg.TFLITE_THREADS = 1
    cfg.CPU_THREADS = max(1, int(args.threads))

    cfg.BANDPASS_FMIN = max(0, min(cfg.SIG_FMAX, int(args.fmin)))
    cfg.BANDPASS_FMAX = max(cfg.SIG_FMIN, min(cfg.SIG_FMAX, int(args.fmax)))

    cfg.AUTOTUNE = args.autotune
    cfg.AUTOTUNE_TRIALS = args.autotune_trials
    cfg.AUTOTUNE_EXECUTIONS_PER_TRIAL = args.autotune_executions_per_trial

    # Train model
    trainModel()
