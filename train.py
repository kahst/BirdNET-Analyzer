"""Module for training a custom classifier.

Can be used to train a custom classifier with new training data.
"""
import argparse
import os

import numpy as np

import audio
import config as cfg
import model
import utils


def _loadTrainingData(cache_mode="none", cache_file=""):
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
            x_train, y_train, labels = utils.loadFromCache(cache_file)
            return x_train, y_train, labels
        else:
            print(f"\t...cache file not found: {cache_file}", flush=True)

    # Get list of subfolders as labels
    labels = list(sorted(utils.list_subdirectories(cfg.TRAIN_DATA_PATH)))

    # Get valid labels
    valid_labels = [l for l in labels if not l.lower() in cfg.NON_EVENT_CLASSES]

    # Load training data
    x_train = []
    y_train = []

    for label in labels:

        # Current label
        print(f"\t- {label}", flush=True)

        # Get label vector
        label_vector = np.zeros((len(valid_labels),), dtype="float32")
        if not label.lower() in cfg.NON_EVENT_CLASSES and not label.startswith("-"):
            label_vector[valid_labels.index(label)] = 1

        # Get list of files
        # Filter files that start with '.' because macOS seems to them for temp files.
        files = filter(
            os.path.isfile,
            (
                os.path.join(cfg.TRAIN_DATA_PATH, label, f)
                for f in sorted(os.listdir(os.path.join(cfg.TRAIN_DATA_PATH, label)))
                if not f.startswith(".") and f.rsplit(".", 1)[-1].lower() in cfg.ALLOWED_FILETYPES
            ),
        )

        # Load files
        for f in files:
            # Load audio
            sig, rate = audio.openAudioFile(f, duration=cfg.SIG_LENGTH if cfg.SAMPLE_CROP_MODE == "first" else None)
            
            # Crop training samples
            if cfg.SAMPLE_CROP_MODE == "center":
                sig_splits = [audio.cropCenter(sig, rate, cfg.SIG_LENGTH)]
            elif cfg.SAMPLE_CROP_MODE == "first":
                sig_splits = [audio.splitSignal(sig, rate, cfg.SIG_LENGTH, cfg.SIG_OVERLAP, cfg.SIG_MINLEN)[0]]
            else:
                sig_splits = audio.splitSignal(sig, rate, cfg.SIG_LENGTH, cfg.SIG_OVERLAP, cfg.SIG_MINLEN)

            # Get feature embeddings
            for sig in sig_splits:
                embeddings = model.embeddings([sig])[0]

                # Add to training data
                x_train.append(embeddings)
                y_train.append(label_vector)

    # Convert to numpy arrays
    x_train = np.array(x_train, dtype="float32")
    y_train = np.array(y_train, dtype="float32")

    # Remove non-event classes from labels
    labels = [l for l in labels if not l.lower() in cfg.NON_EVENT_CLASSES]

    # Save to cache?
    if cache_mode == "save":
        print(f"\t...saving training data to cache: {cache_file}", flush=True)
        try:
            utils.saveToCache(cache_file, x_train, y_train, labels)
        except Exception as e:
            print(f"\t...error saving cache: {e}", flush=True)

    return x_train, y_train, labels


def trainModel(on_epoch_end=None):
    """Trains a custom classifier.

    Args:
        on_epoch_end: A callback function that takes two arguments `epoch`, `logs`.

    Returns:
        A keras `History` object, whose `history` property contains all the metrics.
    """
    # Load training data
    print("Loading training data...", flush=True)
    x_train, y_train, labels = _loadTrainingData(cfg.TRAIN_CACHE_MODE, cfg.TRAIN_CACHE_FILE)
    print(f"...Done. Loaded {x_train.shape[0]} training samples and {y_train.shape[1]} labels.", flush=True)

    # Build model
    print("Building model...", flush=True)
    classifier = model.buildLinearClassifier(y_train.shape[1], x_train.shape[1], cfg.TRAIN_HIDDEN_UNITS, cfg.TRAIN_DROPOUT)
    print("...Done.", flush=True)
    
    # Train model
    print("Training model...", flush=True)
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

    # Best validation AUPRC (at minimum validation loss)
    best_val_auprc = history.history["val_AUPRC"][np.argmin(history.history["val_loss"])]

    if cfg.TRAINED_MODEL_OUTPUT_FORMAT == "both":
        model.save_raven_model(classifier, cfg.CUSTOM_CLASSIFIER, labels)
        model.saveLinearClassifier(classifier, cfg.CUSTOM_CLASSIFIER, labels)
    elif cfg.TRAINED_MODEL_OUTPUT_FORMAT == "tflite":
        model.saveLinearClassifier(classifier, cfg.CUSTOM_CLASSIFIER, labels)
    elif cfg.TRAINED_MODEL_OUTPUT_FORMAT == "raven":
        model.save_raven_model(classifier, cfg.CUSTOM_CLASSIFIER, labels)
    else:
        raise ValueError(f"Unknown model output format: {cfg.TRAINED_MODEL_OUTPUT_FORMAT}")

    print(f"...Done. Best AUPRC: {best_val_auprc}", flush=True)

    return history


if __name__ == "__main__":
    # Parse arguments
    parser = argparse.ArgumentParser(description="Train a custom classifier with BirdNET")
    parser.add_argument("--i", default="train_data/", help="Path to training data folder. Subfolder names are used as labels.")
    parser.add_argument("--crop_mode", default="center", help="Crop mode for training data. Can be 'center', 'first' or 'segments'. Defaults to 'center'.")
    parser.add_argument("--crop_overlap", type=float, default=0.0, help="Overlap of training data segments in seconds if crop_mode is 'segments'. Defaults to 0.")
    parser.add_argument(
        "--o", default="checkpoints/custom/Custom_Classifier", help="Path to trained classifier model output."
    )
    parser.add_argument("--epochs", type=int, default=100, help="Number of training epochs. Defaults to 100.")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size. Defaults to 32.")
    parser.add_argument("--val_split", type=float, default=0.2, help="Validation split ratio. Defaults to 0.2.")
    parser.add_argument("--learning_rate", type=float, default=0.01, help="Learning rate. Defaults to 0.01.")
    parser.add_argument(
        "--hidden_units",
        type=int,
        default=0,
        help="Number of hidden units. Defaults to 0. If set to >0, a two-layer classifier is used.",
    )
    parser.add_argument("--dropout", type=float, default=0.0, help="Dropout rate. Defaults to 0.")
    parser.add_argument("--mixup", action=argparse.BooleanOptionalAction, help="Whether to use mixup for training.")
    parser.add_argument("--upsampling_ratio", type=float, default=0.0, help="Balance train data and upsample minority classes. Values between 0 and 1. Defaults to 0.")
    parser.add_argument("--upsampling_mode", default="repeat", help="Upsampling mode. Can be 'repeat', 'mean' or 'smote'. Defaults to 'repeat'.")
    parser.add_argument("--model_format", default="tflite", help="Model output format. Can be 'tflite', 'raven' or 'both'. Defaults to 'tflite'.")
    parser.add_argument("--cache_mode", default="none", help="Cache mode. Can be 'none', 'load' or 'save'. Defaults to 'none'.")
    parser.add_argument("--cache_file", default="train_cache.npz", help="Path to cache file. Defaults to 'train_cache.npz'.")

    args = parser.parse_args()

    # Config
    cfg.TRAIN_DATA_PATH = args.i
    cfg.SAMPLE_CROP_MODE = args.crop_mode
    cfg.SIG_OVERLAP = args.crop_overlap
    cfg.CUSTOM_CLASSIFIER = args.o
    cfg.TRAIN_EPOCHS = args.epochs
    cfg.TRAIN_BATCH_SIZE = args.batch_size
    cfg.TRAIN_VAL_SPLIT = args.val_split
    cfg.TRAIN_LEARNING_RATE = args.learning_rate
    cfg.TRAIN_HIDDEN_UNITS = args.hidden_units
    cfg.TRAIN_DROPOUT = min(max(0, args.dropout), 0.9)
    cfg.TRAIN_WITH_MIXUP = args.mixup
    cfg.UPSAMPLING_RATIO = min(max(0, args.upsampling_ratio), 1)
    cfg.UPSAMPLING_MODE = args.upsampling_mode
    cfg.TRAINED_MODEL_OUTPUT_FORMAT = args.model_format
    cfg.TRAIN_CACHE_MODE = args.cache_mode.lower()
    cfg.TRAIN_CACHE_FILE = args.cache_file
    cfg.TFLITE_THREADS = 4 # Set this to 4 to speed things up a bit

    # Train model
    trainModel()
