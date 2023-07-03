import numpy

from birdnet.model.main import build_linear_classifier
from birdnet.model.main import save_linear_classifier
from birdnet.model.main import train_linear_classifier
from birdnet.train._training_data_loading import _load_training_data
from birdnet.configuration import config


def train_model(on_epoch_end=None):
    """Trains a custom classifier.
    Args:
        on_epoch_end: A callback function that takes two arguments `epoch`, `logs`.
    Returns:
        A keras `History` object, whose `history` property contains all the metrics.
    """
    # Load training data
    print("Loading training data...", flush=True)
    x_train, y_train, labels = _load_training_data()
    print(
        f"...Done. Loaded "
        f"{x_train.shape[0]} training samples and "
        f"{y_train.shape[1]} labels.",
        flush=True,
    )

    # Build model
    print("Building model...", flush=True)
    classifier = build_linear_classifier(y_train.shape[1], x_train.shape[1], config.TRAIN_HIDDEN_UNITS)
    print("...Done.", flush=True)

    # Train model
    print("Training model...", flush=True)
    classifier, history = train_linear_classifier(
        classifier,
        x_train,
        y_train,
        epochs=config.TRAIN_EPOCHS,
        batch_size=config.TRAIN_BATCH_SIZE,
        learning_rate=config.TRAIN_LEARNING_RATE,
        on_epoch_end=on_epoch_end,
    )

    # Best validation precision (at minimum validation loss)
    best_val_prec = history.history["val_prec"][numpy.argmin(history.history["val_loss"])]

    save_linear_classifier(classifier, config.CUSTOM_CLASSIFIER, labels)
    print(f"...Done. Best top-1 precision: {best_val_prec}", flush=True)

    return history
