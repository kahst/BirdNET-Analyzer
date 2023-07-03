import os

import numpy

import birdnet.audio.audio as audio
import birdnet.audio.audio_file_opening
import birdnet.audio.center_cropping
from birdnet.configuration import config
from birdnet.utils.subdirectories_listing import list_subdirectories
from birdnet.model.main import extract_embeddings


def _load_training_data():
    """Loads the data for training.
    Reads all subdirectories of "config.TRAIN_DATA_PATH" and uses their names
    as new labels. These directories should contain all the training data
    for each label.
    """
    # Get list of subfolders as labels
    labels = list(sorted(list_subdirectories(config.TRAIN_DATA_PATH)))

    # Load training data
    x_train = []
    y_train = []

    for i, label in enumerate(labels):
        # Get label vector
        label_vector = numpy.zeros((len(labels),), dtype="float32")
        if not label.lower() in ["noise", "other", "background", "silence"]:
            label_vector[i] = 1

        # Get list of files
        # Filter files that start with '.' because macOS seems to them for
        # temporary files.
        training_files = filter(
            os.path.isfile,
            (
                os.path.join(config.TRAIN_DATA_PATH, label, f)
                for f in sorted(
                    os.listdir(os.path.join(config.TRAIN_DATA_PATH, label))
                )
                if not f.startswith(".") and
                   f.rsplit(".", 1)[-1].lower()
                   in config.ALLOWED_FILETYPES
            ),
        )

        # Load files
        for training_file in training_files:
            # Load audio
            sig, rate = birdnet.audio.audio_file_opening.open_audio_file(training_file)

            # Crop center segment
            sig = birdnet.audio.center_cropping.crop_center(sig, rate, config.SIG_LENGTH)

            # Get feature embeddings
            embeddings = extract_embeddings([sig])[0]

            # Add to training data
            x_train.append(embeddings)
            y_train.append(label_vector)

    # Convert to numpy arrays
    x_train = numpy.array(x_train, dtype="float32")
    y_train = numpy.array(y_train, dtype="float32")

    return x_train, y_train, labels
