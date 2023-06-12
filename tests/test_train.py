"""Module for training a custom classifier.

Can be used to train a custom classifier with new training data.
"""
from collections import namedtuple

from birdnet.configuration import config
from birdnet.train.train_model import train_model
from tests._paths import ROOT_PATH


def test_train():
    Arguments = namedtuple(
        'Arguments', "i o epochs batch_size learning_rate hidden_units"
    )
    arguments = Arguments(
        i='BirdNET-Training-Data-main/',
        o=str(ROOT_PATH / 'checkpoints/custom/Custom_Classifier.tflite'),
        epochs=1,
        batch_size=1,
        learning_rate=0.01,
        hidden_units=0,
    )

    # Configure
    config.TRAIN_DATA_PATH = str(ROOT_PATH / arguments.i)
    config.CUSTOM_CLASSIFIER = arguments.o
    config.TRAIN_EPOCHS = arguments.epochs
    config.TRAIN_BATCH_SIZE = arguments.batch_size
    config.TRAIN_LEARNING_RATE = arguments.learning_rate
    config.TRAIN_HIDDEN_UNITS = arguments.hidden_units

    # Train model
    train_model()
