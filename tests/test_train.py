"""Module for training a custom classifier.

Can be used to train a custom classifier with new training data.
"""
from collections import namedtuple

import config as cfg
from birdnet.train.train_model import train_model
from tests._paths import ROOT_PATH


def test_train():
    Arguments = namedtuple(
        'Arguments', "i o epochs batch_size learning_rate hidden_units"
    )
    arguments = Arguments(
        i='train_data/',
        o='checkpoints/custom/Custom_Classifier.tflite',
        epochs=100,
        batch_size=32,
        learning_rate=0.01,
        hidden_units=0,
    )

    # Config
    cfg.TRAIN_DATA_PATH = str(ROOT_PATH / arguments.i)
    cfg.CUSTOM_CLASSIFIER = arguments.o
    cfg.TRAIN_EPOCHS = arguments.epochs
    cfg.TRAIN_BATCH_SIZE = arguments.batch_size
    cfg.TRAIN_LEARNING_RATE = arguments.learning_rate
    cfg.TRAIN_HIDDEN_UNITS = arguments.hidden_units

    # Train model
    train_model()
