"""Module for training a custom classifier.

Can be used to train a custom classifier with new training data.
"""


import config as cfg
from birdnet.train.train_model import train_model


def test_train():
    args = {
        'i': 'train_data/',
        'o': 'checkpoints/custom/Custom_Classifier.tflite',
        'epochs': 100,
        'batch_size': 32,
        'learning_rate': 0.01,
        'hidden_units': 0,
    }

    # Config
    cfg.TRAIN_DATA_PATH = args.i
    cfg.CUSTOM_CLASSIFIER = args.o
    cfg.TRAIN_EPOCHS = args.epochs
    cfg.TRAIN_BATCH_SIZE = args.batch_size
    cfg.TRAIN_LEARNING_RATE = args.learning_rate
    cfg.TRAIN_HIDDEN_UNITS = args.hidden_units

    # Train model
    train_model()
