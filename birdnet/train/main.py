"""Module for training a custom classifier.

Can be used to train a custom classifier with new training data.
"""
import argparse


import config as cfg
from birdnet.train.train_model import train_model


if __name__ == "__main__":
    # Parse arguments
    parser = argparse.ArgumentParser(description="Analyze audio files with BirdNET")
    parser.add_argument("--i", default="train_data/", help="Path to training data folder. Subfolder names are used as labels.")
    parser.add_argument(
        "--o", default="checkpoints/custom/Custom_Classifier.tflite", help="Path to trained classifier model output."
    )
    parser.add_argument("--epochs", type=int, default=100, help="Number of training epochs. Defaults to 100.")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size. Defaults to 32.")
    parser.add_argument("--learning_rate", type=float, default=0.01, help="Learning rate. Defaults to 0.01.")
    parser.add_argument(
        "--hidden_units",
        type=int,
        default=0,
        help="Number of hidden units. Defaults to 0. If set to >0, a two-layer classifier is used.",
    )

    args = parser.parse_args()

    # Config
    cfg.TRAIN_DATA_PATH = args.i
    cfg.CUSTOM_CLASSIFIER = args.o
    cfg.TRAIN_EPOCHS = args.epochs
    cfg.TRAIN_BATCH_SIZE = args.batch_size
    cfg.TRAIN_LEARNING_RATE = args.learning_rate
    cfg.TRAIN_HIDDEN_UNITS = args.hidden_units

    # Train model
    train_model()
