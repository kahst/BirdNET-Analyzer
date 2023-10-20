"""Module containing common function.
"""
import os
import traceback
import numpy as np
from pathlib import Path

import config as cfg


def collect_audio_files(path: str):
    """Collects all audio files in the given directory.

    Args:
        path: The directory to be searched.

    Returns:
        A sorted list of all audio files in the directory.
    """
    # Get all files in directory with os.walk
    files = []

    for root, _, flist in os.walk(path):
        for f in flist:
            if not f.startswith(".") and f.rsplit(".", 1)[-1].lower() in cfg.ALLOWED_FILETYPES:
                files.append(os.path.join(root, f))

    return sorted(files)


def readLines(path: str):
    """Reads the lines into a list.

    Opens the file and reads its contents into a list.
    It is expected to have one line for each species or label.

    Args:
        path: Absolute path to the species file.

    Returns:
        A list of all species inside the file.
    """
    return Path(path).read_text(encoding="utf-8").splitlines() if path else []


def list_subdirectories(path: str):
    """Lists all directories inside a path.

    Retrieves all the subdirectories in a given path without recursion.

    Args:
        path: Directory to be searched.

    Returns:
        A filter sequence containing the absolute paths to all directories.
    """
    return filter(lambda el: os.path.isdir(os.path.join(path, el)), os.listdir(path))

def mixup(x, y, alpha=0.3):
    """Apply mixup to the given data.

    Shuffle the data and add samples with random weights.

    Args:
        x: Samples.
        y: One-hot labels.

    Returns:
        Mixed data.
    """

    # Set numpy random seed
    np.random.seed(cfg.RANDOM_SEED)

    # Shuffle data
    indices = np.arange(len(x))
    np.random.shuffle(indices)

    # Random weights
    weight = np.random.beta(alpha, alpha, len(x))

    # Apply weights (to samples only)
    x1 = x[indices] * weight.reshape(-1, 1)
    x2 = x * (1 - weight.reshape(-1, 1))

    # Add weighted samples
    x = x1 + x2

    # Combine labels
    y = np.clip(y[indices] + y, y.min(), 1)

    return x, y

def label_smoothing(y, alpha=0.1):

    # Subtract alpha from correct label when it is >0
    y[y > 0] -= alpha

    # Assigned alpha to all other labels
    y[y == 0] = alpha / y.shape[0]

    return y

def upsampling(x, y, ratio=0.5, mode="repeat"):
    """Balance data through upsampling.

    We upsample minority classes to have at least 10% (ratio=0.1) of the samples of the majority class.

    Args:
        x: Samples.
        y: One-hot labels.
        ratio: The minimum ratio of minority to majority samples.
        mode: The upsampling mode. Either 'repeat', 'mean' or 'smote'.

    Returns:
        Upsampled data.
    """

    # Set numpy random seed
    np.random.seed(cfg.RANDOM_SEED)

    # Determin min number of samples
    min_samples = int(np.max(y.sum(axis=0)) * ratio)

    x_temp = []
    y_temp = []
    if mode == 'repeat':

        # For each class with less than min_samples ranomdly repeat samples
        for i in range(y.shape[1]):

            while y[:, i].sum() + len(y_temp) < min_samples:

                # Randomly choose a sample from the minority class
                random_index = np.random.choice(np.where(y[:, i] == 1)[0])

                # Append the sample and label to a temp list
                x_temp.append(x[random_index])
                y_temp.append(y[random_index])

    elif mode == 'mean':

        # For each class with less than min_samples
        # select two random samples and calculate the mean
        for i in range(y.shape[1]):

            x_temp = []
            y_temp = []
            while y[:, i].sum() + len(y_temp) < min_samples:

                # Randomly choose two samples from the minority class
                random_indices = np.random.choice(np.where(y[:, i] == 1)[0], 2)

                # Calculate the mean of the two samples
                mean = np.mean(x[random_indices], axis=0)

                # Append the mean and label to a temp list
                x_temp.append(mean)
                y_temp.append(y[random_indices[0]])

    elif mode == 'smote':

        # For each class with less than min_samples apply SMOTE
        for i in range(y.shape[1]):

            x_temp = []
            y_temp = []
            while y[:, i].sum() + len(y_temp) < min_samples:

                # Randomly choose a sample from the minority class
                random_index = np.random.choice(np.where(y[:, i] == 1)[0])

                # Get the k nearest neighbors
                k = 5
                distances = np.sqrt(np.sum((x - x[random_index])**2, axis=1))
                indices = np.argsort(distances)[1:k+1]

                # Randomly choose one of the neighbors
                random_neighbor = np.random.choice(indices)

                # Calculate the difference vector
                diff = x[random_neighbor] - x[random_index]

                # Randomly choose a weight between 0 and 1
                weight = np.random.uniform(0, 1)

                # Calculate the new sample
                new_sample = x[random_index] + weight * diff

                # Append the new sample and label to a temp list
                x_temp.append(new_sample)
                y_temp.append(y[random_index])

    # Append the temp list to the original data
    if len(x_temp) > 0:
        x = np.vstack((x, np.array(x_temp)))
        y = np.vstack((y, np.array(y_temp)))

    # Shuffle data
    indices = np.arange(len(x))
    np.random.shuffle(indices)
    x = x[indices]
    y = y[indices]        

    return x, y

def saveToCache(cache_file: str, x_train: np.ndarray, y_train: np.ndarray, labels: list[str]):
    """Saves the training data to a cache file.

    Args:
        cache_file: The path to the cache file.
        x_train: The training samples.
        y_train: The training labels.
        labels: The list of labels.
    """
    # Create cache directory
    os.makedirs(os.path.dirname(cache_file), exist_ok=True)

    # Save to cache
    np.savez_compressed(cache_file, x_train=x_train, y_train=y_train, labels=labels)

def loadFromCache(cache_file: str):
    """Loads the training data from a cache file.

    Args:
        cache_file: The path to the cache file.

    Returns:
        A tuple of (x_train, y_train, labels).

    """    
    # Load from cache
    cache = np.load(cache_file, allow_pickle=True)

    # Get data
    x_train = cache["x_train"]
    y_train = cache["y_train"]
    labels = cache["labels"]

    return x_train, y_train, labels

def clearErrorLog():
    """Clears the error log file.

    For debugging purposes.
    """
    if os.path.isfile(cfg.ERROR_LOG_FILE):
        os.remove(cfg.ERROR_LOG_FILE)


def writeErrorLog(ex: Exception):
    """Writes an exception to the error log.

    Formats the stacktrace and writes it in the error log file configured in the config.

    Args:
        ex: An exception that occurred.
    """
    with open(cfg.ERROR_LOG_FILE, "a") as elog:
        elog.write("".join(traceback.TracebackException.from_exception(ex).format()) + "\n")


