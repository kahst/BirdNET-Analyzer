import os
from collections import namedtuple
from multiprocessing import Pool

from birdnet.configuration import config
from birdnet.embeddings.file_analysing import analyze_file
from birdnet.utils.audio_file_collecting import collect_audio_files

from tests._paths import ROOT_PATH


def test_embeddings():
    Arguments = namedtuple(
        'Arguments', "i o overlap threads batchsize"
    )
    arguments: Arguments = Arguments(
        i=str(ROOT_PATH / 'example/'),
        o=str(ROOT_PATH / 'example/'),
        overlap=0.0,
        threads=1,
        batchsize=1,
    )

    ### Make sure to comment out appropriately if you are not using arguments. ###

    # Set input and output path
    config.INPUT_PATH = arguments.i
    config.OUTPUT_PATH = arguments.o

    # Parse input files
    if os.path.isdir(config.INPUT_PATH):
        config.FILE_LIST = collect_audio_files(config.INPUT_PATH)
    else:
        config.FILE_LIST = [config.INPUT_PATH]

    # Set overlap
    config.SIG_OVERLAP = max(0.0, min(2.9, float(arguments.overlap)))

    # Set number of threads
    if os.path.isdir(config.INPUT_PATH):
        config.CPU_THREADS = int(arguments.threads)
        config.TFLITE_THREADS = 1
    else:
        config.CPU_THREADS = 1
        config.TFLITE_THREADS = int(arguments.threads)

    # Set batch size
    config.BATCH_SIZE = max(1, int(arguments.batchsize))

    # Add config items to each file list entry.
    # We have to do this for Windows which does not
    # support fork() and thus each process has to
    # have its own config. USE LINUX!
    file_list = [
        (
            file_list_entry, config.get_config()
        )
        for file_list_entry in config.FILE_LIST
    ]

    # Analyze files
    if config.CPU_THREADS < 2:
        for file_list_entry in file_list:
            analyze_file(file_list_entry)
    else:
        with Pool(config.CPU_THREADS) as p:
            p.map(analyze_file, file_list)

    # A few examples to test
    # python3 embeddings.py --i example/ --o example/ --threads 4
    # python3 embeddings.py --i example/soundscape.wav --o example/soundscape.birdnet.embeddings.txt --threads 4
