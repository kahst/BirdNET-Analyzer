"""Module used to extract embeddings for samples."""

import argparse
import datetime
import os
from multiprocessing import Pool

import numpy as np

import birdnet_analyzer.analyze as analyze
import birdnet_analyzer.audio as audio
import birdnet_analyzer.config as cfg
import birdnet_analyzer.model as model
import birdnet_analyzer.utils as utils

from perch_hoplite.db import sqlite_usearch_impl
from perch_hoplite.db import interface as hoplite
from ml_collections import ConfigDict
from functools import partial
from tqdm import tqdm

DATASET_NAME: str = "birdnet_analyzer_dataset"
SCRIPT_DIR = os.path.abspath(os.path.dirname(__file__))

def analyzeFile(item, db: sqlite_usearch_impl.SQLiteUsearchDB):
    """Extracts the embeddings for a file.

    Args:
        item: (filepath, config)
    """
    # Get file path and restore cfg
    fpath: str = item[0]
    cfg.setConfig(item[1])

    offset = 0
    duration = cfg.FILE_SPLITTING_DURATION
    fileLengthSeconds = int(audio.getAudioFileLength(fpath))

    # Start time
    start_time = datetime.datetime.now()

    # Status
    print(f"Analyzing {fpath}", flush=True)

    source_id = fpath

    # Process each chunk
    try:
        while offset < fileLengthSeconds:
            chunks = analyze.getRawAudioFromFile(fpath, offset, duration)
            start, end = offset, cfg.SIG_LENGTH + offset
            samples = []
            timestamps = []


            for c in range(len(chunks)):
                # Add to batch
                samples.append(chunks[c])
                timestamps.append([start, end])

                # Advance start and end
                start += cfg.SIG_LENGTH - cfg.SIG_OVERLAP
                end = start + cfg.SIG_LENGTH

                # Check if batch is full or last chunk
                if len(samples) < cfg.BATCH_SIZE and c < len(chunks) - 1:
                    continue

                # Prepare sample and pass through model
                data = np.array(samples, dtype="float32")
                e = model.embeddings(data)

                # Add to results
                for i in range(len(samples)):
                    # Get timestamp
                    s_start, s_end = timestamps[i]

                    s_start = s_start
                    s_end = s_end

                    # create the source in the database to 
                    db._get_source_id(DATASET_NAME, source_id, insert=True)

                    # Check if embedding already exists
                    existing_embedding = db.get_embeddings_by_source(DATASET_NAME, source_id, np.array([s_start, s_end]))
                    
                    if existing_embedding.size == 0:
                        # Get prediction
                        embeddings = e[i]

                        # Store embeddings
                        embeddings_source = hoplite.EmbeddingSource(DATASET_NAME, source_id, np.array([s_start, s_end]))

                        # Insert into database
                        db.insert_embedding(embeddings, embeddings_source)
                        db.commit()

                # Reset batch
                samples = []
                timestamps = []

            offset = offset + duration

    except Exception as ex:
        # Write error log
        print(f"Error: Cannot analyze audio file {fpath}.", flush=True)
        utils.writeErrorLog(ex)

        return

    delta_time = (datetime.datetime.now() - start_time).total_seconds()
    print("Finished {} in {:.2f} seconds".format(fpath, delta_time), flush=True)

def getDatabase(db_path: str):
    """Get the database object. Creates or opens the databse.
    Args:
        db: The path to the database.
    Returns:
        The database object.
    """

    if not os.path.exists(db_path):
        os.makedirs(os.path.dirname(db_path), exist_ok=True)
        db = sqlite_usearch_impl.SQLiteUsearchDB.create(
            db_path=db_path,
            usearch_cfg=sqlite_usearch_impl.get_default_usearch_config(embedding_dim=1024) #TODO dont hardcode this
        )
        return db
    return sqlite_usearch_impl.SQLiteUsearchDB.create(db_path=db_path)

def checkDatabaseSettings(db: sqlite_usearch_impl.SQLiteUsearchDB):
    try:
        settings = db.get_metadata('birdnet_analyzer_settings')
        if (settings["BANDPASS_FMIN"] != cfg.BANDPASS_FMIN or
            settings["BANDPASS_FMAX"] != cfg.BANDPASS_FMAX or
            settings["AUDIO_SPEED"] != cfg.AUDIO_SPEED):
            raise ValueError("Database settings do not match current configuration. DB Settings are: fmin: {}, fmax: {}, audio_speed: {}".format(settings["BANDPASS_FMIN"], settings["BANDPASS_FMAX"], settings["AUDIO_SPEED"]))
    except KeyError:
        settings = ConfigDict({
            "BANDPASS_FMIN": cfg.BANDPASS_FMIN,
            "BANDPASS_FMAX": cfg.BANDPASS_FMAX,
            "AUDIO_SPEED": cfg.AUDIO_SPEED
        }) 
        db.insert_metadata("birdnet_analyzer_settings", settings)
        db.commit()

def run(input, database_path, overlap, threads, batchsize, audio_speed, fmin, fmax):
    # Set paths relative to script path (requested in #3)
    cfg.MODEL_PATH = os.path.join(SCRIPT_DIR, cfg.MODEL_PATH)
    cfg.ERROR_LOG_FILE = os.path.join(SCRIPT_DIR, cfg.ERROR_LOG_FILE)

    # Set input and output path
    cfg.INPUT_PATH = input

    # Parse input files
    if os.path.isdir(cfg.INPUT_PATH):
        cfg.FILE_LIST = utils.collect_audio_files(cfg.INPUT_PATH)
    else:
        cfg.FILE_LIST = [cfg.INPUT_PATH]

    # Set overlap
    cfg.SIG_OVERLAP = max(0.0, min(2.9, float(overlap)))

    cfg.AUDIO_SPEED = max(0.01, audio_speed)

    # Set bandpass frequency range
    cfg.BANDPASS_FMIN = max(0, min(cfg.SIG_FMAX, int(fmin)))
    cfg.BANDPASS_FMAX = max(cfg.SIG_FMIN, min(cfg.SIG_FMAX, int(fmax)))

    # Set number of threads
    if os.path.isdir(cfg.INPUT_PATH):
        cfg.CPU_THREADS = max(1, int(threads))
        cfg.TFLITE_THREADS = 1
    else:
        cfg.CPU_THREADS = 1
        cfg.TFLITE_THREADS = max(1, int(threads))

    cfg.CPU_THREADS = 1 # TODO: with the current implementation, we can't use more than 1 thread

    # Set batch size
    cfg.BATCH_SIZE = max(1, int(batchsize))

    # Add config items to each file list entry.
    # We have to do this for Windows which does not
    # support fork() and thus each process has to
    # have its own config. USE LINUX!
    flist = [(f, cfg.getConfig()) for f in cfg.FILE_LIST]

    db = getDatabase(database_path)
    checkDatabaseSettings(db)

    # Analyze files
    if cfg.CPU_THREADS < 2:
        for entry in tqdm(flist):
            analyzeFile(entry, db)
    else:
        with Pool(cfg.CPU_THREADS) as p:
            tqdm(p.imap(partial(analyzeFile, db=db), flist))
    
    db.db.close() #TODO: needed to close db connection and avoid having wal/shm files


if __name__ == "__main__":
    # Parse arguments
    parser = argparse.ArgumentParser(description="Extract feature embeddings with BirdNET")
    parser.add_argument(
        "--i", default=os.path.join(SCRIPT_DIR, "example/"), help="Path to input file or folder."
    )
    parser.add_argument(
        "--db",
        default="example/hoplite-db/",
        help="Path to the Hoplite database. Defaults to example/hoplite-db/.",
    )
    parser.add_argument(
        "--overlap",
        type=float,
        default=0.0,
        help="Overlap of prediction segments. Values in [0.0, 2.9]. Defaults to 0.0.",
    )
    parser.add_argument("--threads", type=int, default=4, help="Number of CPU threads.")
    parser.add_argument(
        "--batchsize", type=int, default=1, help="Number of samples to process at the same time. Defaults to 1."
    )
    parser.add_argument(
        "--audio_speed",
        type=float,
        default=1.0,
        help="Speed factor for audio playback. Values < 1.0 will slow down the audio, values > 1.0 will speed it up. Defaults to 1.0. Values cant go below 0.01.",
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

    args = parser.parse_args()

    run(args.i, args.db, args.overlap, args.threads, args.batchsize, args.audio_speed, args.fmin, args.fmax)

    # A few examples to test
    # python3 embeddings.py --i example/ --o example/ --threads 4
    # python3 embeddings.py --i example/soundscape.wav --o example/soundscape.birdnet.embeddings.txt --threads 4
