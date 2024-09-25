from chirp.projects.hoplite import brutalism
from chirp.projects.hoplite import sqlite_impl as hpl
import analyze
import argparse
import config as cfg
import model
import numpy as np
import os
import audio

SCRIPT_DIR = os.path.abspath(os.path.dirname(__file__))

def getQueryEmbedding(queryfile_path):
    """
    Extracts the embedding for a query file. Reads only the first 3 seconds
    Args:
        queryfile_path: The path to the query file.
    Returns:
        The query embedding.
    """
    chunks = analyze.getRawAudioFromFile(queryfile_path, 0, 3)
    samples = [chunks[0]]
    data = np.array(samples, dtype="float32")
    query = model.embeddings(data)[0]
    return query

def run(queryfile_path, database_path, output_folder, n_results, fmin, fmax):
    # Set bandpass frequency range
    cfg.BANDPASS_FMIN = max(0, min(cfg.SIG_FMAX, int(fmin)))
    cfg.BANDPASS_FMAX = max(cfg.SIG_FMIN, min(cfg.SIG_FMAX, int(fmax)))

    # Create output folder
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Create database
    db = hpl.SQLiteGraphSearchDB.create(database_path, 1024)
    
    # Get query embedding
    query_embedding = getQueryEmbedding(queryfile_path)

    # Score function according to perch repo
    score_fn = np.dot

    # Execute the search
    results, scores = brutalism.threaded_brute_search(db, query_embedding, n_results, score_fn)

    # Save the results
    for i, r in enumerate(results):
        embedding_source = db.get_embedding_source(r.embedding_id)
        file = embedding_source.source_id
        sig, _ = audio.openAudioFile(file, offset=embedding_source.offsets[0], duration=3)
        result_path = os.path.join(output_folder, f"search_result_{i+1}.wav")
        audio.saveSignal(sig, result_path)

if __name__ == "__main__":
    # Parse arguments
    parser = argparse.ArgumentParser(description="Search audio with BirdNET embeddings")
    parser.add_argument(
        "--queryfile",
        default="example/search_test.wav",
        help="Path to the query file. Only the first 3 seconds will be used for the search."
    )
    parser.add_argument(
        "--db",
        default="example/hoplite-db/db.sqlite",
        help="Path to the Hoplite database. Defaults to example/hoplite-db/db.sqlite.",
    )
    parser.add_argument(
        "--o",
        default="example/search_results",
        help="Path to the output folder."
    )
    parser.add_argument(
        "--n_results",
        default=10,
        help="Number of results to return."
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

    run(args.queryfile, args.db, args.o, args.n_results ,args.fmin, args.fmax)
