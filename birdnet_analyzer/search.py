from perch_hoplite.db import brutalism
from perch_hoplite.db import sqlite_usearch_impl as hpl
import birdnet_analyzer.analyze as analyze
import birdnet_analyzer.audio as audio
import birdnet_analyzer.model as model
import argparse
import birdnet_analyzer.config as cfg
import numpy as np
from scipy.spatial.distance import euclidean
import os

SCRIPT_DIR = os.path.abspath(os.path.dirname(__file__))

def cosine_sim(a, b):
    if a.ndim == 2:
        return np.array([cosine_sim(a[i], b) for i in range(a.shape[0])])
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

def euclidean_scoring(a, b):
    if a.ndim == 2:
        return np.array([euclidean_scoring(a[i], b) for i in range(a.shape[0])])
    return euclidean(a, b)

def euclidean_scoring_inverse(a, b):
    return -euclidean_scoring(a, b)
    
def getQueryEmbedding(queryfile_path):
    """
    Extracts the embedding for a query file. Reads only the first 3 seconds
    Args:
        queryfile_path: The path to the query file.
    Returns:
        The query embedding.
    """
    chunks = analyze.getRawAudioFromFile(queryfile_path, 0, 3) #TODO 
    samples = [chunks[0]]
    data = np.array(samples, dtype="float32")
    query = model.embeddings(data)[0]
    return query

def getDatabase(database_path):
    return hpl.SQLiteUsearchDB.create(database_path).thread_split()

def getSearchResults(queryfile_path, db, n_results, fmin, fmax, score_function: str):
    # Set bandpass frequency range
    cfg.BANDPASS_FMIN = max(0, min(cfg.SIG_FMAX, int(fmin)))
    cfg.BANDPASS_FMAX = max(cfg.SIG_FMIN, min(cfg.SIG_FMAX, int(fmax)))

    # Get query embedding
    query_embedding = getQueryEmbedding(queryfile_path)

    # Set score function
    if score_function == "cosine":
        score_fn = cosine_sim
    elif score_function == "dot":
        score_fn = np.dot
    elif score_function == "euclidean":
        score_fn = euclidean_scoring_inverse # TODO: this is a bit hacky since the search function expects the score to be high for similar embeddings
    else:
        raise ValueError("Invalid score function. Choose 'cosine' or 'dot'.")

    db_embeddings_count = db.count_embeddings()

    if n_results > db_embeddings_count-1:
        n_results = db_embeddings_count-1

    results, scores = brutalism.threaded_brute_search(db, query_embedding, n_results, score_fn) # Threaded Brute-search not working with cosine
    sorted_results = results.search_results
    sorted_results.sort(key=lambda x: x.sort_score, reverse=True)

    if score_function == "euclidean":
        for result in sorted_results:
            result.sort_score *= -1

    return sorted_results


def run(queryfile_path, database_path, output_folder, n_results, fmin, fmax):
    # Create output folder
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Load the database
    db = getDatabase(database_path)

    # Execute the search
    results = getSearchResults(queryfile_path, db, n_results, fmin, fmax)

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
