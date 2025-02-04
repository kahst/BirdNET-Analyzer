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
import json

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
    chunks = analyze.getRawAudioFromFile(queryfile_path, 0, 3 * cfg.AUDIO_SPEED) #TODO check if audiospeed keeps it to 3 seconds 
    samples = [chunks[0]]
    data = np.array(samples, dtype="float32")
    query = model.embeddings(data)[0]
    return query

def getDatabase(database_path):
    return hpl.SQLiteUsearchDB.create(database_path).thread_split()

def getSearchResults(queryfile_path, db, n_results, audio_speed, fmin, fmax, score_function: str):
    # Set bandpass frequency range
    cfg.BANDPASS_FMIN = max(0, min(cfg.SIG_FMAX, int(fmin)))
    cfg.BANDPASS_FMAX = max(cfg.SIG_FMIN, min(cfg.SIG_FMAX, int(fmax)))
    cfg.AUDIO_SPEED = max(0.01, audio_speed)

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
        raise ValueError("Invalid score function. Choose 'cosine', 'euclidean' or 'dot'.")

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


def run(queryfile_path, database_path, output_folder, n_results, score_function):
    # Create output folder
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Load the database
    db = getDatabase(database_path)

    settings_path = os.path.join(database_path, "birdnet_analyzer_settings.json")
    if not os.path.exists(settings_path):
        raise ValueError("Database settings file not found.")
    
    with open(settings_path, 'r') as f:
        settings = json.load(f)
    fmin = settings["BANDPASS_FMIN"]
    fmax = settings["BANDPASS_FMAX"]
    audio_speed = settings["AUDIO_SPEED"]

    # Execute the search
    results = getSearchResults(queryfile_path, db, n_results, audio_speed, fmin, fmax, score_function)

    # Save the results
    for i, r in enumerate(results):
        embedding_source = db.get_embedding_source(r.embedding_id)
        file = embedding_source.source_id
        offset = embedding_source.offsets[0] * audio_speed
        duration = 3 * audio_speed
        sig, _ = audio.openAudioFile(file, offset=offset, duration=duration, fmin=fmin, fmax=fmax, speed=audio_speed)
        result_path = os.path.join(output_folder, f"search_result_{i+1}_score_{r.sort_score:.5f}.wav")
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
        default="example/hoplite-db",
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
        "--score_function",
        default="cosine",
        help="Scoring function to use. Choose 'cosine', 'euclidean' or 'dot'. Defaults to 'cosine'."
    )

    args = parser.parse_args()

    run(args.queryfile, args.db, args.o, args.n_results, args.score_function)
