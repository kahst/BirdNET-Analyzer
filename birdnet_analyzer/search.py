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
from perch_hoplite.db.search_results import SearchResult

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
 
    # Load audio
    sig, rate = audio.openAudioFile(
        queryfile_path,
        duration=cfg.SIG_LENGTH * cfg.AUDIO_SPEED if cfg.SAMPLE_CROP_MODE == "first" else None,
        fmin=cfg.BANDPASS_FMIN,
        fmax=cfg.BANDPASS_FMAX,
        speed=cfg.AUDIO_SPEED,
    )

    # Crop query audio
    if cfg.SAMPLE_CROP_MODE == "center":
        sig_splits = [audio.cropCenter(sig, rate, cfg.SIG_LENGTH)]
    elif cfg.SAMPLE_CROP_MODE == "first":
        sig_splits = [audio.splitSignal(sig, rate, cfg.SIG_LENGTH, cfg.SIG_OVERLAP, cfg.SIG_MINLEN)[0]]
    else:
        sig_splits = audio.splitSignal(sig, rate, cfg.SIG_LENGTH, cfg.SIG_OVERLAP, cfg.SIG_MINLEN)

    #chunks = analyze.getRawAudioFromFile(queryfile_path, 0, 3 * cfg.AUDIO_SPEED) #TODO: Crop Mode  
    samples = sig_splits
    data = np.array(samples, dtype="float32")
    query = model.embeddings(data)
    return query

def getDatabase(database_path):
    return hpl.SQLiteUsearchDB.create(database_path).thread_split()

def getSearchResults(queryfile_path, db, n_results, audio_speed, fmin, fmax, score_function: str, crop_mode, crop_overlap):
    # Set bandpass frequency range
    cfg.BANDPASS_FMIN = max(0, min(cfg.SIG_FMAX, int(fmin)))
    cfg.BANDPASS_FMAX = max(cfg.SIG_FMIN, min(cfg.SIG_FMAX, int(fmax)))
    cfg.AUDIO_SPEED = max(0.01, audio_speed)
    cfg.SAMPLE_CROP_MODE = crop_mode
    cfg.SIG_OVERLAP = max(0.0, min(2.9, float(crop_overlap)))

    # Get query embedding
    query_embeddings = getQueryEmbedding(queryfile_path)

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

    scores_by_embedding_id = {} 

    for embedding in query_embeddings:
        results, scores = brutalism.threaded_brute_search(db, embedding, n_results, score_fn)
        sorted_results = results.search_results

        if score_function == "euclidean":
            for result in sorted_results:
                result.sort_score *= -1
        
        for result in sorted_results:
            if result.embedding_id not in scores_by_embedding_id:
                scores_by_embedding_id[result.embedding_id] = []
            scores_by_embedding_id[result.embedding_id].append(result.sort_score)

    results = []

    for embedding_id, scores in scores_by_embedding_id.items():
        results.append(SearchResult(embedding_id, np.sum(scores) / len(query_embeddings)))

    results.sort(key=lambda x: x.sort_score, reverse=True)

    return results[0:n_results]


def run(queryfile_path, database_path, output_folder, n_results, score_function, crop_mode, crop_overlap):
    # Create output folder
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Load the database
    db = getDatabase(database_path)

    try:
        settings = db.get_metadata('birdnet_analyzer_settings')
    except:
        raise ValueError("No settings present in database.")
    
    fmin = settings["BANDPASS_FMIN"]
    fmax = settings["BANDPASS_FMAX"]
    audio_speed = settings["AUDIO_SPEED"]

    # Execute the search
    results = getSearchResults(queryfile_path, db, n_results, audio_speed, fmin, fmax, score_function, crop_mode, crop_overlap)

    # Save the results
    for i, r in enumerate(results):
        embedding_source = db.get_embedding_source(r.embedding_id)
        file = embedding_source.source_id
        offset = embedding_source.offsets[0] * audio_speed
        duration = 3 * audio_speed
        sig, _ = audio.openAudioFile(file, offset=offset, duration=duration, sample_rate=None)
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
    parser.add_argument(
        "--crop_mode",
        default="center",
    )
    parser.add_argument(
        "--crop_overlap",
        type=float,
        default=0.0,
        help="Overlap of training data segments in seconds if crop_mode is 'segments'. Defaults to 0.",
    )

    args = parser.parse_args()

    run(args.queryfile, args.db, args.o, args.n_results, args.score_function, args.crop_mode, args.crop_overlap)
