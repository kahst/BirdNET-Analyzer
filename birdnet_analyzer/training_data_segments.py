"""Extract segments from audio files based on BirdNET detections.

Can be used to save the segments of the audio files for each detection.
"""

import argparse
import multiprocessing
import os
from multiprocessing import Pool

import numpy as np

import birdnet_analyzer.audio as audio
import birdnet_analyzer.config as cfg
import birdnet_analyzer.utils as utils

from birdnet_analyzer.segments import getHeaderMapping

# Set numpy random seed
np.random.seed(cfg.RANDOM_SEED)
SCRIPT_DIR = os.path.abspath(os.path.dirname(__file__))

def parseFolders(apath: str, rpath: str, annotation_file_suffix: str, allowed_result_filetypes: list[str] = ["txt", "csv"]) -> list[dict]:
    """Read audio and result files.

    Reads all audio files and BirdNET output inside directory recursively.

    Args:
        apath: Path to search for audio files.
        rpath: Path to search for result files.
        allowed_result_filetypes: List of extensions for the result files.

    Returns:
        A list of {"audio": path_to_audio, "result": path_to_result }.
    """
    data = {}
    apath = apath.replace("/", os.sep).replace("\\", os.sep)
    rpath = rpath.replace("/", os.sep).replace("\\", os.sep)

    # Get all audio files
    for root, _, files in os.walk(apath):
        for f in files:
            if f.rsplit(".", 1)[-1].lower() in cfg.ALLOWED_FILETYPES:
                table_key = os.path.join(root.strip(apath), f.rsplit(".", 1)[0])
                data[table_key] = {"audio": os.path.join(root, f), "result": ""}

    # Get all result files
    for root, _, files in os.walk(rpath):
        for f in files:
            if f.rsplit(".", 1)[-1] in allowed_result_filetypes and f".{annotation_file_suffix}." in f: # TODO use suffix from params
                table_key = os.path.join(root.strip(rpath), f.split(f".{annotation_file_suffix}.", 1)[0])
                if table_key in data:
                    data[table_key]["result"] = os.path.join(root, f)

    # Convert to list
    flist = [f for f in data.values() if f["result"]]

    print(f"Found {len(flist)} audio files with valid result file.")

    return flist


def parseFiles(flist: list[dict], max_segments, species_column_name: str):
    """Extracts the segments for all files.

    Args:
        flist: List of dict with {"audio": path_to_audio, "result": path_to_result }.
        max_segments: Number of segments per species.

    Returns:
        TODO @kahst
    """
    species_segments: dict[str, list] = {}

    for f in flist:
        # Paths
        afile = f["audio"]
        rfile = f["result"]

        # Get all segments for result file
        segments = findTrainingDataSegments(afile, rfile, species_column_name)

        # Parse segments by species
        for s in segments:
            if s["species"] not in species_segments:
                species_segments[s["species"]] = []

            species_segments[s["species"]].append(s)

    # Shuffle segments for each species and limit to max_segments
    for s in species_segments:
        np.random.shuffle(species_segments[s])
        species_segments[s] = species_segments[s][:max_segments]

    # Make dict of segments per audio file
    segments: dict[str, list] = {}
    seg_cnt = 0

    for s in species_segments:
        for seg in species_segments[s]:
            if seg["audio"] not in segments:
                segments[seg["audio"]] = []

            segments[seg["audio"]].append(seg)
            seg_cnt += 1

    print(f"Found {seg_cnt} segments in {len(segments)} audio files.")

    # Convert to list
    flist = [tuple(e) for e in segments.items()]

    return flist

def findTrainingDataSegments(afile: str, rfile: str, species_column_name: str):
    """Extracts segments for training data from a result file.
    
    Args:
        afile: Path to the audio file
        rfile: Path to the result file.

    Returns:
        A list of dicts in the form of
        {"audio": afile, "start": start, "end": end, "species": species, "confidence": confidence}
    """
    overlap = 0.5

    segments:list[dict] = [] 

    # Open and parse result file
    lines = utils.readLines(rfile)
    
    # Get mapping from the header column
    header_mapping = getHeaderMapping(lines[0])

    # Get start and end times based on rtype
    start = end = 0.0
    species = ""

    # bounding boxes with start, end, species
    bounding_boxes = [] 

    # Extract bounding boxes
    for i, line in enumerate(lines):
        if i == 0:
            continue
        d = line.split("\t")
        start = float(d[header_mapping["Begin Time (s)"]])
        end = float(d[header_mapping["End Time (s)"]])
        species = d[header_mapping[species_column_name]]
        bounding_boxes.append((start, end, species))
    
    # sort bounding boxes by start time
    bounding_boxes = sorted(bounding_boxes, key=lambda x: x[0])

    # get end of the last bounding box
    file_end = max(bounding_boxes, key=lambda x: x[1])[1]

    # define the starts for all segments
    segment_starts = range(0, int(file_end), int(cfg.SIG_LENGTH))

    # define all segments
    all_segments = [(x, x + cfg.SIG_LENGTH) for x in segment_starts]

    # initialize species per segment
    species_per_segments = {segment: [] for segment in all_segments}

    # iterate over all bounding boxes
    for box in bounding_boxes:
        bb_start = box[0]
        bb_end = box[1]
        bb_species = box[2]

        # number of segments found
        n_segments = 0

        # segments that are too short
        short_segments = []

        # iterate over all segments
        for segment in all_segments:
            # calculate overlap
            overlap_start = max(bb_start, segment[0])
            overlap_end = min(bb_end, segment[1])
            overlap_duration = max(0, overlap_end - overlap_start)

            # check if overlap with segment is long enough
            if overlap_duration > overlap:
                n_segments += 1
                if bb_species not in species_per_segments[segment]:
                    species_per_segments[segment].append(bb_species)
            # if not store segment for later
            elif overlap_duration > 0:
                short_segments.append((segment, overlap_duration))
        
        # if no segment was found with enough overlap use the longest of the remaining segments
        if n_segments == 0 and len(short_segments) > 0:
            longest_overlap_segment = max(short_segments, key=lambda x: x[1])[0]
            if bb_species not in species_per_segments[longest_overlap_segment]:
                species_per_segments[longest_overlap_segment].append(bb_species)

    # create result list, concatenate sorted species or use 'noise' as label if no species was found
    for segment, species in species_per_segments.items():
        species_string = ','.join(sorted(species)) if len(species) > 0  else "noise"
        segments.append({"audio": afile, "start": segment[0], "end": segment[1], "species": species_string})

    return segments 

def extractSegments(item: tuple[tuple[str, list[dict]], float, dict[str]]):
    """Saves each segment separately.

    Creates an audio file for each species segment.

    Args:
        item: A tuple that contains ((audio file path, segments), segment length, config)
    """
    # Paths and config
    afile = item[0][0]
    segments = item[0][1]
    seg_length = item[1]
    cfg.setConfig(item[2])

    # Status
    print(f"Extracting segments from {afile}")

    try:
        # Open audio file
        sig, _ = audio.openAudioFile(afile, cfg.SAMPLE_RATE)
    except Exception as ex:
        print(f"Error: Cannot open audio file {afile}", flush=True)
        utils.writeErrorLog(ex)

        return

    # Extract segments
    for seg_cnt, seg in enumerate(segments, 1):
        try:
            # Get start and end times
            start = int(seg["start"] * cfg.SAMPLE_RATE)
            end = int(seg["end"] * cfg.SAMPLE_RATE)
            offset = ((seg_length * cfg.SAMPLE_RATE) - (end - start)) // 2
            start = max(0, start - offset)
            end = min(len(sig), end + offset)

            # Make sure segment is long enough
            if end > start:
                # Get segment raw audio from signal
                seg_sig = sig[int(start) : int(end)]

                # Make output path
                outpath = os.path.join(cfg.OUTPUT_PATH, seg["species"])
                os.makedirs(outpath, exist_ok=True)

                # Save segment
                seg_name = "{}_{}_{:.1f}s_{:.1f}s.wav".format(
                    seg_cnt,
                    seg["audio"].rsplit(os.sep, 1)[-1].rsplit(".", 1)[0],
                    seg["start"],
                    seg["end"],
                )
                seg_path = os.path.join(outpath, seg_name)
                audio.saveSignal(seg_sig, seg_path)

        except Exception as ex:
            # Write error log
            print(f"Error: Cannot extract segments from {afile}.", flush=True)
            utils.writeErrorLog(ex)
            return False

    return True


if __name__ == "__main__":
    # Parse arguments
    parser = argparse.ArgumentParser(description="Extract training data  from audio files based on raven selection tables.")
    parser.add_argument(
        "--audio", default=os.path.join(SCRIPT_DIR, "example/"), help="Path to folder containing audio files."
    )
    parser.add_argument(
        "--annotations", default=os.path.join(SCRIPT_DIR, "example/"), help="Path to folder containing annotation files."
    )
    parser.add_argument(
        "--o", default=os.path.join(SCRIPT_DIR, "example/"), help="Output folder path for extracted segments."
    )
    parser.add_argument(
        "--max_segments", type=int, default=100, help="Maximum number of randomly extracted segments per species."
    )
    parser.add_argument(
        "--seg_length", type=float, default=3.0, help="Length of extracted segments in seconds. Defaults to 3.0."
    )
    parser.add_argument(
        "--threads", type=int, default=min(8, max(1, multiprocessing.cpu_count() // 2)), help="Number of CPU threads."
    )
    parser.add_argument(
        "--annotation_files_suffix", type=str, default="annotation", help="Suffix of the annotation files for the training data. Annotation files need to be named like '<audio_file_name>.<suffix>.txt'."
    )
    parser.add_argument(
        "--species_column_name", type=str, default="species", help="Name of the column that specifies the species in the annotation files."
    )

    args = parser.parse_args()

    # Parse audio and result folders
    cfg.FILE_LIST = parseFolders(args.audio, args.annotations, args.annotation_files_suffix)

    # Set output folder
    cfg.OUTPUT_PATH = args.o

    # Set number of threads
    cfg.CPU_THREADS = int(args.threads)

    # Parse file list and make list of segments
    cfg.FILE_LIST = parseFiles(cfg.FILE_LIST, max(1, int(args.max_segments)), args.species_column_name)

    # Add config items to each file list entry.
    # We have to do this for Windows which does not
    # support fork() and thus each process has to
    # have its own config. USE LINUX!
    flist = [(entry, max(cfg.SIG_LENGTH, float(args.seg_length)), cfg.getConfig()) for entry in cfg.FILE_LIST]

    # Extract segments
    if cfg.CPU_THREADS < 2:
        for entry in flist:
            extractSegments(entry)
    else:
        with Pool(cfg.CPU_THREADS) as p:
            p.map(extractSegments, flist)

    # A few examples to test
    # python3 segments.py --audio example/ --results example/ --o example/segments/
    # python3 segments.py --audio example/ --results example/ --o example/segments/ --seg_length 5.0 --min_conf 0.1 --max_segments 100 --threads 4
