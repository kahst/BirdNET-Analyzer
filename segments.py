"""Extract segments from audio files based on BirdNET detections.

Can be used to save the segments of the audio files for each detection.
"""
import argparse
import multiprocessing
import os
from multiprocessing import Pool

import numpy as np

import audio
import config as cfg
import utils

# Set numpy random seed
np.random.seed(cfg.RANDOM_SEED)


def detectRType(line: str):
    """Detects the type of result file.

    Args:
        line: First line of text.

    Returns:
        Either "table", "r", "kaleidoscope", "csv" or "audacity".
    """
    if line.lower().startswith("selection"):
        return "table"
    elif line.lower().startswith("filepath"):
        return "r"
    elif line.lower().startswith("indir"):
        return "kaleidoscope"
    elif line.lower().startswith("start (s)"):
        return "csv"
    else:
        return "audacity"


def parseFolders(apath: str, rpath: str, allowed_result_filetypes: list[str] = ["txt", "csv"], ignore_these_files: str = "") -> list[dict]:
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

    # read the ignore_these_files file
    if ignore_these_files != "":
        with open(ignore_these_files, "r") as f:
            # read and parse the file
            ignore_files = f.read().splitlines()

    # Get all audio files
    n_files_ignored = 0
    for root, _, files in os.walk(apath):
        for f in files:
            if f.rsplit(".", 1)[-1].lower() in cfg.ALLOWED_FILETYPES:
                if ignore_these_files != "":
                    if f in ignore_files:
                        n_files_ignored += 1
                        continue
                data[f.rsplit(".", 1)[0]] = {"audio": os.path.join(root, f), "result": ""}

    # Get all result files
    for root, _, files in os.walk(rpath):
        for f in files:
            if f.rsplit(".", 1)[-1] in allowed_result_filetypes and ".BirdNET." in f:
                if f.split(".BirdNET.", 1)[0] in data:
                    data[f.split(".BirdNET.", 1)[0]]["result"] = os.path.join(root, f)

    # Convert to list
    flist = [f for f in data.values() if f["result"]]

    print(f"Found {len(flist)} audio files with valid result file.")
    if ignore_these_files != "":
        print(f"Ignored {n_files_ignored} files.")

    return flist


def parseFiles(flist: list[dict], max_segments=100):
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
        segments = findSegments(afile, rfile)

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


def findSegments(afile: str, rfile: str):
    """Extracts the segments for an audio file from the results file

    Args:
        afile: Path to the audio file.
        rfile: Path to the result file.

    Returns:
        A list of dicts in the form of
        {"audio": afile, "start": start, "end": end, "species": species, "confidence": confidence}
    """
    segments: list[dict] = []

    # Open and parse result file
    lines = utils.readLines(rfile)

    # Auto-detect result type
    rtype = detectRType(lines[0])

    # Get start and end times based on rtype
    confidence = 0
    start = end = 0.0
    species = ""

    for i, line in enumerate(lines):
        if rtype == "table" and i == 0:
            # get header index for Begin Time (s), End Time (s), Species Code, and Confidence
            header = line.split("\t")
            start_index = header.index("Begin Time (s)")
            end_index = header.index("End Time (s)")
            species_index = header.index("Species Code")
            confidence_index = header.index("Confidence")
        if rtype == "table" and i > 0:
            d = line.split("\t")
            start = float(d[start_index])
            end = float(d[end_index])
            species = d[species_index]
            confidence = float(d[confidence_index])

        elif rtype == "audacity":
            d = line.split("\t")
            start = float(d[0])
            end = float(d[1])
            species = d[2].split(", ")[1]
            confidence = float(d[-1])

        elif rtype == "r" and i > 0:
            d = line.split(",")
            start = float(d[1])
            end = float(d[2])
            species = d[4]
            confidence = float(d[5])

        elif rtype == "kaleidoscope" and i > 0:
            d = line.split(",")
            start = float(d[3])
            end = float(d[4]) + start
            species = d[5]
            confidence = float(d[7])

        elif rtype == "csv" and i > 0:
            d = line.split(",")
            start = float(d[0])
            end = float(d[1])
            species = d[3]
            confidence = float(d[4])

        # Check if confidence is high enough and label is not "nocall"
        if confidence >= cfg.MIN_CONFIDENCE and species.lower() != "nocall":
            segments.append({"audio": afile, "start": start, "end": end, "species": species, "confidence": confidence})

    return segments


def extractSegments(item: tuple[tuple[str, list[dict]], float, dict[str]]):
    """Saves each segment separately.

    Creates an audio file for each species segment.

    Args:
        item: A tuple that contains ((audio file path, segments), segment length, config, use_sox).
    """
    # Paths and config
    afile = item[0][0]
    segments = item[0][1]
    seg_length = item[1]
    cfg.setConfig(item[2])
    use_sox = item[3]

    # Status
    print(f"Extracting segments from {afile}")

    if not use_sox:
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
            if not use_sox:
                start = int(seg["start"] * cfg.SAMPLE_RATE)
                end = int(seg["end"] * cfg.SAMPLE_RATE)
                offset = ((seg_length * cfg.SAMPLE_RATE) - (end - start)) // 2
                start = max(0, start - offset)
                end = min(len(sig), end + offset)
            else:
                start = seg["start"]
                end = seg["end"]

            # Make sure segment is long enough
            if end > start:
                # Get segment raw audio from signal
                if not use_sox:
                    seg_sig = sig[int(start) : int(end)]

                # Make output path
                outpath = os.path.join(cfg.OUTPUT_PATH, seg["species"])
                os.makedirs(outpath, exist_ok=True)

                # Save segment
                seg_name = "{:.3f}_{}_{}_{:.1f}s_{:.1f}s.wav".format(
                    seg["confidence"], seg_cnt, seg["audio"].rsplit(os.sep, 1)[-1].rsplit(".", 1)[0], seg["start"], seg["end"]
                )
                seg_path = os.path.join(outpath, seg_name)
                # skip if the segment already exists
                if os.path.exists(seg_path):
                    continue
                if use_sox: 
                    # save the signal using sox, requires sox to be installed
                    # does not require to import the signal, which is much faster (>100x probably)
                    print("saving signal using sox")
                    saveSignalSOX(afile, seg["start"], seg["end"], seg_path)
                else:
                    audio.saveSignal(seg_sig, seg_path)                

        except Exception as ex:
            # Write error log
            print(f"Error: Cannot extract segments from {afile}.", flush=True)
            utils.writeErrorLog(ex)
            return False

    return True

def saveSignalSOX(afile: str, start: float, end: float, outpath: str):
    """Save signal using sox.

    Args:
        afile: Path to the audio file.
        start: Start time in seconds.
        end: End time in seconds.
        outpath: Path to save the output file.
    """
    
    # we should add quotes around paths to avoid issues with spaces
    # add quotes
    afile = f'"{afile}"'
    outpath = f'"{outpath}"'

    # Create command
    cmd = f"sox {afile} {outpath} trim {start} ={end}"

    # Execute command
    print(f"Executing: {cmd}")
    import subprocess
    subprocess.run(cmd, shell=True)

    return True


if __name__ == "__main__":
    # Parse arguments
    parser = argparse.ArgumentParser(description="Extract segments from audio files based on BirdNET detections.")
    parser.add_argument("--audio", default="example/", help="Path to folder containing audio files.")
    parser.add_argument("--results", default="example/", help="Path to folder containing result files.")
    parser.add_argument("--o", default="example/", help="Output folder path for extracted segments.")
    parser.add_argument(
        "--min_conf", type=float, default=0.1, help="Minimum confidence threshold. Values in [0.01, 0.99]. Defaults to 0.1."
    )
    parser.add_argument("--max_segments", type=int, default=100, help="Number of randomly extracted segments per species.")
    parser.add_argument(
        "--seg_length", type=float, default=3.0, help="Length of extracted segments in seconds. Defaults to 3.0."
    )
    parser.add_argument("--threads", type=int, default=min(8, max(1, multiprocessing.cpu_count() // 2)), help="Number of CPU threads.")
    parser.add_argument("--use_sox", action="store_true", default=False, help="Use sox to extract segments. Default is False.")
    parser.add_argument("--ignore_these_files", type=str, default="", help="Text file path containing list of files to ignore. This may be used to ignore files that were used in training.")

    args = parser.parse_args()

    # Parse audio and result folders
    cfg.FILE_LIST = parseFolders(args.audio, args.results, ignore_these_files=args.ignore_these_files)

    # Set output folder
    cfg.OUTPUT_PATH = args.o

    # Set number of threads
    cfg.CPU_THREADS = int(args.threads)

    # Set confidence threshold
    cfg.MIN_CONFIDENCE = max(0.01, min(0.99, float(args.min_conf)))

    # Parse file list and make list of segments
    cfg.FILE_LIST = parseFiles(cfg.FILE_LIST, max(1, int(args.max_segments)))

    # Add config items to each file list entry.
    # We have to do this for Windows which does not
    # support fork() and thus each process has to
    # have its own config. USE LINUX!
    flist = [(entry, max(cfg.SIG_LENGTH, float(args.seg_length)), cfg.getConfig(), args.use_sox) for entry in cfg.FILE_LIST]

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
