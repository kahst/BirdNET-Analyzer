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

# Set numpy random seed
np.random.seed(cfg.RANDOM_SEED)
SCRIPT_DIR = os.path.abspath(os.path.dirname(__file__))


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


def getHeaderMapping(line: str) -> dict:
    rtype = detectRType(line)
    if rtype == "table" or rtype == "audacity":
        sep = "\t"
    else:
        sep = ","

    cols = line.split(sep)

    mapping = {}
    for i, col in enumerate(cols):
        mapping[col] = i

    return mapping


def parseFolders(apath: str, rpath: str, allowed_result_filetypes: list[str] = ["txt", "csv"]) -> list[dict]:
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

    # Check if combined selection table is present and read that.
    if os.path.exists(os.path.join(rpath, cfg.OUTPUT_RAVEN_FILENAME)):
        # Read combined Raven selection table
        rfile = os.path.join(rpath, cfg.OUTPUT_RAVEN_FILENAME)
        data["combined"] = {"isCombinedFile": True, "result": rfile}
    elif os.path.exists(os.path.join(rpath, cfg.OUTPUT_CSV_FILENAME)):
        rfile = os.path.join(rpath, cfg.OUTPUT_CSV_FILENAME)
        data["combined"] = {"isCombinedFile": True, "result": rfile}
    elif os.path.exists(os.path.join(rpath, cfg.OUTPUT_KALEIDOSCOPE_FILENAME)):
        rfile = os.path.join(rpath, cfg.OUTPUT_KALEIDOSCOPE_FILENAME)
        data["combined"] = {"isCombinedFile": True, "result": rfile}
    elif os.path.exists(os.path.join(rpath, cfg.OUTPUT_RTABLE_FILENAME)):
        rfile = os.path.join(rpath, cfg.OUTPUT_RTABLE_FILENAME)
        data["combined"] = {"isCombinedFile": True, "result": rfile}
    else:
        # Get all audio files
        for root, _, files in os.walk(apath):
            for f in files:
                if f.rsplit(".", 1)[-1].lower() in cfg.ALLOWED_FILETYPES:
                    table_key = os.path.join(root.strip(apath), f.rsplit(".", 1)[0])
                    data[table_key] = {"audio": os.path.join(root, f), "result": ""}

        # Get all result files
        for root, _, files in os.walk(rpath):
            for f in files:
                if f.rsplit(".", 1)[-1] in allowed_result_filetypes and ".BirdNET." in f:
                    table_key = os.path.join(root.strip(rpath), f.split(".BirdNET.", 1)[0])
                    if table_key in data:
                        data[table_key]["result"] = os.path.join(root, f)

    # Convert to list
    flist = [f for f in data.values() if f["result"]]

    print(f"Found {len(flist)} audio files with valid result file.")

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

    is_combined_rfile = len(flist) == 1 and flist[0].get("isCombinedFile", False)

    if is_combined_rfile:
        rfile = flist[0]["result"]
        segments = findSegmentsFromCombined(rfile)

        # Parse segments by species
        for s in segments:
            if s["species"] not in species_segments:
                species_segments[s["species"]] = []

            species_segments[s["species"]].append(s)
    else:
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


def findSegmentsFromCombined(rfile: str):
    """Extracts the segments from a combined results file

    Args:
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

    if rtype == "audacity":
        raise Exception("Audacity files are not supported for combined results.")

    # Get mapping from the header column
    header_mapping = getHeaderMapping(lines[0])

    # Get start and end times based on rtype
    confidence = 0
    start = end = 0.0
    species = ""
    afile = ""

    for i, line in enumerate(lines):
        if rtype == "table" and i > 0:
            d = line.split("\t")
            file_offset = float(d[header_mapping["File Offset (s)"]])
            start = file_offset
            end = file_offset + (float(d[header_mapping["End Time (s)"]]) - float(d[header_mapping["Begin Time (s)"]]))
            species = d[header_mapping["Species Code"]]
            confidence = float(d[header_mapping["Confidence"]])
            afile = d[header_mapping["Begin Path"]].replace("/", os.sep).replace("\\", os.sep)

        elif rtype == "r" and i > 0:
            d = line.split(",")
            start = float(d[header_mapping["start"]])
            end = float(d[header_mapping["end"]])
            species = d[header_mapping["common_name"]]
            confidence = float(d[header_mapping["confidence"]])
            afile = d[header_mapping["filepath"]].replace("/", os.sep).replace("\\", os.sep)

        elif rtype == "kaleidoscope" and i > 0:
            d = line.split(",")
            start = float(d[header_mapping["OFFSET"]])
            end = float(d[header_mapping["DURATION"]]) + start
            species = d[header_mapping["scientific_name"]]
            confidence = float(d[header_mapping["confidence"]])
            in_dir = d[header_mapping["INDIR"]]
            folder = d[header_mapping["FOLDER"]]
            in_file = d[header_mapping["IN FILE"]]
            afile = os.path.join(in_dir, folder, in_file).replace("/", os.sep).replace("\\", os.sep)

        elif rtype == "csv" and i > 0:
            d = line.split(",")
            start = float(d[header_mapping["Start (s)"]])
            end = float(d[header_mapping["End (s)"]])
            species = d[header_mapping["Common name"]]
            confidence = float(d[header_mapping["Confidence"]])
            afile = d[header_mapping["File"]].replace("/", os.sep).replace("\\", os.sep)

        # Check if confidence is high enough and label is not "nocall"
        if confidence >= cfg.MIN_CONFIDENCE and species.lower() != "nocall" and afile:
            segments.append({"audio": afile, "start": start, "end": end, "species": species, "confidence": confidence})

    return segments


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

    # Get mapping from the header column
    header_mapping = getHeaderMapping(lines[0])

    # Get start and end times based on rtype
    confidence = 0
    start = end = 0.0
    species = ""

    for i, line in enumerate(lines):
        if rtype == "table" and i > 0:
            d = line.split("\t")
            start = float(d[header_mapping["Begin Time (s)"]])
            end = float(d[header_mapping["End Time (s)"]])
            species = d[header_mapping["Species Code"]]
            confidence = float(d[header_mapping["Confidence"]])

        elif rtype == "audacity":
            d = line.split("\t")
            start = float(d[0])
            end = float(d[1])
            species = d[2].split(", ")[1]
            confidence = float(d[-1])

        elif rtype == "r" and i > 0:
            d = line.split(",")
            start = float(d[header_mapping["start"]])
            end = float(d[header_mapping["end"]])
            species = d[header_mapping["common_name"]]
            confidence = float(d[header_mapping["confidence"]])

        elif rtype == "kaleidoscope" and i > 0:
            d = line.split(",")
            start = float(d[header_mapping["OFFSET"]])
            end = float(d[header_mapping["DURATION"]]) + start
            species = d[header_mapping["scientific_name"]]
            confidence = float(d[header_mapping["confidence"]])

        elif rtype == "csv" and i > 0:
            d = line.split(",")
            start = float(d[header_mapping["Start (s)"]])
            end = float(d[header_mapping["End (s)"]])
            species = d[header_mapping["Common name"]]
            confidence = float(d[header_mapping["Confidence"]])

        # Check if confidence is high enough and label is not "nocall"
        if confidence >= cfg.MIN_CONFIDENCE and species.lower() != "nocall":
            segments.append({"audio": afile, "start": start, "end": end, "species": species, "confidence": confidence})

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
                seg_name = "{:.3f}_{}_{}_{:.1f}s_{:.1f}s.wav".format(
                    seg["confidence"],
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
    parser = argparse.ArgumentParser(description="Extract segments from audio files based on BirdNET detections.")
    parser.add_argument(
        "--audio", default=os.path.join(SCRIPT_DIR, "example/"), help="Path to folder containing audio files."
    )
    parser.add_argument(
        "--results", default=os.path.join(SCRIPT_DIR, "example/"), help="Path to folder containing result files."
    )
    parser.add_argument(
        "--o", default=os.path.join(SCRIPT_DIR, "example/"), help="Output folder path for extracted segments."
    )
    parser.add_argument(
        "--min_conf",
        type=float,
        default=0.1,
        help="Minimum confidence threshold. Values in [0.01, 0.99]. Defaults to 0.1.",
    )
    parser.add_argument(
        "--max_segments", type=int, default=100, help="Number of randomly extracted segments per species."
    )
    parser.add_argument(
        "--seg_length", type=float, default=3.0, help="Length of extracted segments in seconds. Defaults to 3.0."
    )
    parser.add_argument(
        "--threads", type=int, default=min(8, max(1, multiprocessing.cpu_count() // 2)), help="Number of CPU threads."
    )

    args = parser.parse_args()

    # Parse audio and result folders
    cfg.FILE_LIST = parseFolders(args.audio, args.results)

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
