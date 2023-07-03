from birdnet.configuration import config
from birdnet.segments.result_file_type_detection import detect_result_file_type
from birdnet.utils.lines_reading import read_lines

def find_segments(afile: str, rfile: str):
    """Extracts the segments for an audio file from the results file
    Args:
        afile: Path to the audio file.
        rfile: Path to the result file.
    Returns:
        A list of dicts in the form of
        {
            "audio": afile,
            "start": start,
            "end": end,
            "species": species,
            "confidence": confidence,
        }
    """
    segments: list[dict] = []

    # Open and parse result file
    lines = read_lines(rfile)

    # Auto-detect result type
    rtype = detect_result_file_type(lines[0])

    # Get start and end times based on rtype
    confidence = 0
    start = end = 0.0
    species = ""

    for i, line in enumerate(lines):
        if rtype == "table" and i > 0:
            d = line.split("\t")
            start = float(d[3])
            end = float(d[4])
            species = d[-2]
            confidence = float(d[-1])

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

        # Check if confidence is high enough
        if confidence >= config.MIN_CONFIDENCE:
            segments.append(
                {
                    "audio": afile,
                    "start": start,
                    "end": end,
                    "species": species,
                    "confidence": confidence,
                }
            )

    return segments
