from birdnet.segments.segments_finding import find_segments

import numpy


def parse_files(flist: list[dict], max_segments=100):
    """Extracts the segments for all files.
    Args:
        flist: List of dict with {
            "audio": path_to_audio, "result": path_to_result
        }.
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
        segments = find_segments(afile, rfile)

        # Parse segments by species
        for s in segments:
            if s["species"] not in species_segments:
                species_segments[s["species"]] = []

            species_segments[s["species"]].append(s)

    # Shuffle segments for each species and limit to max_segments
    for s in species_segments:
        numpy.random.shuffle(species_segments[s])
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
