from collections import namedtuple
from multiprocessing import Pool

from birdnet.segments.folders_parsing import parse_folders
from birdnet.segments.files_parsing import parse_files
from birdnet.segments.segments_extracting import extract_segments

from birdnet.configuration import config


def test_segments():
    Arguments = namedtuple(
        'Arguments', "audio results o min_conf max_segments seg_length threads"
    )
    arguments: Arguments = Arguments(
        audio='example/',
        results='example/',
        o='example/',
        min_conf=0.1,
        max_segments=1,
        seg_length=3.0,
        threads=1,
    )

    # Parse audio and result folders
    config.FILE_LIST = parse_folders(arguments.audio, arguments.results)

    # Set output folder
    config.OUTPUT_PATH = arguments.o

    # Set number of threads
    config.CPU_THREADS = int(arguments.threads)

    # Set confidence threshold
    config.MIN_CONFIDENCE = max(0.01, min(0.99, float(arguments.min_conf)))

    # Parse file list and make list of segments
    config.FILE_LIST = parse_files(
        config.FILE_LIST,
        max(
            1,
            int(arguments.max_segments)
        )
    )

    # Add config items to each file list entry.
    # We have to do this for Windows which does not
    # support fork() and thus each process has to
    # have its own config. USE LINUX!
    flist = [
        (
            entry,
            max(
                config.SIG_LENGTH,
                float(arguments.seg_length)
            ),
            config.get_config(),
        )
        for entry in config.FILE_LIST
    ]

    # Extract segments
    if config.CPU_THREADS < 2:
        for entry in flist:
            extract_segments(entry)
    else:
        with Pool(config.CPU_THREADS) as p:
            p.map(extract_segments, flist)

    # A few examples to test
    # python3 segments.py --audio example/ --results example/ --o example/segments/
    # python3 segments.py --audio example/ --results example/ --o example/segments/ --seg_length 5.0 --min_conf 0.1 --max_segments 100 --threads 4
