import os
from typing import Tuple

import birdnet.audio.audio_file_opening
import birdnet.audio.signal_saving
import birdnet.embeddings.error_log_writing
from birdnet.configuration import config
from birdnet.utils.error_log_writing import write_error_log


def extract_segments(
    item: Tuple[Tuple[str, list[dict]], float, dict[str, str]],
):
    """Saves each segment separately.
    Creates an audio file for each species segment.

    Args:
        item: A tuple that contains (
            (audio file path, segments),
            segment length,
            config
        )
    """
    # Paths and config
    afile = item[0][0]
    segments = item[0][1]
    seg_length = item[1]
    config.set_config(item[2])

    # Status
    print(f"Extracting segments from {afile}")

    try:
        # Open audio file
        sig, _ = birdnet.audio.audio_file_opening.open_audio_file(
            afile,
            config.SAMPLE_RATE,
        )
    except Exception as ex:
        print(f"Error: Cannot open audio file {afile}", flush=True)
        write_error_log(ex)

        return

    # Extract segments
    for seg_cnt, seg in enumerate(segments, 1):
        try:
            # Get start and end times
            start = int(seg["start"] * config.SAMPLE_RATE)
            end = int(seg["end"] * config.SAMPLE_RATE)
            offset = ((seg_length * config.SAMPLE_RATE) - (end - start)) // 2
            start = max(0, start - offset)
            end = min(len(sig), end + offset)

            # Make sure segment is long enough
            if end > start:
                # Get segment raw audio from signal
                seg_sig = sig[int(start) : int(end)]

                # Make output path
                outpath = os.path.join(config.OUTPUT_PATH, seg["species"])
                os.makedirs(outpath, exist_ok=True)

                # Save segment
                seg_name = "{:.3f}_{}_{}.wav".format(
                    seg["confidence"],
                    seg_cnt,
                    seg["audio"].rsplit(os.sep, 1)[-1].rsplit(".", 1)[0]
                )
                seg_path = os.path.join(outpath, seg_name)
                birdnet.audio.signal_saving.save_signal(seg_sig, seg_path)

        except Exception as ex:
            # Write error log
            print(f"Error: Cannot extract segments from {afile}.", flush=True)
            write_error_log(ex)
            return False

    return True
