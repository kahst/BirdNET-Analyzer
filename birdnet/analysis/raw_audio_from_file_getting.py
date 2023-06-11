from birdnet.audio.audio_file_opening import open_audio_file
from birdnet.audio.signal_splitting import split_signal
from birdnet.configuration import config


def get_raw_audio_from_file(fpath: str):
    """Reads an audio file.
    Reads the file and splits the signal into chunks.
    Args:
        fpath: Path to the audio file.
    Returns:
        The signal split into a list of chunks.
    """
    # Open file
    sig, rate = open_audio_file(fpath, config.SAMPLE_RATE)

    # Split into raw audio chunks
    chunks = split_signal(
        sig, rate, config.SIG_LENGTH, config.SIG_OVERLAP, config.SIG_MINLEN
    )

    return chunks
