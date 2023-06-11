from birdnet.audio import audio
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
    sig, rate = audio.open_audio_file(fpath, config.SAMPLE_RATE)

    # Split into raw audio chunks
    chunks = audio.split_signal(sig, rate, config.SIG_LENGTH, config.SIG_OVERLAP, config.SIG_MINLEN)

    return chunks
