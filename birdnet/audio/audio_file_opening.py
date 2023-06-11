def open_audio_file(path: str, sample_rate=48000, offset=0.0, duration=None):
    """Open an audio file.

    Opens an audio file with librosa and the given settings.

    Args:
        path: Path to the audio file.
        sample_rate: The sample rate at which the file should be processed.
        offset: The starting offset.
        duration: Maximum duration of the loaded content.

    Returns:
        Returns the audio time series and the sampling rate.
    """
    # Open file with librosa (uses ffmpeg or libav)
    import librosa

    sig, rate = librosa.load(
        path,
        sr=sample_rate,
        offset=offset,
        duration=duration,
        mono=True,
        res_type="kaiser_fast",
    )

    return sig, rate
