def save_signal(sig, fname: str):
    """Saves a signal to file.

    Args:
        sig: The signal to be saved.
        fname: The file path.
    """
    import soundfile as sf

    sf.write(fname, sig, 48000, "PCM_16")
