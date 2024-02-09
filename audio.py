"""Module containing audio helper functions.
"""
import numpy as np

import config as cfg

RANDOM = np.random.RandomState(cfg.RANDOM_SEED)


def openAudioFile(path: str, sample_rate=48000, offset=0.0, duration=None, fmin=cfg.SIG_FMIN, fmax=cfg.SIG_FMAX):
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

    sig, rate = librosa.load(path, sr=sample_rate, offset=offset, duration=duration, mono=True, res_type="kaiser_fast")

    # Bandpass filter
    sig = bandpass(sig, rate, fmin, fmax)

    return sig, rate

def getAudioFileLength(path, sample_rate=48000):    
    
    # Open file with librosa (uses ffmpeg or libav)
    import librosa

    return librosa.get_duration(filename=path, sr=sample_rate)

def get_sample_rate(path: str):
    import librosa
    return librosa.get_samplerate(path)


def saveSignal(sig, fname: str):
    """Saves a signal to file.

    Args:
        sig: The signal to be saved.
        fname: The file path.
    """
    import soundfile as sf

    sf.write(fname, sig, 48000, "PCM_16")


def noise(sig, shape, amount=None):
    """Creates noise.

    Creates a noise vector with the given shape.

    Args:
        sig: The original audio signal.
        shape: Shape of the noise.
        amount: The noise intensity.

    Returns:
        An numpy array of noise with the given shape.
    """
    # Random noise intensity
    if amount == None:
        amount = RANDOM.uniform(0.1, 0.5)

    # Create Gaussian noise
    try:
        noise = RANDOM.normal(min(sig) * amount, max(sig) * amount, shape)
    except:
        noise = np.zeros(shape)

    return noise.astype("float32")


def splitSignal(sig, rate, seconds, overlap, minlen):
    """Split signal with overlap.

    Args:
        sig: The original signal to be split.
        rate: The sampling rate.
        seconds: The duration of a segment.
        overlap: The overlapping seconds of segments.
        minlen: Minimum length of a split.
    
    Returns:
        A list of splits.
    """
    sig_splits = []

    for i in range(0, len(sig), int((seconds - overlap) * rate)):
        split = sig[i : i + int(seconds * rate)]

        # End of signal?
        if len(split) < int(minlen * rate) and len(sig_splits) > 0:
            break

        # Signal chunk too short?
        if len(split) < int(rate * seconds):
            split = np.hstack((split, noise(split, (int(rate * seconds) - len(split)), 0.5)))

        sig_splits.append(split)

    return sig_splits


def cropCenter(sig, rate, seconds):
    """Crop signal to center.

    Args:
        sig: The original signal.
        rate: The sampling rate.
        seconds: The length of the signal.
    """
    if len(sig) > int(seconds * rate):
        start = int((len(sig) - int(seconds * rate)) / 2)
        end = start + int(seconds * rate)
        sig = sig[start:end]

    # Pad with noise
    elif len(sig) < int(seconds * rate):
        sig = np.hstack((sig, noise(sig, (int(seconds * rate) - len(sig)), 0.5)))

    return sig

def bandpass(sig, rate, fmin, fmax):

    if (fmin == cfg.SIG_FMIN and fmax == cfg.SIG_FMAX) or fmin >= fmax:
        return sig

    from scipy.signal import butter, lfilter
    nyquist = 0.5 * rate

    # Lowpass?
    if fmin > cfg.SIG_FMIN and fmax == cfg.SIG_FMAX:  
        
        low = fmin / nyquist
        b, a = butter(5, low, btype="low")
        sig = lfilter(b, a, sig)

    # Highpass?
    elif fmin == cfg.SIG_FMIN and fmax < cfg.SIG_FMAX:

        high = fmax / nyquist
        b, a = butter(5, high, btype="high")
        sig = lfilter(b, a, sig)

    # Bandpass?
    elif fmin > cfg.SIG_FMIN and fmax < cfg.SIG_FMAX:

        low = fmin / nyquist
        high = fmax / nyquist
        b, a = butter(5, [low, high], btype="band")
        sig = lfilter(b, a, sig)

    return sig
