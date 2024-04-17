"""Module containing audio helper functions.
"""
import numpy as np

import config as cfg

RANDOM = np.random.RandomState(cfg.RANDOM_SEED)


def openAudioFile(path: str, sample_rate=48000, offset=0.0, duration=None, fmin=None, fmax=None):
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
    if fmin != None and fmax != None:
        sig = bandpass(sig, rate, fmin, fmax)
        #sig = bandpassKaiserFIR(sig, rate, fmin, fmax)

    return sig, rate

def getAudioFileLength(path, sample_rate=48000):    
    
    # Open file with librosa (uses ffmpeg or libav)
    import librosa

    return int(librosa.get_duration(filename=path, sr=sample_rate))

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


def pad(sig, seconds, srate, amount=None):
    """Creates noise.

    Creates a noise vector with the given shape.

    Args:
        sig: The original audio signal.
        shape: Shape of the noise.
        amount: The noise intensity.

    Returns:
        An numpy array of noise with the given shape.
    """

    target_len = int(srate * seconds)

    if len(sig) < target_len:
        noise_shape = target_len - len(sig)

        if not cfg.USE_NOISE:
            noise = np.zeros(noise_shape)
        else:
            # Random noise intensity
            if amount == None:
                amount = RANDOM.uniform(0.1, 0.5)

            # Create Gaussian noise
            try:
                noise = RANDOM.normal(min(sig) * amount, max(sig) * amount, noise_shape)
            except:
                noise = np.zeros(noise_shape)

        return np.hstack((sig, noise.astype("float32")))
    
    return sig


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

        split = pad(split, seconds, rate, 0.5)

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
    else:
        sig = pad(sig, seconds, rate, 0.5)

    return sig

def bandpass(sig, rate, fmin, fmax, order=5):

    # Check if we have to bandpass at all
    if fmin == cfg.SIG_FMIN and fmax == cfg.SIG_FMAX or fmin > fmax:
        return sig

    from scipy.signal import butter, lfilter
    nyquist = 0.5 * rate

    # Highpass?
    if fmin > cfg.SIG_FMIN and fmax == cfg.SIG_FMAX:  
        
        low = fmin / nyquist
        b, a = butter(order, low, btype="high")
        sig = lfilter(b, a, sig)

    # Lowpass?
    elif fmin == cfg.SIG_FMIN and fmax < cfg.SIG_FMAX:

        high = fmax / nyquist
        b, a = butter(order, high, btype="low")
        sig = lfilter(b, a, sig)

    # Bandpass?
    elif fmin > cfg.SIG_FMIN and fmax < cfg.SIG_FMAX:

        low = fmin / nyquist
        high = fmax / nyquist
        b, a = butter(order, [low, high], btype="band")
        sig = lfilter(b, a, sig)

    return sig.astype("float32")

# Raven is using Kaiser window FIR filter, so we try to emulate it.
# Raven uses the Window method for FIR filter design. 
# A Kaiser window is used with a default transition bandwidth of 0.02 times
# the Nyquist frequency and a default stop band attenuation of 100 dB. 
# For a complete description of this method, see Discrete-Time Signal Processing 
# (Second Edition), by Alan Oppenheim, Ronald Schafer, and John Buck, Prentice Hall 1998, pp. 474-476.
def bandpassKaiserFIR(sig, rate, fmin, fmax, width=0.02, stopband_attenuation_db=100):

    # Check if we have to bandpass at all
    if fmin == cfg.SIG_FMIN and fmax == cfg.SIG_FMAX or fmin > fmax:
        return sig

    from scipy.signal import kaiserord, firwin, lfilter
    nyquist = 0.5 * rate

    # Calculate the order and Kaiser parameter for the desired specifications.
    N, beta = kaiserord(stopband_attenuation_db, width)

    # Highpass?
    if fmin > cfg.SIG_FMIN and fmax == cfg.SIG_FMAX: 
        low = fmin / nyquist
        taps = firwin(N, low, window=('kaiser', beta), pass_zero=False)

    # Lowpass?
    elif fmin == cfg.SIG_FMIN and fmax < cfg.SIG_FMAX:
        high = fmax / nyquist
        taps = firwin(N, high, window=('kaiser', beta), pass_zero=True)

    # Bandpass?
    elif fmin > cfg.SIG_FMIN and fmax < cfg.SIG_FMAX:
        low = fmin / nyquist
        high = fmax / nyquist
        taps = firwin(N, [low, high], window=('kaiser', beta), pass_zero=False)

    # Apply the filter to the signal.
    sig = lfilter(taps, 1.0, sig)

    return sig.astype("float32")


