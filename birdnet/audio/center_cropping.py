import numpy

from birdnet.audio.noise_creation import create_noise


def crop_center(sig, rate, seconds):
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
        sig = numpy.hstack(
            (
                sig, create_noise(sig, (int(seconds * rate) - len(sig)), 0.5)
             )
        )

    return sig
