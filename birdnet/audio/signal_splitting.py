from birdnet.audio.noise_creation import create_noise


def split_signal(sig, rate, seconds, overlap, minlen):
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
        if len(split) < int(minlen * rate):
            break

        # Signal chunk too short?
        if len(split) < int(rate * seconds):
            split = numpy.hstack(
                (
                    split,
                    create_noise(
                        split, (int(rate * seconds) - len(split)), 0.5
                    )
                )
            )

        sig_splits.append(split)

    return sig_splits
