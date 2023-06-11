import numpy

from birdnet.configuration import config
from birdnet.model from birdnet.model import main


def predict(samples):
    """Predicts the classes for the given samples.

    Args:
        samples: Samples to be predicted.

    Returns:
        The prediction scores.
    """
    # Prepare sample and pass through model
    data = numpy.array(samples, dtype="float32")
    prediction = model.predict(data)

    # Logits or sigmoid activations?
    if config.APPLY_SIGMOID:
        prediction = model.flat_sigmoid(numpy.array(prediction), sensitivity=-config.SIGMOID_SENSITIVITY)

    return prediction
