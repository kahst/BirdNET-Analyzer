import numpy

from birdnet.configuration import config
from birdnet.model.main import flat_sigmoid
from birdnet.model.main import predict_by_sample


def predict_classes(samples):
    """Predicts the classes for the given samples.

    Args:
        samples: Samples to be predicted.

    Returns:
        The prediction scores.
    """
    # Prepare sample and pass through model
    data = numpy.array(samples, dtype="float32")
    prediction = predict_by_sample(data)

    # Logits or sigmoid activations?
    if config.APPLY_SIGMOID:
        prediction = flat_sigmoid(
            numpy.array(prediction),
            sensitivity=-config.SIGMOID_SENSITIVITY
        )

    return prediction
