"""Module containing audio helper functions.
"""
import numpy

from birdnet.audio.noise_creation import create_noise

from birdnet.configuration import config

RANDOM = numpy.random.RandomState(config.RANDOM_SEED)
