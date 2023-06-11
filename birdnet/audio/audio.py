"""Module containing audio helper functions.
"""
import numpy

from birdnet.configuration import config

RANDOM = numpy.random.RandomState(config.RANDOM_SEED)
