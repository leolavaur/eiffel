"""Eiffel utilities."""

import tensorflow as tf
from tensorflow import keras


def set_seed(seed: int) -> None:
    """Set the random seed.

    The seed is set for NumPy, Python's random module, and TensorFlow.

    Parameters
    ----------
    seed : int
        The seed to use for random number generation.
    """
    assert isinstance(seed, int), "Seed must be an integer."

    tf.random.set_seed(seed)
    keras.utils.set_random_seed(seed)
    tf.config.experimental.enable_op_determinism()
