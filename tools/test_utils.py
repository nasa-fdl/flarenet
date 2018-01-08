"""Utilities related to Keras unit tests."""
import numpy as np
from numpy.testing import assert_allclose
import six

from keras import backend as K

def keras_test(func):
    """Function wrapper to clean up after TensorFlow tests.

    # Arguments
        func: test function to clean up after.

    # Returns
        A function wrapping the input function.
    """
    @six.wraps(func)
    def wrapper(*args, **kwargs):
        output = func(*args, **kwargs)
        if K.backend() == 'tensorflow':
            K.clear_session()
        return output
    return wrapper
