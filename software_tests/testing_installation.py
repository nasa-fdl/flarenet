"""
These tests check whether the basic requirements of the
solar deep learning library have been satisfied.
This is not a comprehensive test suite.
"""

import pytest
from tools.test_utils import keras_test # when using the backend, you apply the "@keras_test" decorator to reset the backend on every test

def test_imports():
    """
    Test whether you are able to successfully import all the packaged software
    """
    from keras.layers import Input, Dense, Conv2D, MaxPooling2D, Dropout, Flatten, Activation
    from keras.models import Model
    from keras.callbacks import ModelCheckpoint
    from keras import backend as K
    import numpy as np
    from keras.callbacks import TensorBoard
    import os
    import random
    from datetime import timedelta, datetime
    import argparse
    import sys
    import argparse

if __name__ == '__main__':
    pytest.main([__file__])
