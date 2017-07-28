import unittest

class TestForSmoke(unittest.TestCase):
    """
    These tests check whether the basic requirements of the
    solar deep learning library have been satisfied.
    This is not a comprehensive test suite.
    """

    def test_imports(self):
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
        import psutil
        import argparse

    def test_importation_of_deep_sun_packages(self):
        try:
            from network_models.training_callbacks import TrainingCallbacks
            from dataset_models.sdo.aia import aia
        except Exception:
            print("""
            You were unable to import all the packages included in this directory.
            You may need to add the project to your path:
            > export PYTHONPATH=INSERT_PATH_TO_DIRECTORY_HERE/solar-forecast/:$PYTHONPATH
            """)
            assert False

    def test_presence_of_aia_data(self):
        from network_models.training_callbacks import TrainingCallbacks
        from dataset_models.sdo.aia import aia
        aia = aia.AIA(32)
        self.assertTrue(aia.is_downloaded())

if __name__ == '__main__':
    unittest.main()
