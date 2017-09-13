"""These tests confirm that the synthetic domain is working as expected"""

import dataset_models.synthetic.synthetic as synthetic
import numpy as np

def test_synthetic_downloaded():
    s = synthetic.Synthetic()
    s.download_dataset()
    assert s.is_downloaded()

def test_synthetic_summation():
    """Make sure the dependent variable matches the summation of the layer"""
    s = synthetic.Synthetic(samples_per_step=32, input_width=2,
                 input_height=2, training_set_size=1000, validation_set_size=1000,
                 scale_signal_pixels=1.0, scale_noise_pixels=1.0,
                 scale_dependent_variable=1.0,
                 dependent_variable_additive_noise_variance=0.0,
                 noise_channel_count=0, signal_channel_count=1,
                 active_regions=False)
    generator = s.get_training_generator()
    sample = generator.next()
    total = np.sum(sample[0])
    assert total == 66.255974312950372, "Channel was not as expected: %d" % total

def test_get_dimensions():
    s = synthetic.Synthetic(samples_per_step=32, input_width=400,
                 input_height=400, training_set_size=1000, validation_set_size=1000,
                 scale_signal_pixels=1.0, scale_noise_pixels=1.0,
                 scale_dependent_variable=1.0,
                 dependent_variable_additive_noise_variance=0.0,
                 noise_channel_count=3, signal_channel_count=2,
                 active_regions=False)
    dims = s.get_dimensions()
    assert dims[0] == 400  # width
    assert dims[1] == 400 # height
    assert dims[2] == 5 # channels

def test_get_validation_step_count():
    s = synthetic.Synthetic(samples_per_step=32, input_width=400,
                 input_height=400, training_set_size=1000, validation_set_size=1000,
                 scale_signal_pixels=1.0, scale_noise_pixels=1.0,
                 scale_dependent_variable=1.0,
                 dependent_variable_additive_noise_variance=0.0,
                 noise_channel_count=3, signal_channel_count=2,
                 active_regions=False)
    assert s.get_validation_step_count() == 1000
