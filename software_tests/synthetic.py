"""These tests confirm that the synthetic domain is still working as expected"""

import dataset_models.synthetic.synthetic as synthetic

def test_synthetic_downloaded():
    s = synthetic.Synthetic()
    s.download_dataset()
    assert s.is_downloaded()

def test_synthetic_initialization():
    s = synthetic.Synthetic()
    s.get_dimensions()
    s.get_validation_step_count()

def test_synthetic_accessing():
    s = synthetic.Synthetic()
    s.get_validation_data()
    s.get_validation_generator()
    s.training_generator()

def test_network_evaluation():
    # todo: this should fail at this point
    s = synthetic.Synthetic()
    s.evaluate_network("network_model_path")