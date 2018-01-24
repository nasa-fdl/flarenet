import yaml
import os.path

class Dataset(object):
    """
    An abstract class defining the interface of all models to the data.
    """

    def __init__(self, samples_per_step=32):

        # Open the root config file
        abspath = os.path.abspath(__file__)
        head, tail = os.path.split(abspath)
        with open(head + "/../config.yml", "r") as config_file:
            self.config = yaml.load(config_file)

        self.samples_per_step = samples_per_step  # Batch size

    def get_dimensions(self):
        """
        Helper function returning the dimensions of the inputs.
        """
        return (self.input_width, self.input_height, self.input_channels)

    def get_validation_step_count(self):
        """
        Return the current count of valid validation samples. The number changes based
        on when data is available and other factors.
        """
        raise NotImplementedError

    def get_validation_data(self):
        """
        Load samples for validation dataset. This will load the entire validation dataset
        into memory. If you have a very large validation dataset you should likely
        refactor this to be a data generator that will stage the data into memory incrementally.
        """
        raise NotImplementedError

    def get_validation_generator(self):
        """
        Iterate through the entire validation dataset, then loop back to the first validation
        sample. You should randomize the order of the validation set within the initialization
        of the dataset_model.
        """
        raise NotImplementedError

    def get_training_generator(self):
        """
        Generate samples for training by selecting a random subsample of
        files located in the training directory. The training data will
        then be collected with the additional timesteps of images
        and side channel information.
        """
        raise NotImplementedError

    def get_training_generator_multiprocess(self):
        """
        Return data for training in a threadsafe manner. To
        instantiate this method you should generally
        utilize the Keras Sequence class.
        """
        raise NotImplementedError

    def get_validation_generator_multiprocess(self):
        """
        Return data for validation in a threadsafe manner. To
        instantiate this method you should generally
        utilize the Keras Sequence class.
        """
        raise NotImplementedError

    def evaluate_network(self, network_model_path):
        """
        Generate a CSV file with the true and the predicted values for the network.
        """
        raise NotImplementedError

    def download_dataset(self):
        """
        Download the datasets expected by this data adapter to the directory
        specified by the config.yml file.
        """
        raise NotImplementedError

    def is_downloaded(self):
        """
        Test whether the dataset is currently present.
        """
        raise NotImplementedError
