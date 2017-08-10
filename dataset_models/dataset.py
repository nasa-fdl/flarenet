import yaml

class Dataset(object):
    """
    An abstract class defining the interface of all models to the data.
    """

    def __init__(self, samples_per_step=32):

        # Load the configuration file indicating where the files are stored,
        # then load the names of the data files
        with open("config.yml", "r") as config_file:
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

    def training_generator(self):
        """
        Generate samples for training by selecting a random subsample of
        files located in the training directory. The training data will
        then be collected with the additional timesteps of images
        and side channel information.
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
