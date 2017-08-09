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
        Generate a CSV file with the true and the predicted values for
        x-ray flux.
        """
        raise NotImplementedError
        from keras.models import load_model

        # todo: use the accessors that will have been defined within the subclass
        # to get the data

        custom_objects = {"LogWhiten": LogWhiten}
        model = load_model(network_model_path,
                           custom_objects=custom_objects)

        def save_performance(file_names, file_path, outfile_path):
            """
            Evaluate the files with the model and output them
            @param files {list[string]}
            @param outfile_path {string}
            """

            x_predictions = {}
            for filename in file_names:
                data_x_image_1 = np.load(file_path + filename)
                data_x_image_2 = np.load(self.training_directory + self.get_prior_x_filename(filename))
                prediction = model.predict(
                    [
                        data_x_image_1.reshape(1, self.input_width, self.input_height, self.input_channels),
                        data_x_image_2.reshape(1, self.input_width, self.input_height, self.input_channels),
                        np.array(self.get_prior_y(filename)).reshape(1)], verbose=0)
                x_predictions[filename] = [prediction, self.get_flux_delta(filename), self.get_flux(filename), self.get_prior_y(filename)]

            with open(outfile_path, "w") as out:
                out.write("datetime, prediction, true y delta, true y, true prior y\n")
                keys = list(x_predictions)
                keys = sorted(keys)
                for key in keys:
                    cur = x_predictions[key]
                    out.write(key + "," + str(cur[0][0][0]) + "," + str(cur[1]) + "," + str(cur[2]) + "," + str(cur[3]) + "\n")

        save_performance(self.train_files[0::100], self.training_directory, network_model_path + "training.performance")
        save_performance(self.validation_files, self.validation_directory, network_model_path + "validation.performance")
        print "#########"
        print network_model_path + "training.performance"
        print network_model_path + "validation.performance"
        print "#########"

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
