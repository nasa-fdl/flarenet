import os
import numpy as np
from datetime import timedelta, datetime
import random
import math
import dataset_models.dataset
from operator import truediv as div
from operator import sub
import feather
import scipy.ndimage as sp

class AIA(dataset_models.dataset.Dataset):
    """
    A class for managing the download
    and interface of the AIA data.
    """

    def __init__(self,
                 samples_per_step=32,
                 dependent_variable="flux delta",
                 lag="01hr",
                 catch="24hr",
                 aia_image_count=2,
                 side_channels=["", "true_value", "current_goes", "hand_tailored"]):
        """
        Get a directory listing of the AIA data and load all the filenames
        into memory. We will loop over these filenames while training or
        evaluating the network.
        @param dependent_variable {enum} The valid values for this
        enumerated type are 'flux delta', which indicates we are concerned
        with predicting the change in x-ray flux through time, or
        'forecast' which is concerned with predicting the total x-ray flux
        output at the next time step.
        @param lag {str} the amount of time lag until we start making forecasts. 
        "00min","12min","24min","36min","01hr","24hr"
        @param catch {str} the time over which we find the maximum x-ray flux value.
        "12min","24min","36min","01hr","24hr"
        @param aia_image_count {int} The number of images from AIA to concatenate together to
        form a single image. The minimum value is currently 1 and the maximum value is defined
        by your hardware.
        @param side_channels {list[str]} A list of the side channels to include in
        the independent variable list. These currently include the currently measured GOES
        reading for x-ray flux, and a complete set of hand tailored features applied in earlier
        works.
        """
        super(AIA, self).__init__()
        
        self.samples_per_step = samples_per_step  # Batch size
        self.dependent_variable = dependent_variable # Target forecast

        self._initialize_side_channels(side_channels)
        self.y_filepath = self.config["aia_path"] + "y/All_Ys_" + lag + "Delay_" + catch + "Max.csv"

        # Dimensions
        self.input_width = 1024
        self.input_height = 1024
        self.input_channels = 8

        # Standardize the random number generator to consistent shuffles
        random.seed(0)

        # Ensure the dataset is downloaded
        assert(self.is_downloaded())
        self.training_directory = self.config["aia_path"] + "training/"
        self.validation_directory = self.config["aia_path"] + "validation/"
        self.train_files = os.listdir(self.training_directory)
        self.validation_files = os.listdir(self.validation_directory)

        # The number of image timesteps to include as the independent variable
        self.aia_image_count = aia_image_count

        # Load the y variables into memory
        self.y_dict = {}
        self.y_prior_dict = {}
        with open(self.y_filepath, "r") as f:
            f.readline()
            for line in f:
                split_y = line.split(",")
                #cur_y = float(split_y[1])
                cur_y = float(split_y[3])
                self.y_dict[split_y[0]] = cur_y*1e6
                self.y_prior_dict[split_y[0]] = float(split_y[2])
        self._clean_data()

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
        return len(self.validation_files)

    def get_validation_generator(self):
        """
        Load samples for validation dataset. This will load the entire validation dataset
        into memory. If you have a very large validation dataset you should likely
        refactor this to be a data generator that will stage the data into memory incrementally.
        """
        while True:
            data_y = []
            data_x = []
            for f in self.validation_files:
                sample = self._get_x_data(f, aia_image_count=self.aia_image_count, training=False)
                self._sample_append(data_x, sample)
                data_y.append(self._get_y(f))
            yield self._finalize_dataset(data_x, data_y)

    def get_training_generator(self):
        """
        Generate samples for training by selecting a random subsample of
        files located in the training directory. The training data will
        then be collected with the additional timesteps of images
        and side channel information.
        """
        files = self.train_files
        directory = self.training_directory
        data_y = []
        data_x = []
        i = 0
        while 1:
            f = files[i]
            i += 1
            sample = self._get_x_data(f, aia_image_count=self.aia_image_count, training=True)
            self._sample_append(data_x, sample)
            data_y.append(self._get_y(f))

            if i == len(files):
                i = 0
                random.shuffle(files)

            if self.samples_per_step == len(data_x[0]):
                yield self._finalize_dataset(data_x, data_y)
                data_x = []
                data_y = []

    def get_network_model(self, network_model_path):
        """Load a network model from file.
        @param network_model_path {string} The file path to the network model.
        """
        from keras.models import load_model
        from dataset_models.sdo.layers import LogWhiten
        custom_objects = {"LogWhiten": LogWhiten}
        model = load_model(network_model_path,
                           custom_objects=custom_objects)
        return model

    def examine_weights(self, network_model_path):
        """
        Print the weights of the network.
        """
        model = self.get_network_model(network_model_path)
        for layer in model.layers:
            weights = layer.get_weights() # list of numpy arrays
            print(weights)

    def evaluate_network(self, network_model_path):
        """
        Generate a CSV file with the true and the predicted values for
        x-ray flux.
        """
        model = self.get_network_model(network_model_path)
        def save_performance(file_names, outfile_path, training=None):
            """
            Evaluate the files with the model and output them
            @param files {list[string]}
            @param outfile_path {string}
            """
            x_predictions = {}
            for filename in file_names:
                data_x = []
                data_y = []
                sample = self._get_x_data(filename, aia_image_count=self.aia_image_count, training=training)
                self._sample_append(data_x, sample)
                data_y.append(-999999999999.0)
                prediction = model.predict(self._finalize_dataset(data_x, data_y)[0], verbose=0)
                x_predictions[filename] = [prediction, self._get_flux_delta(filename), self._get_flux(filename), self._get_prior_y(filename)]

            with open(outfile_path, "w") as out:
                out.write("datetime, prediction, true y delta, true y, true prior y\n")
                keys = list(x_predictions)
                keys = sorted(keys)
                for key in keys:
                    cur = x_predictions[key]
                    out.write(key + "," + str(cur[0][0][0]) + "," + str(cur[1]) + "," + str(cur[2]) + "," + str(cur[3]) + "\n")

        save_performance(self.train_files[0::100], network_model_path + ".training.performance", training=True)
        save_performance(self.validation_files, network_model_path + ".validation.performance", training=False)
        print("#########")
        print("performance data has been saved to the following locations")
        print(network_model_path + ".training.performance")
        print(network_model_path + ".validation.performance")
        print("#########")

    def download_dataset(self):
        """
        Download the datasets expected by this data adapter to the directory
        specified by the config.yml file.
        """
        raise NotImplementedError

    def is_downloaded(self):
        """
        Determine whether the AIA dataset has been downloaded.
        """
        if not os.path.isdir(self.config["aia_path"]):
            print("WARNING: the data directory specified in config.yml does not exist")
            return False
        if not os.path.isdir(self.config["aia_path"] + "validation"):
            print("WARNING: you have no validation folder")
            print("place these data into " + self.config["aia_path"] + "validation")
            return False
        if not os.path.isdir(self.config["aia_path"] + "training"):
            print("WARNING: you have no training folder")
            print("place these data into " + self.config["aia_path"] + "training")
            return False
        if not os.path.isdir(self.config["aia_path"] + "y"):
            print("WARNING: you have no dependent variable folder")
            print("place these data into " + self.config["aia_path"] + "y")
            return False

        return True # todo: update these files

        if not os.path.isfile(self.config["aia_path"] + "y/Y_GOES_XRAY_201401_201406_00minDELAY_01hrMAX.csv"):
            print("WARNING: you have no results datasets")
            print("place these data into " + self.config["aia_path"] + "y")
            return False
        if not os.path.isfile(self.config["aia_path"] + "training/AIA20140617_2136_08chnls.dat"):
            print("WARNING: you have no independent variable training dataset")
            print("place these data into " + self.config["aia_path"] + "training")
            return False
        if not os.path.isfile(self.config["aia_path"] + "validation/AIA20140308_1400_08chnls.dat"):
            print("WARNING: you have no independent variable validation dataset")
            print("place these data into " + self.config["aia_path"] + "validation")
            return False
        return True

    def get_side_channel_length(self):
        """
        Get the length of the side channel information.
        """
        length = 0
        if "hand_tailored" in self.side_channels:
            length += 25
        if "true_value" in self.side_channels:
            length += 1
        if "current_goes" in self.side_channels:
            length += 1
        return length

    def _initialize_side_channels(self, side_channels):
        """
        Setup the side channels for the network.
        """
        self.side_channels = side_channels
        self.side_channel_filepath = self.config["aia_path"] + "side_channel/HMI_features.csv"
        self.side_channel_dict = {}
        if "hand_tailored" not in self.side_channels:
            return
        with open(self.side_channel_filepath, "rb") as f:
            side_channel_means = [
                3.84E+03,
                7.96E+23,
                8.13E+13,
                4.70E+02,
                2.12E+13,
                5.41E+22,
                2.35E+03,
                2.44E+04,
                1.12E+01,
                9.24E+01,
                1.91E+02,
                2.43E+02,
                7.88E+02,
                7.96E+02,
                3.59E+02,
                2.25E-02,
                -1.35E+00,
                1.82E-02,
                7.56E+10,
                -7.33E+03,
                7.15E+02,
                -3.17E+02,
                -1.18E+00,
                -3.34E-02,
                3.67E-02
            ]
            side_channel_std_dev = [
                3.11E+03,
                8.75E+23,
                6.26E+13,
                6.04E+02,
                3.21E+13,
                4.10E+22,
                1.71E+03,
                1.28E+04,
                6.12E+00,
                4.95E+01,
                8.99E+01,
                1.16E+02,
                4.35E+02,
                4.40E+02,
                1.83E+02,
                1.18E+00,
                1.87E+02,
                1.21E-01,
                5.83E+10,
                5.93E+03,
                3.34E+03,
                6.01E+03,
                7.55E-01,
                3.91E-01,
                4.33E-01
            ]
            def clean(elem):
                if math.isinf(elem) or math.isnan(elem):
                    return 0.0
                else:
                    return elem
            for line in f:
                split_sc = line.split(",")
                split_sc[1:] = map(float, split_sc[1:])
                split_sc[1:] = map(sub, split_sc[1:], side_channel_means)
                split_sc[1:] = map(div, split_sc[1:], side_channel_std_dev)
                split_sc[1:] = map(clean, split_sc[1:])
                self.side_channel_dict[split_sc[0][:8]] = split_sc[1:]

    def _finalize_dataset(self, data_x, data_y):
        """
        Reshape the dataset to be appropriate for training and validation.
        """
        for index in range(0, self.aia_image_count):
            data_x[index] = np.reshape(data_x[index], (len(data_x[index]), self.input_width, self.input_height, self.input_channels)).astype('float32')

        assert len(self.side_channels) < 2, "You need to fix this for arbitrary side channel selection"
        if "current_goes" in self.side_channels:
            data_x[-1] = np.reshape(data_x[-1], (len(data_x[-1]), 1)).astype('float32')
        if "true_value" in self.side_channels:
            data_x[-1] = np.reshape(data_x[-1], (len(data_x[-1]), 1)).astype('float32')
        if "hand_tailored" in self.side_channels:
            data_x[-1] = np.reshape(data_x[-1], (len(data_x[-1]), len(data_x[-1][0]))).astype('float32')
        ret_y = np.reshape(data_y, (len(data_y)))
        return (data_x, ret_y)

    def _sample_append(self, data_x, sample):
        """
        Append a sample to the current dataset.
        """
        if not data_x:
            for _ in sample:
                data_x.append([])
        for idx, part in enumerate(sample):
            data_x[idx].append(sample[idx])
        return

    def _get_flux_delta(self, filename):
        """
        Return the change in the flux value from the last time step to this one.
        """
        k = filename[9:22]
        future = self.y_dict[k]
        current = self._get_prior_y(filename)
        delta = future - current
        return delta

    def _get_flux(self, filename):
        """
        Return the flux value for the current time step.
        """
        length = len(filename)
        assert(length == 44)
        k = filename[9:22]
        future = self.y_dict[k]
        return future

    def _get_y(self, filename):
        """
        Get the true forecast result for the current filename.
        """
        if self.dependent_variable == "flux delta":
            return self._get_flux_delta(filename)
        elif self.dependent_variable == "forecast":
            return self._get_flux(filename)
        else:
            assert False # There are currently no other valid dependent variables
            return None

    def _get_prior_timestep_string(self, filename, timestep):
        """
        Get the filename of the previous timestep
        """
        datetime_format = '%Y%m%d_%H%M'
        datetime_object = datetime.strptime(filename[9:22], datetime_format)
        td = timedelta(minutes=(-12*timestep))
        prior_datetime_object = datetime_object + td
        prior_datetime_string = datetime.strftime(prior_datetime_object, datetime_format)
        return prior_datetime_string

    def _get_prior_x_filename(self, filename, timestep):
        """Get the name of the file preceding the filename
        by timestep files.
        """
        identifier = self._get_prior_timestep_string(filename, timestep)
        if filename[:5] == "noflr":
            suffix = "_8chnls_1024_0" + str(12 * timestep) + "h.fthr"
        else:
            suffix = "_8chnls_1024_0" + str(12 + 12 * timestep) + "h.fthr"
        return filename[0:9] + identifier + suffix

    def _get_prior_y(self, filename):
        """
        Get the y value for the prior time step. This will
        generally be used so we can capture the delta in the
        prediction value. We also feed it into the neural network
        as side information.
        """
        k = filename[9:22]
        return self.y_prior_dict[k]

    def _clean_data(self):
        """
        Remove all samples that lack the required y value.
        """
        starting_training_count = len(self.train_files)
        starting_validation_count = len(self.validation_files)
        # todo: this closure no longer does anything
        def filter_closure(training):
            def filter_files(filename):
                try:
                    if self.aia_image_count > 1:
                        for i in range(0, self.aia_image_count):
                            prior_x_file = self._get_prior_x_filename(filename, i)
                        if prior_x_file not in self.train_files and prior_x_file not in self.validation_files:
                            return False
                    self._get_y(filename)
                    if len(self.side_channels) > 0:
                        self._get_side_channel_data(filename)
                except (KeyError, ValueError) as e:
                    return False
                return True
            return filter_files
        self.train_files = list(filter(filter_closure(True), self.train_files))
        self.validation_files = list(filter(filter_closure(False), self.validation_files))

        print("Training " + str(starting_training_count) + "-> " + str(len(self.train_files)))
        print("Validation " + str(starting_validation_count) + "-> " + str(len(self.validation_files)))

    def _get_aia_image(self, filename, directory, previous=0):
        """
        Get the requested AIA image, or an image from a previous
        step as indicated by the previous parameter.
        @param filename {str} The name of the current AIA image.
        @param directory {str} The path to the directory where the image will be located.
        @param previous {int} How many steps back we will look for an image.
        Note: it is currently your responsibility to have the clean_data()
        function prevent requests to this function for data that does not
        exist.
        todo: improve this function. It will always be called many times in successesion so it isn't
              necessary to keep finding older files by walking back through the history.
        """
        assert previous >= 0, "previous should be a non-negative integer, it is currently " + previous
        assert previous < 5, "previous should not be a very large integer, it is currently " + previous
        if previous == 0:
            data = feather.read_dataframe(directory + filename).values.reshape(1024,1024,8)
            return data
        else:
            previous_filename = self._get_prior_x_filename(filename, previous)
            data = feather.read_dataframe(directory + previous_filename).values.reshape(1024,1024,8)
            return data
          
    def _get_hand_tailored_side_channel_data(self, filename):
        """
        Get the vector of side channel information that summarizes the magnetogram.
        """
        return self.side_channel_dict[filename[9:17]]

    def _get_side_channel_data(self, filename):
        """
        Get the side channel information defined for the current filename.
        todo: this function has slightly different semantics than the other
              data accessor. It decided what to return based on an instance
              variable instead of a parameter. I should standardize.
        """
        if "true_value" in self.side_channels:
            return np.array([self._get_y(filename)])
        if "current_goes" in self.side_channels:
            return np.array([self._get_prior_y(filename)])
        if "hand_tailored" in self.side_channels:
            return np.array(self._get_hand_tailored_side_channel_data(filename))

    def _get_x_data(self, filename, aia_image_count=2, training=None):
        """
        Get the list of data associated with the sample filename.
        @param filename {string} The name of the file which we are currently sampling.
        @param aia_image_count {int} The total number of timestep images to be composited.
        @param current_data {list} The data that we will append to.
        @param training {bool} Indicates whether we are currently training or testing.
           This will determine where we look for x-files.
        """
        current_data = []
        if training:
            directory = self.training_directory
        else:
            directory = self.validation_directory
        for index in range(0, aia_image_count):
            current_data.append(self._get_aia_image(filename, directory, previous=index))
        if self.side_channels:
            data_x_side_channel_sample = self._get_side_channel_data(filename)
            current_data.append(data_x_side_channel_sample)
        return current_data
