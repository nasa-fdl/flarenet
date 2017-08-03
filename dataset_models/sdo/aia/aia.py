import yaml
import os
import numpy as np
from datetime import timedelta, datetime
import psutil
import random
import math
from keras.models import load_model


class AIA:
    """
    A class for managing the download
    and interface of the AIA data.
    """

    def __init__(self, samples_per_step=32, dependent_variable="flux delta"):
        """
        Get a directory listing of the AIA data and load all the filenames
        into memory. We will loop over these filenames while training or
        evaluating the network.
        @param dependent_variable {enum} The valid values for this
        enumerated type are 'flux delta', which indicates we are concerned
        with predicting the change in x-ray flux through time, or
        'forecast' which is concerned with predicting the total x-ray flux
        output at the next time step.
        """

        #  Dictionary caching filenames to their normalized in-memory result
        self.cache = {}

        self.samples_per_step = samples_per_step  # Batch size
        self.dependent_variable = dependent_variable # Target forecast

        # Dimensions
        self.input_width = 1024
        self.input_height = 1024
        self.input_channels = 8

        # Standardize the random number generator to consistent shuffles
        random.seed(0)

        # Load the configuration file indicating where the files are stored,
        # then load the names of the data files
        with open("config.yml", "r") as config_file:
            self.config = yaml.load(config_file)
        assert(self.is_downloaded())
        self.train_files = os.listdir(self.config["aia_path"] + "training")
        self.validation_files = os.listdir(self.config["aia_path"] + "validation")
        self.validation_directory = self.config["aia_path"] + "validation/"
        self.training_directory = self.config["aia_path"] + "training/"

        # Load the y variables into memory
        self.minimum_y = float("Inf")
        self.maximum_y = float("-Inf")
        self.y_dict = {}
        with open(self.config["aia_path"] + "y/Y_GOES_XRAY_201401.csv", "rb") as f:
            for line in f:
                split_y = line.split(",")
                cur_y = float(split_y[1])
                self.y_dict[split_y[0]] = cur_y
                self.minimum_y = min(self.minimum_y, cur_y)
                self.maximum_y = max(self.maximum_y, cur_y)
        self.y_spread = self.maximum_y - self.minimum_y
        self.clean_data()

    def get_dimensions(self):
        """
        Helper function returning the dimensions of the inputs.
        """
        return (self.input_width, self.input_height, self.input_channels)

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
        if not os.path.isfile(self.config["aia_path"] + "y/Y_GOES_XRAY_201401.csv"):
            print("WARNING: you have no results dataset")
            print("place these data into " + self.config["aia_path"] + "y")
            return False
        if not os.path.isfile(self.config["aia_path"] + "training/20140121_1400_AIA_08_1024_1024.dat"):
            print("WARNING: you have no independent variable training dataset")
            print("place these data into " + self.config["aia_path"] + "training")
            return False
        if not os.path.isfile(self.config["aia_path"] + "validation/20140120_1524_AIA_08_1024_1024.dat"):
            print("WARNING: you have no independent variable validation dataset")
            print("place these data into " + self.config["aia_path"] + "validation")
            return False
        return True

    def get_flux_delta(self, filename):
        """
        Return the change in the flux value from the last time step to this one.
        """
        split_filename = filename.split("_")
        k = split_filename[0] + "_" + split_filename[1]
        future = self.y_dict[k]
        current = self.get_prior_y(filename)
        return math.log(future - current + self.y_spread + 1)

    def get_flux(self, filename):
        """
        Return the flux value for the current time step.
        """
        split_filename = filename.split("_")
        k = split_filename[0] + "_" + split_filename[1]
        future = self.y_dict[k]
        return math.log(future + self.y_spread + 1)

    def get_y(self, filename):
        """
        Get the true forecast result for the current filename.
        """
        if self.dependent_variable == "flux delta":
            return self.get_flux_delta(filename)
        elif self.dependent_variable == "forecast":
            return self.get_flux(filename)
        else:
            assert False # There are currently no other valid dependent variables
            return None

    def get_prior_y(self, filename):
        """
        Get the y value for the prior time step. This will
        generally be used so we can capture the delta in the
        prediction value.
        """
        f = filename.split("_")
        datetime_format = '%Y%m%d_%H%M'
        datetime_object = datetime.strptime(f[0]+"_"+f[1], datetime_format)
        td = timedelta(minutes=-12)
        prior_datetime_object = datetime_object + td
        prior_datetime_string = datetime.strftime(prior_datetime_object, datetime_format)
        return self.y_dict[prior_datetime_string]

    def clean_data(self):
        """
        Remove all samples that lack the corresponding y value.
        """
        starting_training_count = len(self.train_files)
        starting_validation_count = len(self.validation_files)
        def filter_files(filename):
            try:
                self.get_y(filename)
            except (KeyError, ValueError) as e:
                return False
            return True
        self.train_files = filter(filter_files, self.train_files)
        self.validation_files = filter(filter_files, self.validation_files)
        print "Training " + str(starting_training_count) + "-> " + str(len(self.train_files))
        print "Validation " + str(starting_validation_count) + "-> " + str(len(self.validation_files))


    def generator(self, training=True):
        """
        Generate samples
        """
        if training:
            files = self.train_files
            directory = self.training_directory
        else:
            files = self.validation_files
            directory = self.validation_directory

        x_mean_vector = [2.2832, 10.6801, 226.4312, 332.5245, 174.1384, 27.1904, 4.7161, 67.1239]
        x_standard_deviation_vector = [12.3858, 26.1799, 321.5300, 475.9188, 289.4842, 42.3820, 10.3813, 72.7348]

        data_x = []
        data_y = []
        i = 0
        while 1:
            f = files[i]
            shape = (self.input_width*self.input_height, self.input_channels)
            data_x_sample = np.load(directory + f)
            data_x_sample = ((data_x_sample.astype('float32').reshape(shape) - x_mean_vector) / x_standard_deviation_vector).reshape(shape) # Standardize to [-1,1]
            data_y_sample = self.get_y(f)

            if not data_y_sample:
                assert False
            data_x.append(data_x_sample)
            data_y.append(data_y_sample)

            i += 1

            if i == len(files):
                i = 0
                random.shuffle(files)

            if self.samples_per_step == len(data_x):
                ret_x = np.reshape(data_x, (len(data_x), self.input_width, self.input_height, self.input_channels))
                ret_y = np.reshape(data_y, (len(data_y)))
                yield (ret_x, ret_y)
                data_x = []
                data_y = []


    def evaluate_network(self, network_model_path):
        """
        Generate a CSV file with the true and the predicted values for
        x-ray flux.
        """
        model = load_model(network_model_path)

        # Load each of the x values and predict the y values with the best performing network
        x_predictions = {}
        for filename in self.train_files:
            data_x_sample = np.load(self.training_directory + filename)
            prediction = model.predict(
                data_x_sample.reshape(1, self.input_width, self.input_height, self.input_channels), verbose=0)
            x_predictions[filename] = [prediction, self.get_flux_delta(filename), self.get_flux(filename), self.get_prior_y(filename)]
        for filename in self.validation_files:
            data_x_sample = np.load(self.validation_directory + filename)
            prediction = model.predict(
                data_x_sample.reshape(1, self.input_width, self.input_height, self.input_channels), verbose=0)
            x_predictions[filename] = [prediction, self.get_flux_delta(filename), self.get_flux(filename), self.get_prior_y(filename)]

        with open(network_model_path + ".performance", "w") as out:
            out.write("datetime, prediction, true y delta, true y, true prior y\n")
            keys = list(x_predictions)
            keys = sorted(keys)
            for key in keys:
                cur = x_predictions[key]
                out.write(key + "," + str(cur[0][0][0]) + "," + str(cur[1]) + "," + str(cur[2]) + "," + str(cur[3]) + "\n")
