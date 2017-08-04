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

    def __init__(self, samples_per_step=32, dependent_variable="flux delta", lag="00min", catch="24hr"):
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
        """

        # Load the configuration file indicating where the files are stored,
        # then load the names of the data files
        with open("config.yml", "r") as config_file:
            self.config = yaml.load(config_file)
        
        self.samples_per_step = samples_per_step  # Batch size
        self.dependent_variable = dependent_variable # Target forecast

        self.y_filepath = self.config["aia_path"] + "y/Y_GOES_XRAY_201401_201406_" + lag + "DELAY_" + catch + "MAX.csv"
        
        # Dimensions
        self.input_width = 1024
        self.input_height = 1024
        self.input_channels = 8

        # Standardize the random number generator to consistent shuffles
        random.seed(0)

        assert(self.is_downloaded())
        self.training_directory = self.config["aia_path"] + "training/"
        self.validation_directory = self.config["aia_path"] + "validation/"
        self.train_files = os.listdir(self.training_directory)
        self.validation_files = os.listdir(self.validation_directory)

        # Load the y variables into memory
        self.y_dict = {}

        # The number of image timesteps to include as the independent variable
        self.image_count = 2
        
        self.y_prior_dict = {}
        self.y_prior_filepath = self.config["aia_path"] + "y/Y_GOES_XRAY_201401_201406_00minDELAY_12minMAX.csv"
        with open(self.y_prior_filepath, "rb") as f:
            for line in f:
                split_y = line.split(",")
                cur_y = float(split_y[1])
                self.y_prior_dict[split_y[0]] = cur_y

        with open(self.y_filepath, "rb") as f:
            for line in f:
                split_y = line.split(",")
                cur_y = float(split_y[1])
                self.y_dict[split_y[0]] = cur_y
        self.clean_data()

    def get_dimensions(self):
        """
        Helper function returning the dimensions of the inputs.
        """
        return (self.input_width, self.input_height, self.input_channels)

    def get_flux_delta(self, filename):
        """
        Return the change in the flux value from the last time step to this one.
        """
        k = filename[3:11] + filename[11:16]
        future = self.y_dict[k]
        current = self.get_prior_y(filename)
        return future - current

    def get_flux(self, filename):
        """
        Return the flux value for the current time step.
        """
        k = filename[3:11] + filename[11:16]
        future = self.y_dict[k]
        return future

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

    def get_prior_timestep_string(self, filename):
        """
        Get the filename of the previous timestep
        """
        datetime_format = '%Y%m%d_%H%M'
        datetime_object = datetime.strptime(filename[3:11] + filename[11:16], datetime_format)
        td = timedelta(minutes=-12)
        prior_datetime_object = datetime_object + td
        prior_datetime_string = datetime.strftime(prior_datetime_object, datetime_format)
        return prior_datetime_string

    def get_prior_x_filename(self, filename):
        identifier = self.get_prior_timestep_string(filename)
        return "AIA" + identifier + "_08chnls.dat"

    def get_prior_y(self, filename):
        """
        Get the y value for the prior time step. This will
        generally be used so we can capture the delta in the
        prediction value. We also feed it into the neural network
        as side information.
        """
        prior_datetime_string = self.get_prior_timestep_string(filename)
        return self.y_prior_dict[prior_datetime_string]

    def clean_data(self):
        """
        Remove all samples that lack the required y value.
        """
        starting_training_count = len(self.train_files)
        starting_validation_count = len(self.validation_files)
        def filter_closure(training):
            def filter_files(filename):
                try:
                    self.get_prior_y(filename)
                    self.get_y(filename)
                    prior_x_file = self.get_prior_x_filename(filename)
                except (KeyError, ValueError) as e:
                    return False
                if prior_x_file not in self.train_files:
                    return False
                else:
                    return True
            return filter_files
        self.train_files = filter(filter_closure(True), self.train_files)
        self.validation_files = filter(filter_closure(False), self.validation_files)
        print "Training " + str(starting_training_count) + "-> " + str(len(self.train_files))
        print "Validation " + str(starting_validation_count) + "-> " + str(len(self.validation_files))

    def get_centering_tensor(self):
        """
        Get a tensor for centering the data on the GPU.
        """
        ret = []
        x_mean_vector = [
            0.5015,
            3.225,
            111.2,
            170.6,
            57.03,
            7.897,
            1.187,
            21.98
        ]
        return np.array(x_mean_vector).reshape((1,1,1,self.input_channels))

    def get_unit_deviation_tensor(self):
        """
        Get a tensor for changing the data to have unit variance.
        """
        x_standard_deviation_vector = [
            3.593,
            11.11,
            160.0,
            246.3,
            98.08,
            13.45,
            3.238,
            24.78
        ]
        return np.array(x_standard_deviation_vector).reshape((1,1,1,self.input_channels))

    def get_x_data(self, filename, directory, image_count=2, current_data=None):
        """
        Get the list of data associated with the sample filename.
        @param filename {string} The name of the file which we are currently sampling.
        @param directory {string} The location in which we will look for the file.
        @param image_count {int} The total number of timestep images to be composited.
        @param current_data {list} The data that we will append to.
        todo: make this better
        """
        current_data[0].append(np.load(directory + filename))
        if image_count > 1:
            previous_filename = self.get_prior_x_filename(filename)
            current_data[1].append(np.load(self.training_directory + previous_filename))
        data_x_side_channel_sample = np.array([self.get_prior_y(filename)])
        current_data[-1].append(data_x_side_channel_sample)
        return current_data

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
        data_y = []
        data_x = []
        for index in range(0, self.image_count + 1):
            data_x.append([])
        shape = (self.input_width * self.input_height, self.input_channels)
        i = 0
        while 1:
            f = files[i]
            i += 1
            self.get_x_data(f, directory, image_count=self.image_count, current_data=data_x)
            data_y.append(self.get_y(f))

            if i == len(files):
                i = 0
                if training:
                    random.shuffle(files)

            if self.samples_per_step == len(data_x[0]) or not training:
                for index in range(0, len(data_x)-1):
                    data_x[index] = np.reshape(data_x[index], (len(data_x[index]), self.input_width, self.input_height, self.input_channels))
                data_x[-1] = np.reshape(data_x[-1], (len(data_x[-1]), 1))
                ret_y = np.reshape(data_y, (len(data_y)))
                yield (data_x, ret_y)
                data_x = []
                for index in range(0, self.image_count + 1):
                    data_x.append([])
                data_y = []

    def evaluate_network(self, network_model_path):
        """
        Generate a CSV file with the true and the predicted values for
        x-ray flux.
        """

        custom_objects = {"centering_tensor": self.get_centering_tensor(),
                          "scaling_tensor": self.get_unit_deviation_tensor()}
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

    def download_dataset(self):
        """
        Download the datasets expected by this data adapter to the directory
        specified by the config.yml file.
        """
        assert False, "This has not yet been implemented"

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
