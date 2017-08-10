import numpy as np
import random

class Synthetic:
    """
    A synthetic data generator. This class is meant to check for numerical and other
    issues that may confuse training. These synthetic data produce y values equal
    to the sum of the values in the cells. The values in the cells are all small
    to force precision issues.
    """

    def __init__(self, samples_per_step=32, input_width=2, input_height=2, input_channels=2, whiten=True):
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

        #  Dictionary caching indices to their normalized in-memory result
        self.training_data = []
        self.validation_data = []

        self.samples_per_step = samples_per_step

        self.input_width = input_width
        self.input_height = input_height
        self.input_channels = input_channels
        self.whiten = whiten

        self.training_set_size = 1000
        self.validation_set_size = 100
        self.training_indices = range(0, self.training_set_size)
        self.validation_indices = range(0, self.validation_set_size)
        self.total_y_per_object = 4.0e-6
        self.construct_data()

    def construct_data(self):
        """
        Pre-generate a set of synthetic data.
        """
        def generate_sample(index):
            """
            Allocate 4.0e-6 total float value randomly among the
            image.
            """
            random.seed(index)
            data = np.zeros((self.input_width, self.input_height, self.input_channels))
            per_add = (self.total_y_per_object)/float(self.input_width * self.input_height)
            for index in range(0, self.input_width * self.input_height):
                x = random.randint(0, self.input_width - 1)
                y = random.randint(0, self.input_height - 1)
                z = random.randint(0, self.input_channels - 1)
                data[x][y][z] += per_add
            if self.whiten:
                data = data - (self.total_y_per_object / (self.input_width * self.input_height * self.input_channels))
            return data
        for index in range(0, self.training_set_size):
            self.training_data.append([generate_sample(index), 4.0e-6])
        for index in range(0, self.validation_set_size):
            self.validation_data.append([generate_sample(index + self.training_set_size), 4.0e-6])

    def get_dimensions(self):
        return (self.input_width, self.input_height, self.input_channels)

    def get_y(self, index, training=True):
        """
        Get the true forecast result for the current filename.
        """
        if training:
            return self.training_data[index][1]
        else:
            return self.validation_data[index][1]

    def generator(self, training=True):
        """
        Generate samples
        """
        index_list = self.validation_indices
        data = self.validation_data
        if training:
            index_list = self.training_indices
            data = self.training_data
        data_x = []
        data_y = []
        i = 0
        while 1:
            data_x_sample = data[index_list[i]][0]
            data_y_sample = data[index_list[i]][1]
            data_x.append(data_x_sample)
            data_y.append(data_y_sample)

            i += 1

            if i == len(index_list):
                i = 0
                random.shuffle(index_list)

            if self.samples_per_step == len(data_x):
                ret_x = np.reshape(data_x, (len(data_x), self.input_width, self.input_height, self.input_channels))
                ret_y = np.reshape(data_y, (len(data_y)))
                yield (ret_x, ret_y)
                data_x = []
                data_y = []
