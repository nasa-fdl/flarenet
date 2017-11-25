from dataset_models.synthetic import synthetic
from network_models import experiment
from keras.optimizers import adam
from keras.layers import Input, Dense, Conv2D, Dropout, Flatten
from keras.models import Model
import yaml
import os

#####################################
#        INITIALIZING DATA          #
#####################################

print "initializing data"

# Load the configuration file and assign paths
file_path = os.path.dirname(os.path.realpath(__file__))
output_path = file_path + "/trained_models/"
with open(file_path + "/config.yml", "r") as config_file:
    config = yaml.load(config_file)

dataset_model = synthetic.Synthetic(samples_per_step=config["samples_per_step"], input_width=32,
                 input_height=32, training_set_size=100000, validation_set_size=1000,
                 scale_signal_pixels=1.0, scale_noise_pixels=1.0,
                 scale_dependent_variable=1.0,
                 dependent_variable_additive_noise_variance=0.0,
                 noise_channel_count=4, signal_channel_count=4,
                 active_regions=False)

#####################################
#         SPECIFYING DATA           #
#####################################

# Get the input dimensions from the dataset_model
input_width, input_height, input_channels = dataset_model.get_dimensions()
image_shape = (input_width, input_height, input_channels)
input_image = Input(shape=image_shape)

#####################################
#        Network Specification      #
#####################################

x = input_image
x = Conv2D(1, (1,1), strides=(1,1), padding='same', activation="relu")(x)
x = Flatten()(x)
x = Dropout(.5)(x)
x = Dense(2, activation="relu")(x)
prediction = Dense(1, activation="linear")(x)
network_model = Model(inputs=input_image, outputs=prediction)

#####################################
#       Optimization Options        #
#####################################

adam = adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0, clipnorm=1.0)
network_model.compile(optimizer=adam, loss=config["loss"])

#####################################
#        Run the Experiment         #
#####################################

experiment.experiment(network_model, output_path, dataset_model=dataset_model, args=None, config=config)
