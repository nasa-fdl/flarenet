"""This is a starting point for specifying your own architectures for the solar flare prediction problem.
"""

# Neural network specification
from keras.layers import Input, Dense, Conv2D, Conv1D, MaxPooling2D, Dropout, Flatten, Activation, AveragePooling2D, concatenate, Lambda
from keras.models import Model

# Utilities for this script
import os
import sys
import yaml

# Libraries packaged with this repository
from network_models import experiment
import dataset_models.sdo.aia as dataset_model_module
import dataset_models.sdo.layers as custom_layers_module

# Optimizer used in training
from keras.optimizers import adam

#####################################
#        INITIALIZING DATA          #
#####################################

print("initializing data")

# How many images will be composited
aia_image_count = 1

# Change current directory to the network file
abspath = os.path.abspath(__file__)
head, tail = os.path.split(abspath)
os.chdir(head)

# Load the configuration file. You should never change
# the configuration within this file.
with open("config.yml", "r") as config_file:
    config = yaml.load(config_file)

# Uncomment only the side channel you want to include.
side_channels = []
#side_channels = ["current_goes"]
#side_channels = ["hand_tailored"]
#side_channels = ["true_value"]

dataset_model = dataset_model_module.AIA(samples_per_step=config["samples_per_step"],
                        side_channels=side_channels,
                        aia_image_count=aia_image_count,
                        dependent_variable="forecast")

#####################################
#      SPECIFYING INPUT DATA        #
#####################################

input_width, input_height, input_channels = dataset_model.get_dimensions()

image_shape = (input_width, input_height, input_channels)
input_images = []
all_inputs = []
for _ in range(0, aia_image_count):
    image = Input(shape=image_shape)
    input_images.append(image)
    all_inputs.append(image)
side_channel_length = dataset_model.get_side_channel_length()
if side_channel_length > 0:
    input_side_channel = Input(shape=(side_channel_length,), name="Side_Channel")
    all_inputs.append(input_side_channel)

#####################################
#     Constructing Architecture     #
#####################################

print("constructing network in the Keras functional API")

# Center and scale the input data
for idx, input_image in enumerate(input_images):
    input_images[idx] = custom_layers_module.LogWhiten()(input_image)
if len(input_images) > 1:
    x = concatenate(input_images)
else:
    x = input_images[0]
    
x = Conv2D(16, kernel_size=16, padding = 'same', activation='relu', use_bias=False, kernel_initializer = 'lecun_uniform')(x)
x = Conv2D(16, kernel_size=16, padding = 'same', activation='relu', use_bias=False, kernel_initializer = 'lecun_uniform')(x)
x = MaxPooling2D(pool_size=8, padding = 'same')(x)
x = Conv2D(32, kernel_size=8, padding = 'same', activation='relu', use_bias=False, kernel_initializer = 'lecun_uniform')(x)
x = Conv2D(32, kernel_size=8, padding = 'same', activation='relu', use_bias=False, kernel_initializer = 'lecun_uniform')(x)
x = MaxPooling2D(pool_size=8, padding = 'same')(x)
x = Conv2D(64, kernel_size=8, padding = 'same', activation='relu', use_bias=False, kernel_initializer = 'lecun_uniform')(x)
x = Conv2D(64, kernel_size=8, padding = 'same', activation='relu', use_bias=False, kernel_initializer = 'lecun_uniform')(x)
x = MaxPooling2D(pool_size=4, padding = 'same')(x)
x = Flatten()(x)
x = Dropout(.2)(x)

# Add the side channel data to the first fully connected layer
if side_channel_length > 0:
    x = concatenate([x, input_side_channel])

x = Dense(10, activation='relu', use_bias=False)(x)
prediction = Dense(1, activation="linear")(x)

network_model = Model(inputs=all_inputs, outputs=prediction)
adam = adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0, clipnorm=1.0)
network_model.compile(optimizer=adam, loss="mean_squared_error")

#####################################
#        Run the Experiment         #
#####################################

output_path = head + "/"
experiment.experiment(network_model, output_path, dataset_model=dataset_model, args=None, config=config)
