"""
This is a script for finding the best performing neural network architecture among
a set of potential architectures. While you can run this script directly from the
command line, the idea here is to have a Bayesian model select the architecture
most likely to improve on the past experiences. Through many repeated
experiments the Bayesian model finds a high-performing network
and the space weather person is responsible for interpreting the
best performing network.

The network architectures searchable by this script follow
the form:

INPUT -> [[CONV -> RELU]*N -> POOL?]*M -> [FC -> RELU]*K -> FC

Unpacking this statement, the architecture starts with the inputs, then
passes through a convolutional layer with a rectified linear unit N times.
Next the layers are max pooled. The combination of convolutions
and pooling is repeated M times, before passing through K fully connected
layers having a rectified linear unit activation. The final layer of the
nework is a fully connected layer having a single output.

Layer parameters can be specified via the command line.
Each of the parameters have defaults so you can just start the script and watch it
train. If you want to specify parameters from the command line, they are specified with
flags. So to run with the default parameters you would run:
`python architectures.py -pool_1_width 2 -pool_1_height 2 -pool_1_stride 1 -conv_1_channels 8 -conv_1_width 1 -conv_1_height 1 -conv_1_stride 1 -pool_2_width 2 -pool_2_height 2 -pool_2_stride 1 -conv_2_channels 4 -conv_2_width 4 -conv_2_height 4 -conv_2_stride 1 -dropout_rate .3 -dense_1_count 128``

If you don't specify these parameters at the command line, then the default value
will be used.

Before running the script, we recommend you start the tensorboard server so you
can track the progress.

`tensorboard --logdir=/tmp/version4`

"""

#####################################
#        Importing Modules          #
#####################################

# Neural network specification
from keras.layers import Input, Dense, Conv2D, Conv1D, MaxPooling2D, Dropout, Flatten, Activation, AveragePooling2D, concatenate, Lambda, advanced_activations
from keras.models import Model
from keras.callbacks import ModelCheckpoint
from keras import backend as K

# Linear algebra library within Python
import numpy as np

# Deep learning training library
from keras.callbacks import TensorBoard

# Utilities for this script
import os
import random
import argparse
import sys
import yaml

# Library for parsing arguments
import argparse

# Libraries packaged with this repository
from network_models.training_callbacks import TrainingCallbacks
from dataset_models.sdo.aia import aia, layers
from tools import tools

from keras.optimizers import adam

# Uncomment to force training to take place on the CPU
import os
#os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"   # see issue #152
#os.environ["CUDA_VISIBLE_DEVICES"] = ""
#os.environ["CUDA_VISIBLE_DEVICES"] = "1"



#####################################
#        Specifying Network         #
#####################################

"""
These parameters specify the network architecture and are set from the
command line either by you (the user) or by a program that is
searching/optimizing the structure of the architecture.

You can change these values from the command line, or you can
modify the soure code to have hard-coded values. We generally
recommend you use the command line as stated above.
"""

parser = argparse.ArgumentParser(description='Train a neural network.')

parser.add_argument('ignore', metavar='N', type=str, nargs='*',
                    help='ignore this argument. It is used to accumulate positional arguments from SMAC')

# Set all pooling parameters to 1 to skip pooling layer
parser.add_argument('-pool_1_width', type=int, nargs="?", default=4)
parser.add_argument('-pool_1_height', type=int, nargs="?", default=4)
parser.add_argument('-pool_1_stride', type=int, nargs="?", default=4)

# Set conv_1_channels to 0 to not include this layer
parser.add_argument('-conv_1_channels', type=int, nargs="?", default=8)
parser.add_argument('-conv_1_width', type=int, nargs="?", default=4)
parser.add_argument('-conv_1_height', type=int, nargs="?", default=4)
parser.add_argument('-conv_1_stride', type=int, nargs="?", default=1)
conv_1_activation = "relu" # Not available initially

# Set all pooling parameters to 1 to skip pooling layer
parser.add_argument('-pool_2_width', type=int, nargs="?", default=4)
parser.add_argument('-pool_2_height', type=int, nargs="?", default=4)
parser.add_argument('-pool_2_stride', type=int, nargs="?", default=4)

# Set conv_2_channels to 0 to not include this layer
parser.add_argument('-conv_2_channels', type=int, nargs="?", default=8)
parser.add_argument('-conv_2_width', type=int, nargs="?", default=2)
parser.add_argument('-conv_2_height', type=int, nargs="?", default=2)
parser.add_argument('-conv_2_stride', type=int, nargs="?", default=1)
conv_2_activation = "relu" # Not available initially

parser.add_argument('-dropout_rate', type=float, nargs="?", default=.3)
parser.add_argument('-dense_1_count', type=int, nargs="?", default=16)
dense_1_activation = "relu" # Not available for search initially

# Final output for regression
dense_2_count = 1
dense_2_activation = "linear"

args = parser.parse_args()

#####################################
#        CONFIGURE OUTPUTS          #
#####################################

# How many images will be composited
aia_image_count = 1

# Set the paths
model_directory_path = "network_models/xray_flux_forecast/fdl4/trained_models/"
abspath = os.path.abspath(__file__)
tools.change_directory_to_root()
head, tail = os.path.split(abspath)
training_callbacks = TrainingCallbacks(model_directory_path, args)

#####################################
#        INITIALIZING DATA          #
#####################################

print("initializing data")

# Load the configuration file. You should never change
# the configuration within this file.
with open("config.yml", "r") as config_file:
    config = yaml.load(config_file)

# Uncomment only the side channel you want to include.
side_channels = []
#side_channels = ["current_goes"]
#side_channels = ["hand_tailored"]
#side_channels = ["true_value"]

dataset_model = aia.AIA(config["samples_per_step"], side_channels=side_channels, aia_image_count=aia_image_count, dependent_variable="forecast")

#####################################
#         SPECIFYING DATA           #
#####################################

seed = 0
random.seed(seed)
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

steps_per_epoch = config["steps_per_epoch"]
samples_per_step = config["samples_per_step"] # batch size
epochs = config["epochs"]

#####################################
#     Constructing Architecture     #
#####################################

print("constructing network in the Keras functional API")

# Center and scale the input data
for idx, input_image in enumerate(input_images):
    input_images[idx] = layers.LogWhiten()(input_image)
if len(input_images) > 1:
    x = concatenate(input_images)
else:
    x = input_images[0]

def relu_advanced(x):
    return K.relu(x, alpha=0.01)    

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


forecaster = Model(inputs=all_inputs, outputs=prediction)
adam = adam(lr=0.000001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0001, clipnorm=2.0)
forecaster.compile(optimizer=adam, loss="mean_squared_error")

# Print the netwrok summary information
forecaster.summary()
orig_stdout = sys.stdout
f = open(model_directory_path + training_callbacks.timestr + "/summary.txt", 'w')
sys.stdout = f
forecaster.summary() # This does not return the summary string so we capture standard out
sys.stdout = orig_stdout
f.close()

#print junk

print("##################")
print("Run identifier: " + str(training_callbacks.timestr))
print("You can find the results from this run in a folder named " + str(training_callbacks.timestr))
print("##################")

# Do not allow a configuration with more than 150 million parameters
if forecaster.count_params() > 150000000:
    print("exiting since this network architecture will contain too many parameters")
    print("Result for SMAC: SUCCESS, 0, 0, 999999999, 0") #  todo: figure out the failure string within SMAC
    exit()

#####################################
#   Optimizing the Neural Network   #
#####################################

# Save intermediate outputs including the full model
tensorboard_log_data_path = model_directory_path + "/tmp/version1/"
tensorboard_callbacks = TensorBoard(log_dir=tensorboard_log_data_path)
model_output_path = model_directory_path + training_callbacks.timestr + "/epochs/weights.{epoch:02d}-{val_loss:.2f}.hdf5"
if not os.path.exists(model_output_path):
    os.makedirs(model_output_path)
model_checkpoint = ModelCheckpoint(model_output_path)

history = forecaster.fit_generator(dataset_model.training_generator(),
                                   steps_per_epoch,
                                   max_queue_size=10,
                                   epochs=epochs,
                                   validation_data=dataset_model.get_validation_data(),
                                   callbacks=[tensorboard_callbacks, training_callbacks, model_checkpoint],
                                   workers=1,
)

# Loss on the training set
print("printing loss history")
print(history.history['loss'])

# Loss on the validation set
if 'val_loss' in history.history.keys():
    print("printing history of validation loss over all epochs:")
    print(history.history['val_loss'])

# Print the performance of the network for the SMAC algorithm
print("Result for SMAC: SUCCESS, 0, 0, %f, 0" % history.history['loss'][-1])
