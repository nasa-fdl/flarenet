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

`tensorboard --logdir=/tmp/version0`

"""

#####################################
#        Importing Modules          #
#####################################

# Neural network specification
from keras.layers import Input, Dense, Conv2D, MaxPooling2D, Dropout, Flatten, Activation
from keras.models import Model
from keras import backend as K

# Linear algebra library within Python
import numpy as np

# Deep learning training library
from keras.callbacks import TensorBoard

# Utilities for this script
import os
import random
import datetime
import argparse
import sys

# Library for parsing arguments
import argparse

# Uncomment to force training to take place on the CPU
#import os
#os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"   # see issue #152
#os.environ["CUDA_VISIBLE_DEVICES"] = ""

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
parser.add_argument('-conv_1_width', type=int, nargs="?", default=1)
parser.add_argument('-conv_1_height', type=int, nargs="?", default=1)
parser.add_argument('-conv_1_stride', type=int, nargs="?", default=1)
conv_1_activation = "relu" # Not available initially

# Set all pooling parameters to 1 to skip pooling layer
parser.add_argument('-pool_2_width', type=int, nargs="?", default=2)
parser.add_argument('-pool_2_height', type=int, nargs="?", default=2)
parser.add_argument('-pool_2_stride', type=int, nargs="?", default=4)

# Set conv_2_channels to 0 to not include this layer
parser.add_argument('-conv_2_channels', type=int, nargs="?", default=4)
parser.add_argument('-conv_2_width', type=int, nargs="?", default=4)
parser.add_argument('-conv_2_height', type=int, nargs="?", default=4)
parser.add_argument('-conv_2_stride', type=int, nargs="?", default=1)
conv_2_activation = "relu" # Not available initially

parser.add_argument('-dropout_rate', type=float, nargs="?", default=.3)
parser.add_argument('-dense_1_count', type=int, nargs="?", default=64)
dense_1_activation = "relu" # Not available for search initially

# Final output for regression
dense_2_count = 1
dense_2_activation = "linear"

args = parser.parse_args()

#####################################
#         SPECIFYING DATA           #
#####################################

data_directory = "/data/sw/version0/AIA_171/"
tensorboard_log_data_path = "/tmp/version0/"
seed = 0
random.seed(seed)
input_channels = 1
input_width = 4096
input_height = 4096
maximum_y_value = 0.00054361000000000004
y_reweight = 1839.55409209 # Scale maximum value to 1
input_image = Input(shape=(input_width, input_height, input_channels))
validation_steps = 20
x = input_image

#####################################
#     Constructing Architecture     #
#####################################

print "constructing network in the Keras functional API"

if args.pool_1_width != 1 or args.pool_1_height != 1 or args.pool_1_stride != 1:
    x = MaxPooling2D((args.pool_1_width, args.pool_1_height), padding='same', strides=args.pool_1_stride)(x)

if args.conv_1_channels is not 0:
    x = Conv2D(args.conv_1_channels, (args.conv_1_width, args.conv_1_height), activation=conv_1_activation, padding='same')(x)

if args.pool_2_width != 1 or args.pool_2_height != 1 or args.pool_2_stride != 1:
    x = MaxPooling2D((args.pool_2_width, args.pool_2_height), padding='same', strides=args.pool_2_stride)(x)

if args.conv_2_channels is not 0:
    x = Conv2D(args.conv_2_channels, (args.conv_2_width, args.conv_2_height), activation=conv_2_activation, padding='same')(x)
    
x = Flatten()(x)
x = Dropout(args.dropout_rate)(x)
x = Dense(args.dense_1_count, activation=dense_1_activation)(x)
prediction = Dense(1, activation=dense_2_activation)(x)

forecaster = Model(input_image, prediction)
forecaster.compile(optimizer='adadelta', loss='mean_absolute_error')

print forecaster.summary()

# Do not allow a configuration with more than 150 million parameters
if forecaster.count_params() > 150000000:
    print "exiting since this network architecture will contain too many paramters"
    print "Result for SMAC: SUCCESS, 0, 0, 999999999, 0" #  todo: figure out the failure string within SMAC
    exit()

"""
Debugging code:
  Uncomment to plot the network architecture.
"""
#from keras.utils import plot_model
#plot_model(forecaster, to_file='model3.png', show_shapes=True)
#exit()

#####################################
#        GENERATING DATA            #
#####################################

print "defining data generators"

# get a directory listing of the sdo data
filenames = os.listdir(data_directory)
random.shuffle(filenames)
train_files = filenames[:-validation_steps]
test_files = filenames[-validation_steps:]

print "loading results file"
results_path = data_directory + "../Y_GOES_XRAY_20120221_20120311_1hr_456.dat"
y_file = np.load(results_path)

def get_y(filename, y_file=y_file):
    """
    Get the true forecast result for the current filename.
    """
    index = int(filename.split("_")[0])    
    return y_file[index] * y_reweight  # Scale maximum value to 1

def generator():
    """
    Generate training samples
    """
    while 1:
        for f in train_files:
            data_x = np.load(data_directory + f)
            data_y = get_y(f)
            data_x = data_x.astype('float32') / 16509. # Standardize to [-1,1]
            data_x = np.reshape(data_x, (1, input_width, input_height, input_channels))
            data_y = np.reshape(data_y, (1))
            yield (data_x, data_y)
        random.shuffle(train_files)

def validation_generator():
    """
    Generate validation samples
    """
    while 1:
        for f in test_files:
            data_x = np.load(data_directory + f)
            data_y = get_y(f)
            data_x = data_x.astype('float32') / 16509. # Standardize to [-1,1]
            data_x = np.reshape(data_x, (1, input_width, input_height, input_channels))
            data_y = np.reshape(data_y, (1))
            yield (data_x, data_y)
        random.shuffle(test_files)


print "loading image dataset"

#####################################
#   Optimizing the Neural Network   #
#####################################

history = forecaster.fit_generator(generator(),
                                   100,  #  steps per epoch
                                   epochs=100,
                                   validation_data=validation_generator(),
                                   validation_steps=validation_steps,
                                   callbacks=[TensorBoard(log_dir=tensorboard_log_data_path)])

# Loss on the training set
print history.history['loss']

# Loss on the validation set
if 'val_loss' in history.history.keys():
    print history.history['val_loss']

# Print the performance of the network for the SMAC algorithm
print "Result for SMAC: SUCCESS, 0, 0, %f, 0" % history.history['loss'][-1]
