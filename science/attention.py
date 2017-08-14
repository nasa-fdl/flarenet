"""
The purpose of this script is to uncover what the network is paying attention to in the image.
This is accomplished by examining the gradients of the loss function with respect to the
inputs. Basically, the script asks, "how much will the loss function change if the input
value changes".
"""
import argparse
from dataset_models.sdo.aia import aia
import numpy as np
from keras import backend as K
from scipy.misc import imsave
from keras.layers import Input

print "WARNING: This script is incomplete and currently assumes"
print "you will use the AIA dataset model. Please update appropriately."
print "Potential updates include the image count and the side channel selection."

# Parse the command line arguments. You can view these from the command line
# by issuing `python evaluate_network.py -h`
parser = argparse.ArgumentParser(description='Generate a set of images showing the saliency of the input space')
parser.add_argument('side_channels', metavar='N', type=str, nargs=1,
                    help='Specify the side channel as none, true_value, current_goes, or hand_tailored.')
parser.add_argument('aia_image_count', metavar='N', type=int, nargs=1,
                    help='Specify the number of AIA images that were composited for the network.')
parser.add_argument('network_model', metavar='N', type=str, nargs=1,
                    help='the full path to the network model that we want to evaluate. This will be a file with the .hdf5 extension.')
args = parser.parse_args()

# Assign thearguments from the command line
side_channels = [args.side_channels[0]]
if side_channels[0] not in ["true_value", "current_goes", "hand_tailored"]:
    side_channels = []
aia_image_count = args.aia_image_count[0]
network_model_path = args.network_model[0]

# We set the learning phase to "test" so that the network will not apply dropout
K.set_learning_phase(0)

# Specify the data. This currently defaults to models including the hand_tailored side channel
dataset_model = aia.AIA(side_channels=side_channels, aia_image_count=aia_image_count, dependent_variable="flux delta")
network_model = dataset_model.get_network_model(network_model_path)

# Compile the function that generates the gradients for input images based on the network
def compile_saliency_function(model, grad_input_index=0):
    """
    Compiles a function to compute the saliency maps and predicted classes
    for a given minibatch of input images.
    @param model {Keras Model} The model loaded from a hdf5 format file.
    @param grad_input_index {int} The input component for which we are interested in the gradient.
        The first indexed items will be the AIA images, followed by the side channel information.
    """
    inp = model.input

    # Handle the case where there is a single input
    if type(inp) != list:
        inp = [inp]
    output = model.layers[-1].output
    loss = output[0]
    grads = K.gradients(loss, inp)
    #grads[0] /= (K.sqrt(K.mean(K.square(grads[0]))) + 1e-5) # regularize the gradients. Important if doing gradient updating to the inputs
    iterate = K.function(inp, [loss, grads[grad_input_index]])
    return iterate

def write_images(grads_value, image_index):
    """Write out the attention images to disc
    @param grads_value {tensor} The gradient tensor for the image.
    @param image_index {int} The index of the image we are wanting
        to print gradients for.
    """    
    for layer_idx in range(0,8):
        layer = grads_value[image_index].reshape(1,8,1024,1024)[0][layer_idx]
        most = float("-Inf")
        least = float("Inf")
        for idx in range(0, 1024):
            for idx2 in range(0, 1024):
                most = max(layer[idx][idx2], most)
                least = min(layer[idx][idx2], least)
        most = float(most)
        least = float(least)
        print "Layer " + str(layer_idx) + " largest gradient: " + str(most) + " smallest gradient: " + str(least)
        for idx in range(0, 1024):
            for idx2 in range(0, 1024):
                layer[idx][idx2] = 255.0 * (layer[idx][idx2] - least)/(most - least)

        imsave('gradient_image' + str(image_index) + "_layer_" + str(layer_idx) + '.png', layer)


# Get the validation data we are testing this script on, then select the first records
dataset = dataset_model.get_validation_data()
x_inputs = dataset[0]

# Add the first image to the inputs
query_input =  [x_inputs[0][0].reshape(1, 1024, 1024, 8)]

# Append all the images experienced by the network
if aia_image_count > 1:
    for i in range(1, aia_image_count):
        query_input.append(x_inputs[i][0].reshape(1, 1024, 1024, 8))

# Append the side channel data. This will currently only append one side channel.
if "hand_tailored" in side_channels:
    query_input.append(x_inputs[aia_image_count][0].reshape(1,25))
elif len(side_channels) > 0:
    query_input.append(x_inputs[aia_image_count][0].reshape(1,1))

# Write the image output for each of the channels of the AIA images
for image_index in range(0, aia_image_count):
    # Compile the saliency function for the input image
    saliency_function =  compile_saliency_function(network_model, grad_input_index=image_index)
    loss_value, grads_value = saliency_function(query_input)
    write_images(grads_value, image_index)
