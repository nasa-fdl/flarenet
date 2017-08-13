import argparse
from dataset_models.sdo.aia import aia
import numpy as np
from keras import backend as K
from scipy.misc import imsave

print "WARNING: This script is incomplete and currently assumes"
print "you will use the AIA dataset model. Please update appropriately."
print "Potential updates include the image count and the side channel selection."

# Parse the command line arguments. You can view these from the command line
# by issuing `python evaluate_network.py -h`
parser = argparse.ArgumentParser(description='Evaluate a network on data')
parser.add_argument('network_model', metavar='N', type=str, nargs=1,
                    help='the full path to the network model that we want to evaluate. This will be a file with the .hdf5 extension.')
args = parser.parse_args()

# Specify the data
#dataset_model = aia.AIA(side_channels=["hand_tailored"], aia_image_count=1, dependent_variable="flux delta")
dataset_model = aia.AIA(side_channels=[""], aia_image_count=1, dependent_variable="flux delta")
network_model = dataset_model.get_network_model(args.network_model[0])

def compile_saliency_function(model):
    """
    Compiles a function to compute the saliency maps and predicted classes
    for a given minibatch of input images.
    """
    inp1 = model.layers[0].input
    #inp2 = model.get_layer("Side_Channel").input
    output = model.layers[-1].output
    loss = output[0]
    grads = K.gradients(loss, inp1)[0]
    grads /= (K.sqrt(K.mean(K.square(grads))) + 1e-5)
    iterate = K.function([inp1, K.learning_phase()], [loss, grads])
    return iterate

saliency_function =  compile_saliency_function(network_model)

dataset = dataset_model.get_validation_data()
x_inputs = dataset[0]
single_point =  x_inputs[0][0].reshape(1, 1024, 1024, 8)

loss_value, grads_value = saliency_function([single_point, 0])

layer = grads_value.reshape(1,8,1024,1024)[0][0]
most = float("-Inf")
least = float("Inf")
print layer.shape
for idx in range(0, 1024):
    for idx2 in range(0, 1024):
        most = max(layer[idx][idx2], most)
        least = min(layer[idx][idx2], least)
most = float(most)
least = float(least)
for idx in range(0, 1024):
    for idx2 in range(0, 1024):
        layer[idx][idx2] = 255.0 * (layer[idx][idx2] - least)/(most - least)

imsave('gradient.png', layer)
