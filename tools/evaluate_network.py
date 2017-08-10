import argparse
from dataset_models.sdo.aia import aia

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
dataset_model = aia.AIA(side_channels=["current_goes"], aia_image_count=3)

# Load and evaluate the network
dataset_model.evaluate_network(args.network_model[0])
