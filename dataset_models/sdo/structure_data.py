"""Create the training and validation separation on the data.
"""

import os
import os.path
import glob
import random
import yaml

print "This script creates different training and test set separations depending on which task you are working on."
print "The default will collect all the positive flaring instances and randomly subset them into training and test instances."

# Load the configuration file indicating where the files are stored
with open("config.yml", "r") as config_file:
    global_config = yaml.load(config_file)
    output_directory = global_config["aia_path"]

data_directory = output_directory + "AIA_data_Flares/"
training_directory = output_directory + "training/"
validation_directory = output_directory + "validation/"

for directory in [data_directory, training_directory, validation_directory]:
    if not os.path.exists(directory):
        os.makedirs(directory)

if glob.glob(training_directory + "*"):
    print glob.glob(training_directory + "*")
    print "training directory already contains files, exiting..."
    exit()
if glob.glob(validation_directory + "*"):
    print "validation directory already contains files, exiting..."
    exit()

file_names = glob.glob(data_directory + "*_AIA2*_8chnls_*.fthr")

random.seed(0)
random.shuffle(file_names)

validation_size = min(100, len(file_names)/10)
training_files = file_names[0:len(file_names)-validation_size]
validation_files = file_names[len(file_names)-validation_size:]

for training_file in training_files:
    os.symlink(data_directory + training_file.split("/")[-1], training_directory + training_file.split("/")[-1])
for validation_file in validation_files:
    os.symlink(data_directory + validation_file.split("/")[-1], validation_directory + validation_file.split("/")[-1])
