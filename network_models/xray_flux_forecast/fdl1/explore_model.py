import numpy as np
from keras.models import load_model
import os
from datetime import timedelta, datetime

######################################
#               Paths                #
######################################

data_directory = "/data/sw/version1/x/"
results_path = "/data/sw/version1/y/Y_GOES_XRAY_201401.csv"
model_directory_path = "../../models/version1/"

# Record the paths and the models we will be evaluating, [[path, [models]], ...]
model_directory_paths = []
foldernames = os.listdir(model_directory_path)
for folder in foldernames:
    model_path = model_directory_path + folder + "/"
    model_filenames = os.listdir(model_path)
    model_filenames = [i for i in model_filenames if i.endswith(".hdf5") and os.path.isfile(model_path + i)]
    model_directory_paths.append([model_path, model_filenames])


######################################
#             Load Data              #
######################################

y_dict = {}
with open(results_path, "rb") as f:
    for line in f:
        split_y = line.split(",")
        y_dict[split_y[0]] = float(split_y[1])

def get_y_delta(filename, y_dict=y_dict):
    """
    Get the amount the flux increased or decreased between timesteps
    """
    split_filename = filename.split("_")
    k = split_filename[0] + "_" + split_filename[1]
    try:   
        future = y_dict[k]
    except Exception:
        future = 9999
    current = get_prior_y(filename)
    return abs(future - current)

        
def get_y(filename, y_dict=y_dict):
    """
    Get the true forecast result for the current filename.
    """
    split_filename = filename.split("_")
    k = split_filename[0] + "_" + split_filename[1]
    try:   
        future = y_dict[k]
    except Exception:
        future = 9999
    return future

def get_prior_y(filename, y_dict=y_dict):
    """
    Get the y value for the prior time step. This will
    generally be used so we can capture the delta in the
    prediction value.
    """
    f = filename.split("_")
    datetime_format = '%Y%m%d_%H%M'
    datetime_object = datetime.strptime(f[0]+"_"+f[1], datetime_format)
    td = timedelta(minutes=-12)
    prior_datetime_object = datetime_object + td
    prior_datetime_string = datetime.strftime(prior_datetime_object, datetime_format)
    try:   
        prior = y_dict[prior_datetime_string]
    except Exception:
        prior = 9999
    return prior

######################################
#               Predict              #
######################################

for model_path in model_directory_paths:
    for model_file in model_path[1]:
        model_full_path = model_path[0] + model_file
        model = load_model(model_full_path)

        # Load each of the x values and predict the y values with the best performing network
        x_predictions = {}
        filenames = os.listdir(data_directory) # get a directory listing of the sdo data

        samples = []
        for filename in filenames:
            data_x_sample = np.load(data_directory + filename)
            prediction = model.predict(data_x_sample.reshape(1, 1024, 1024, 8), verbose=0)
            x_predictions[filename] = [prediction, get_y_delta(filename), get_y(filename), get_prior_y(filename)]

        ######################################
        #           Write Output             #
        ######################################

        with open(model_full_path + ".performance", "w") as out:
            out.write("datetime, prediction, true y delta, true y, true prior y\n")
            keys = list(x_predictions)
            keys = sorted(keys)
            for key in keys:
                cur = x_predictions[key]
                out.write(key + "," + str(cur[0][0][0]) + "," + str(cur[1]) + "," + str(cur[2]) + "," + str(cur[3]) + "\n")
