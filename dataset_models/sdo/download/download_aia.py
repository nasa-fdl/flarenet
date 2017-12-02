#!/usr/bin/env python
"""Download AIA SDO data from jsoc.stanford.edu. This script will download
data that are exclusively Flaring (one day before the flare event), and then a random
set of non-flaring events.
"""

import sys
sys.dont_write_bytecode=True

import datetime
import urllib
from astropy.io import fits
import numpy as np
import json
import pickle
import time
import urllib
import glob
import pandas as pd
import feather
import os
import os.path
import yaml

# Load the configuration file indicating where the files are stored
with open("config.yml", "r") as config_file:
    global_config = yaml.load(config_file)
    output_directory = global_config["aia_path"]

def download(only_flaring=True, output_directory=output_directory):
    """Download AIA data. The parameter determines whether
    you download the flaring on the non-flaring instances.
    """
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)

    Coeff_file_path = 'dataset_models/sdo/aia_degrade.txt'
    Coeff_data = pd.read_csv(Coeff_file_path, header=0)
    Coeff_init_date = datetime.datetime.strptime(Coeff_data['Date'].values[0][:-4],'%Y-%m-%dT%H:%M:%S')

    def get_Coeff_index(date_s):
        date = datetime.datetime.strptime(date_s,'%Y%m%d')
        delta = date-Coeff_init_date
        delta = delta.days + 1
        return int(delta)

    if only_flaring:
        flaresCSV = pd.read_csv('dataset_models/sdo/download/flares_CMX_Flux_2010_2017.csv',header = 1)
        flares = flaresCSV.values[flaresCSV.values[:,11]>=1e-6,:]
    else:
        flaresCSV = pd.read_csv('dataset_models/sdo/download/No_flares_Flux_2010_2017.csv',header = 1)
        flares = flaresCSV.values

    aia_ls = [94,131,171,193,211,304,335,1600]

    count = 0;
    it = np.nditer(flares[:,11], flags=['f_index'])
    while not it.finished:
        date_s = '%4d%02d%02d_%02d%02d'%(flares[it.index,0],flares[it.index,1],flares[it.index,2],flares[it.index,9],flares[it.index,10])
        date   = datetime.datetime.strptime(date_s,'%Y%m%d_%H%M')
        if only_flaring:
            date -= datetime.timedelta(seconds=3600*12)
        date_s = datetime.datetime.strftime(date,'%Y%m%d_%H%M')

        inx = get_Coeff_index(date_s[0:8])

        if only_flaring:
            if not os.path.exists(output_directory + 'AIA_data_Flares'):
                os.makedirs(output_directory + 'AIA_data_Flares')
            outfile_name = output_directory + 'AIA_data_Flares/%05d_AIA'%(count) + date_s + '_8chnls_1024_012h.fthr'
        else:
            if not os.path.exists(output_directory + 'AIA_data_NoFlares'):
                os.makedirs(output_directory + 'AIA_data_NoFlares')
            outfile_name = '/home/solardynamo/AIA_data_NoFlares/AIA' + date_s + '_8chnls_1024_0m.dat'
        if not os.path.isfile(outfile_name):
            print "writing " + outfile_name

            data = []
            is_complete = 1
            for aia_l in aia_ls:
                url = "http://jsoc.stanford.edu/data/aia/synoptic/%d/%02d/%02d/H%02d00/AIA%s_%04d.fits"%(date.year, date.month, date.day, date.hour,date_s,aia_l)
                print url
                response = urllib.urlopen(url)
                if response.getcode() == 200: 
                    chromosphere_image = fits.open(url, cache = False)
                    chromosphere_image.verify("fix")
                else:
                    is_complete = 0
                    break

                exptime = chromosphere_image[1].header['EXPTIME']
                if exptime == 0:
                    is_complete = 0
                    break

                if chromosphere_image[1].header['QUALITY'] != 0:
                    is_complete = 0
                    break
                flattened_image = chromosphere_image[1].data.flatten()
                data.append(flattened_image/exptime/Coeff_data[str(aia_l)].values[inx])

            if is_complete:
                data = np.array(data, dtype = np.float32).transpose()
                df = pd.DataFrame(data, columns=aia_ls)
                np.save(file(outfile_name,'w'),chromosphere_image[1].header)
                df = feather.write_dataframe(df, outfile_name)

        count += 1

        it.iternext()

if __name__ == "__main__":
    print "\n\n\n"
    print "################################################################"
    print "# Downloading all the flaring instances from the Flare Catalog #"
    print "################################################################"
    print "\n\n\n"
    download(only_flaring=True)
    print "\n\n\n\n\n\n##### FINISHED DOWNLOADING FLARING CASES, MOVING TO NON-FLARING #####\n\n\n\n\n\n\n"
    download(only_flaring=False)