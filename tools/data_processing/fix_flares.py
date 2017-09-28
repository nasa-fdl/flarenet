#!usr/env/bin python

import sys
sys.dont_write_bytecode=True
import csv
import datetime
import json
import numpy as np
import glob
import feather
import pandas as pd

aia_ls = [94,131,171,193,211,304,335,1600]

Coeff_file_path = 'aia_degrade.txt'
Coeff_data = pd.read_csv('aia_degrade.txt',header=0)
Coeff_init_date = datetime.datetime.strptime(Coeff_data['Date'].values[0][:-4],'%Y-%m-%dT%H:%M:%S')

flare_files = glob.glob('/home/solardynamo/AIA_data_Flares/*.fthr')

def get_Coeff_index(date_s):
    date = datetime.datetime.strptime(date_s,'%Y%m%d')
    delta = date-Coeff_init_date
    delta = delta.days + 1
    return int(delta)

for i in range(0,len(flare_files)):
    print flare_files[i]
    df = feather.read_dataframe(flare_files[i])
    date_s = flare_files[i][43:51]

    inx = get_Coeff_index(date_s)

    for aia_l in aia_ls:
    	df[str(aia_l)] /= Coeff_data[str(aia_l)].values[inx]

    #df = feather.write_dataframe(df, flare_files[i])
 	
