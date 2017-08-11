"""
This is a script for assembling a y file based on the files currently in the training and validation folders.

It assumes you are using version 4 data.

It looks for the training and validation folders inside the data folder specified in config.yml

Delays and catches can be specified in the arrays below.

The resulting file has the following columns:

1. Date Stamp
2. Max Xray during catch after delay
3. Delta Xray flux (Column 2 minus current Xray flux)
4. File name


"""

#!usr/env/bin python

import sys
sys.dont_write_bytecode=True
import csv
import datetime
import json
import numpy as np
import glob
import yaml
import re

with open("../../config.yml", "r") as config_file:
    config = yaml.load(config_file)

filePath = config['aia_path']

#Y_data
Y_file_path = 'Flux_2010_2017_max.csv'
Y_data = csv.reader(open(Y_file_path,'r'))
Y_data = list(Y_data)
Y_init_date = datetime.datetime.strptime(Y_data[1][0],'%Y-%m-%d %H:%M:%S')


###############################
#                             #
#  Delay and Catch definition #
#                             #
###############################

delays = [60]
catches = [12, 36, 60, 24*60]
#catches = [12, 36, 60, 24*60]

for delay in delays:
    for catch in catches:
        
        def get_Y_index(date):
            """Function that finds the index on the large y file for a given date"""
            delta = date-Y_init_date
            delta = delta.days*24*30 + delta.seconds/120 + 1
            return int(delta)
        
        def get_yval(filename, index):
            """Function that finds the maximum value within the catch window"""
            this_arr = []
            end_index = index + catch/2
            if end_index > len(Y_data):
                end_index = len(Y_data)  
            for j in range(index,end_index):
                if Y_data[j][1] != 'NA':
                    this_arr.append(float(Y_data[j][1]))
                else:
                    this_arr.append(0.0)
            return max(this_arr)
        
        #Adding column names
        Y_vals = [] #[[Date, Y-data - 12min/36min/1hr/24hr max, Channel 0,7 coefficients]]
        this_term = ['Date Stamp']
        this_term += ['Flux']
        this_term += ['Delta']
        this_term += ['Filename']
        Y_vals.append(this_term)
        
        #Processing flare files
        flare_files = glob.glob(filePath + 'validation/*_AIA*060m.fthr')
        flare_files += glob.glob(filePath + 'training/*_AIA*060m.fthr')
        for f in flare_files:

            inxSlash =  [m.start() for m in re.finditer('/', f)]
            inxSlash = inxSlash[len(inxSlash)-1]
            f = f[inxSlash+1:]
            inxUndr =  [m.start() for m in re.finditer('_', f)]
            inxUndr = inxUndr[0]
            flare_s = f[inxUndr+1:]            
            date_s = flare_s[3:16]

            #Current Xray flux
            date = datetime.datetime.strptime(date_s,'%Y%m%d_%H%M')
            Y_indexC = get_Y_index(date)

            #Future Xray flux after delay
            date += datetime.timedelta(seconds=60*delay)
            Y_indexF = get_Y_index(date)

            #Store values
            if Y_indexF < len(Y_data):
                this_term = [date_s]
                this_term += [get_yval(flare_s, Y_indexF)]  #Future Xray Flux
                this_term += [get_yval(flare_s, Y_indexF) - get_yval(flare_s, Y_indexC)]  #Delta
                this_term += [f]  #File name
                Y_vals.append(this_term)

        #Processing no flare files
        no_flare_files = glob.glob(filePath + 'validation/AIA*000m.fthr')
        no_flare_files += glob.glob(filePath + 'training/AIA*000m.fthr')
        for f in no_flare_files:
            inxSlash =  [m.start() for m in re.finditer('/', f)]
            inxSlash = inxSlash[len(inxSlash)-1]
            f = f[inxSlash+1:]

            date_s = f[3:16]

            #Current Xray flux
            date = datetime.datetime.strptime(date_s,'%Y%m%d_%H%M')
            Y_indexC = get_Y_index(date)

            #Future Xray flux after delay
            date += datetime.timedelta(seconds=60*delay)
            Y_indexF = get_Y_index(date)

            #Store values
            if Y_indexF < len(Y_data):
                this_term = [date_s]
                this_term += [get_yval(flare_s, Y_indexF)]  #Future Xray Flux
                this_term += [get_yval(flare_s, Y_indexF) - get_yval(flare_s, Y_indexC)]  #Delta
                this_term += [f]  #File name
                Y_vals.append(this_term)              
        
        if delay >= 60:
            this_delay = '%02dhr'%(delay/60)
        else:
            this_delay = '%02dmin'%(delay)
        if catch >= 60:
            this_dur = '%02dhr'%(catch/60)
        else:
            this_dur = '%02dmin'%(catch)
        
        print len(Y_vals)
        writer = csv.writer(file(filePath + 'y/All_Ys_%sDelay_%sMax.csv'%(this_delay,this_dur),'w'))
        writer.writerows(Y_vals)


##['2010-01-01 00:02:00', '6.2712e-08', '58']


