"""
This is a script for assembling a y file based on the files currently in the training and validation folders.

It assumes you are using version 4 data.

It looks for the training and validation folders inside the data folder specified in config.yml

Delays and durations need to be specified in the arrays below.


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

delays = [60]
durations = [12, 36, 60, 24*60]
#durations = [12, 36, 60, 24*60]

for delay in delays:
    for duration in durations:
        
        def get_Y_index(date):
            delta = date-Y_init_date
            delta = delta.days*24*30 + delta.seconds/120 + 1
            delta += delay/2  
            return int(delta)
        
        def get_yval(filename, index):
            this_arr = []
            end_index = index + duration/2
            if end_index > len(Y_data):
                end_index = len(Y_data)  
            for j in range(index,end_index):
                if Y_data[j][1] != 'NA':
                    this_arr.append(float(Y_data[j][1]))
                else:
                    this_arr.append(0.0)
            return max(this_arr)
        
        Y_vals = [] #[[Date, Y-data - 12min/36min/1hr/24hr max, Channel 0,7 coefficients]]
        this_term = ['Date Stamp']
        this_term += ['Flux']
        this_term += ['Filename']
        Y_vals.append(this_term)
        
        flare_files = glob.glob(filePath + '/validation/*_AIA*%03dm.fthr'%delay)
        flare_files += glob.glob(filePath + '/training/*_AIA*%03dm.fthr'%delay)
        for f in flare_files:

            inxSlash =  [m.start() for m in re.finditer('/', f)]
            inxSlash = inxSlash[len(inxSlash)-1]
            f = f[inxSlash+1:]
            inxUndr =  [m.start() for m in re.finditer('_', f)]
            inxUndr = inxUndr[0]
            flare_s = f[inxUndr+1:]            
            date_s = flare_s[3:16]
            date = datetime.datetime.strptime(date_s,'%Y%m%d_%H%M')
            date += datetime.timedelta(seconds=60*delay)

            Y_index = get_Y_index(date)
            if Y_index < len(Y_data):
                this_term = [date_s]
                this_term += [get_yval(flare_s, Y_index)]
                this_term += [f]
                Y_vals.append(this_term)


        no_flare_files = glob.glob(filePath + '/validation/AIA*%03dm.fthr'%(delay-60))
        no_flare_files += glob.glob(filePath + '/training/AIA*%03dm.fthr'%(delay-60))
        for f in no_flare_files:
            inxSlash =  [m.start() for m in re.finditer('/', f)]
            inxSlash = inxSlash[len(inxSlash)-1]
            f = f[inxSlash+1:]

            date_s = f[3:16]
            date = datetime.datetime.strptime(date_s,'%Y%m%d_%H%M')
            date += datetime.timedelta(seconds=60*delay)

            Y_index = get_Y_index(date)
            if Y_index < len(Y_data):
                this_term = [date_s]
                this_term += [get_yval(flare_s, Y_index)]
                this_term += [f]
                Y_vals.append(this_term)                
        
        if delay >= 60:
            this_delay = '%02dhr'%(delay/60)
        else:
            this_delay = '%02dmin'%(delay)
        if duration >= 60:
            this_dur = '%02dhr'%(duration/60)
        else:
            this_dur = '%02dmin'%(duration)
        
        print len(Y_vals)
        writer = csv.writer(file(filePath + '/y/All_Ys_%sDelay_%sMax.csv'%(this_delay,this_dur),'w'))
        writer.writerows(Y_vals)


##['2010-01-01 00:02:00', '6.2712e-08', '58']


