#!/usr/bin/env python
 
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

Coeff_file_path = 'aia_degrade.txt'
Coeff_data = pd.read_csv('aia_degrade.txt',header=0)
Coeff_init_date = datetime.datetime.strptime(Coeff_data['Date'].values[0][:-4],'%Y-%m-%dT%H:%M:%S')

def get_Coeff_index(date_s):
    date = datetime.datetime.strptime(date_s,'%Y%m%d')
    delta = date-Coeff_init_date
    delta = delta.days + 1
    return int(delta)

flaresCSV = pd.read_csv('No_flares_Flux_2010_2017.csv',header = 1)
flares = flaresCSV.values

aia_ls = [94,131,171,193,211,304,335,1600]

count = 0;
it = np.nditer(flares[:,11], flags=['f_index'])
while not it.finished:
    date_s = '%4d%02d%02d_%02d%02d'%(flares[it.index,0],flares[it.index,1],flares[it.index,2],flares[it.index,9],flares[it.index,10])
    date   = datetime.datetime.strptime(date_s,'%Y%m%d_%H%M')
    print date
    #date -= datetime.timedelta(seconds=720*5)
    print date
    date_s = datetime.datetime.strftime(date,'%Y%m%d_%H%M')

    print date_s[0:8]
    inx = get_Coeff_index(date_s[0:8])    

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

        data.append(chromosphere_image[1].data.flatten()/exptime/Coeff_data[str(aia_l)].values[inx])          

    if is_complete:
        data = np.array(data, dtype = np.float32).transpose()
        #print data.shape

        df = pd.DataFrame(data, columns=aia_ls)
        #print df

        #np.save(file('AIA_data_2014/%05d_%s_AIA_%02d_1024_1024.dat'%(count,date_s,len(aia_ls)),'w'),data)  
        print '/home/solardynamo/AIA_data_NoFlares/AIA' + date_s + '_8chnls_1024_0m.dat'
        np.save(file('/home/solardynamo/AIA_data_NoFlares/AIA' + date_s + '_8chnls_1024_0m.hdr','w'),chromosphere_image[1].header)
        #np.save(file('/home/solardynamo/AIA_data_Flares/%05d_12m_AIA'%(count) + date_s + '_8chnls_1024.dat','w'),data) 
        df = feather.write_dataframe(df, '/home/solardynamo/AIA_data_NoFlares/AIA' + date_s + '_8chnls_1024_0m.fthr')

    count += 1             
   
    it.iternext()





# date = datetime.datetime.strptime('2012.02.21_00:00:00','%Y.%m.%d_%H:%M:%S')
# end_date = datetime.datetime.strptime('2012.03.11_00:00:00','%Y.%m.%d_%H:%M:%S')

# start_time = time.time()

# #big_d = []
# count = 0
# while date < end_date:
#     date_s = datetime.datetime.strftime(date,'%Y.%m.%d_%H:%M:%S')
#     print date_s
#    url = "http://jsoc.stanford.edu/cgi-bin/ajax/jsoc_info?ds=aia.lev1[%s_TAI/12s][?WAVELNTH=171?]&op=rs_list&key=T_REC&seg=image_lev1"%date_s
#    print url
#    response = urllib.urlopen(url)
#    data = json.loads(response.read())
#    filename = data['segments'][0]['values'][0]
#    url = "http://jsoc.stanford.edu"+filename
#    chromosphere_image = fits.open(url)
#    chromosphere_image.verify("fix")
#    chromosphere_image[1].data.dump('AIA_171/%03d_%s_171_4096_4096.dat'%(count,date_s))
#    date += datetime.timedelta(seconds=3600)
#    count += 1
#
##    d = np.load('AIA_171/171_%s_4096.dat'%date_s)
##    big_d.append(d)
##    print date
##    date += datetime.timedelta(seconds=3600)
##
##big_d = np.array(big_d, dtype = np.int16)
##np.save(file('123.dat','w'),big_d)
##print time.time() - start_time
#print count

# date = datetime.datetime.strptime('20140601_0000','%Y%m%d_%H%M')
# end_date = datetime.datetime.strptime('20140801_0000','%Y%m%d_%H%M')

# start_time = time.time()

# aia_ls = [94,131,171,193,211,304,335,1600]
# count = 0
# missing_dates = []
# while date <= end_date:
#     date_s = datetime.datetime.strftime(date,'%Y%m%d_%H%M')
#     print date_s

#     data = []
#     #datal = []
#     is_complete = 1

#     for aia_l in aia_ls:
#         rfile = '/home/thernandez/AIA2014/AIA' + date_s + '_%04d.fits'%(aia_l)
#         print rfile
#         tmp = glob.glob(rfile)

#         if len(tmp) == 0:
#             is_complete = 0
#             break

#         try:
#             chromosphere_image = fits.open(rfile,ignore_missing_end=True)
#         except:
#             is_complete = 0
#             break

#         exptime = chromosphere_image[1].header['EXPTIME']
#         if exptime == 0:
#             is_complete = 0
#             break

#         chromosphere_image.verify("fix")
#         dataNorm = chromosphere_image[1].data/exptime
#         data.append(dataNorm)
        
#         # dataNorm[dataNorm <= 0] = 1
#         # datal.append(np.log(dataNorm))


# #         url = "http://jsoc.stanford.edu/data/aia/synoptic/%d/%02d/%02d/H%02d00/AIA%s_%04d.fits"%(date.year, date.month, date.day, date.hour,date_s,aia_l)
# #         print url
# #         response = urllib.urlopen(url)
# #         if response.getcode() == 200: 
# #             chromosphere_image = fits.open(url)
# #             chromosphere_image.verify("fix")
# #             #chromosphere_image[1].data *= 16.0
# #             data.append(chromosphere_image[1].data)
# #         else:
# #             is_complete = 0
# #             missing_dates.append(date_s)
# #             break
# #     if is_complete:
# #         data = np.array(data, dtype = np.int16)
#          #np.save(file('/home/solardynamo/AIA_data_2014/AIA_%02d_%02dchnls.dat'%(date_s,len(aia_ls)),'w'),data)

#     if is_complete:
#         data = np.array(data, dtype = np.int16)
#         #Normalized
#         np.save(file('/home/solardynamo/AIA_data_2014/AIA' + date_s + '_%02dchnls.hdr'%(len(aia_ls)),'w'),chromosphere_image[1].header)
#         np.save(file('/home/solardynamo/AIA_data_2014/AIA' + date_s + '_%02dchnls.dat'%(len(aia_ls)),'w'),data)

#         #Log
#         # np.save(file('/home/solardynamo/AIA_data_2014_log/AIA' + date_s + '_%02dchnls_log.hdr'%(len(aia_ls)),'w'),chromosphere_image[1].header)
#         # np.save(file('/home/solardynamo/AIA_data_2014_log/AIA' + date_s + '_%02dchnls_log.dat'%(len(aia_ls)),'w'),datal)

#     date += datetime.timedelta(seconds=720)
#     count += 1

#writer = csv.writer(open('AIA_data_2014/missing_dates.csv','w'))
#writer.writerows(missing_dates)
# print count

