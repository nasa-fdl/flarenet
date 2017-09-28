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
import os
from sunpy.net import jsoc
from astropy import units as u

print dir(jsoc)


flaresCSV = pd.read_csv('HMI_Control_No_flares_Flux.csv',header = 1)
flares = flaresCSV.values


count = 0;
it = np.nditer(flares[:,11], flags=['f_index'])
while (not it.finished) and (count < 50):

    newpath = '/home/solardynamo/HMI_NoFlares/f%05d'%(count) 
    if not os.path.exists(newpath):
        os.makedirs(newpath)

    print newpath

    date_s = '%4d%02d%02d_%02d%02d'%(flares[it.index,0],flares[it.index,1],flares[it.index,2],flares[it.index,9],flares[it.index,10])
    date   = datetime.datetime.strptime(date_s,'%Y%m%d_%H%M')

    dateStr = date - datetime.timedelta(seconds=3600*4)
    dateEnd = date + datetime.timedelta(seconds=3600*4)
    #print date
    date_s = datetime.datetime.strftime(dateStr,'%Y-%m-%dT%H:%M:00')
    date_e = datetime.datetime.strftime(dateEnd,'%Y-%m-%dT%H:%M:00')

    print date_s
    print date_e

    client = jsoc.JSOCClient()
    response = client.query(jsoc.Time(date_s, date_e),jsoc.Series('hmi.M_720s'), jsoc.Notify('amunozj@boulder.swri.edu'), jsoc.Sample(2*u.hour))
    print response

    res = client.get(response, path = newpath)

    print count

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

