# Fetch all the 128X128 images from Mark's server. You should create the "bin" directory
# before running this.

import scrape_list
import urllib
import os.path

for add in scrape_list.addresses:
    outfile_name = "bin/" + add.split("/")[-1]
    if not os.path.isfile(outfile_name):
        print add
        urllib.urlretrieve (add, outfile_name)
    else:
        print "skipping: " + add
        
