import argparse
import os
import sys
import shutil
import numpy as np
from meld_classifier.paths import MELD_DATA_PATH
import tempfile
import urllib.request

# --- download data from figshare ---
def _fetch_url(url, fname):
    def dlProgress(count, blockSize, totalSize):
        percent = int(count*blockSize*100/totalSize)
        sys.stdout.write("\r" + url + "...%d%%" % percent)
        sys.stdout.flush()
    return urllib.request.urlretrieve(url, fname, reporthook=dlProgress)

if __name__ == '__main__':
    # ensure that all data is downloaded
    parser = argparse.ArgumentParser(description="Update the classifier for the new release enabling distributed combat on new site")
    args = parser.parse_args()

    print("Only downloading new data")
    #TODO: change
    url = "https://figshare.com/ndownloader/files/36007619?private_link=961d477ae30c9fc99640"
    test_data_dir = MELD_DATA_PATH
    os.makedirs(test_data_dir, exist_ok=True)
    print('downloading update data from June 2022 release to '+ test_data_dir)
    with tempfile.TemporaryDirectory() as tmpdirname:
        # download to tmpdir
        _fetch_url(url, os.path.join(tmpdirname, "updateJune2022_data2.tar.gz"))
        # unpack
        shutil.unpack_archive(os.path.join(tmpdirname, "updateJune2022_data2.tar.gz"), test_data_dir)
    print(f"\nunpacked data to {test_data_dir}")      
    print("Done.")