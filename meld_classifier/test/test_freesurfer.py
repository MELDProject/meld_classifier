#### tests for Freesurfer ####
# it test  : 
# - if Freesurfer is activated 
# - TODO : check Freesurfer version? 

import subprocess
from subprocess import Popen
import os
import pytest

@pytest.mark.slow
def test_freesurfer():
    # get freesurfer path
    command = format("$FREESURFER_HOME/bin/recon-all -sd /home/mathilde/Documents/projects/MELD_classifier/process/270123_test_meld_classifier/meld_data/output/fs_outputs -s MELD_H101_3T_C_00002 -all -dontrun")
    proc = Popen(command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, encoding='utf-8')
    stdout, stderr= proc.communicate()
    assert proc.returncode<=1 , "Oopsie ! Freesurfer failing with error {}".format(stdout)

    #check version
    command = format("$FREESURFER_HOME/bin/recon-all -version")
    proc = Popen(command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, encoding='utf-8')
    stdout, stderr= proc.communicate()
    version_ok = stdout.count("7.3.")
    assert version_ok==0 , "Oopsie ! Your Freesurfer version is 7.3 and is not compatible with MELD pipeline. Please downgrade to 7.2"
 
