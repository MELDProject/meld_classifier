## This script 
## 1. Extract surface-based features needed for the classifier :
##    1. Samples the features
##    2. Creates the registration to the template surface fsaverage_sym
##    3. Moves the features to the tempalte surface
##    4. Write feature in hdf5
## 2. Preprocess features : 
##    1. Smooth features and write in hdf5
##    2. Combat harmonised and write in hdf5
##    3. Normalise intra-subject, normalise inter-subject (by controls) and write in hdf5
##    4. Takes raw features and does intra-subject, asymmetry and then normalises by controls and write in hdf5

## To run : python new_pt_pirpine_script2.py -ids <text_file_with_subject_ids>  -site <site_code>


import os
import sys
import argparse
import subprocess as sub
import glob
from meld_classifier.paths import BASE_PATH, FS_SUBJECTS_PATH,  SCRIPTS_DIR
        
if __name__ == '__main__':

    #parse commandline arguments 
    parser = argparse.ArgumentParser(description='perform cortical parcellation using recon-all from freesurfer')
    parser.add_argument('-ids','--list_ids',
                        help='Subjects IDs in a text file',
                        required=True,)
    parser.add_argument('-site','--site_code',
                        help='Site code',
                        required=True,)
    parser.add_argument("--withoutflair",
			action="store_true",
			default=False,
			help="do not use flair information") 
    args = parser.parse_args()
    subject_ids=str(args.list_ids)
    site_code=str(args.site_code)
    scripts_dir = os.path.join(SCRIPTS_DIR,'scripts')
    
    # Launch script to preprocess features : combat harmonised & normalise
    if args.withoutflair:
        command = format(f"python {scripts_dir}/data_preparation/run_data_processing_new_subjects.py -ids {subject_ids} -site {site_code} -d {BASE_PATH} --withoutflair")
    else:
        command = format(f"python {scripts_dir}/data_preparation/run_data_processing_new_subjects.py -ids {subject_ids} -site {site_code} -d {BASE_PATH}")
    sub.check_call(command, shell=True)

    

    
