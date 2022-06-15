#!/bin/sh

"""
Pipeline to prepare data from new patients : 
1) combat harmonise (make sure you have computed the combat parameters for your site prior)
2) inter & intra normalisation
3) Save data in the "combat" hdf5 matrix
"""


# Import packages
import os
import sys
import argparse
import pandas as pd
import numpy as np
from meld_classifier.meld_cohort import MeldCohort, MeldSubject
from meld_classifier.data_preprocessing import Preprocess, Feature
from meld_classifier.paths import BASE_PATH, NORM_CONTROLS_PARAMS_FILE, COMBAT_PARAMS_FILE, NEWSUBJECTS_DATASET, MELD_SITE_CODES

def create_dataset_file(subjects, output_path):
    df=pd.DataFrame()
    subjects_id = [subject for subject in subjects]
    df['subject_id']=subjects_id
    df['split']=['test' for subject in subjects]
    df.to_csv(output_path)

def which_combat_file(site_code):
    file_site=os.path.join(BASE_PATH, f'MELD_{site_code}', f'{site_code}_combat_parameters.hdf5')
    if site_code in MELD_SITE_CODES:
        print('Use combat parameters from MELD cohort')
        return os.path.join(BASE_PATH,COMBAT_PARAMS_FILE)
    elif os.path.isfile(file_site):
        print(f'Use combat parameters from new site {file_site}')
        return file_site
    else:
        sys.exit('No parameters for combat harmonisation were found for this site code. You need to harmonise your new data with the new_site_harmonisation_script.py before to run this script')
  
    
if __name__ == "__main__":
    #parse commandline arguments 
    parser = argparse.ArgumentParser(description='data-processing on new subject ')
    parser.add_argument('-ids','--list_ids',
                        help='Subject ID.',
                        required=True,)
    parser.add_argument('-site', '--site_code', 
                        type=str, 
                        help='site code', 
                         required=True,)
    parser.add_argument('-d', '--output_dir', 
                        type=str, 
                        help='path to store hdf5 files',
                        default=BASE_PATH)   
    parser.add_argument("--withoutflair",
						action="store_true",
						default=False,
						help="do not use flair information") 
    args = parser.parse_args()
    subject_ids = np.array(np.loadtxt(args.list_ids, dtype='str',ndmin=1))
    output_dir = args.output_dir  
    site_code = args.site_code   
    dataset_newSubject = os.path.join(BASE_PATH, NEWSUBJECTS_DATASET)

    # Set features and smoothed values
    if args.withoutflair:
        features = {
		".on_lh.thickness.mgh": 10,
		".on_lh.w-g.pct.mgh" : 10,
		".on_lh.pial.K_filtered.sm20.mgh": None,
		'.on_lh.sulc.mgh' : 5,
		'.on_lh.curv.mgh' : 5,
			}
    else:
        features = {
		".on_lh.thickness.mgh": 10,
		".on_lh.w-g.pct.mgh" : 10,
		".on_lh.pial.K_filtered.sm20.mgh": None,
		'.on_lh.sulc.mgh' : 5,
		'.on_lh.curv.mgh' : 5,
		'.on_lh.gm_FLAIR_0.25.mgh' : 10,
		'.on_lh.gm_FLAIR_0.5.mgh' : 10,
		'.on_lh.gm_FLAIR_0.75.mgh' : 10,
		".on_lh.gm_FLAIR_0.mgh": 10,
		'.on_lh.wm_FLAIR_0.5.mgh' : 10,
		'.on_lh.wm_FLAIR_1.mgh' : 10,
    			}
    feat = Feature()
    features_smooth = [feat.smooth_feat(feature, features[feature]) for feature in features]
    features_combat = [feat.combat_feat(feature) for feature in features_smooth]
    
    ### INITIALISE ###
    #create dataset
    create_dataset_file(subject_ids, dataset_newSubject)
    

    ### COMBAT DATA ###
    #-----------------------------------------------------------------------------------------------
    print('PROCESS 2 : COMBAT')
    #create cohort for the new subject
    c_smooth = MeldCohort(hdf5_file_root='{site_code}_{group}_featurematrix_smoothed.hdf5', dataset=dataset_newSubject)
    #get combat parameters
    combat_params_file = which_combat_file(site_code)
    #create object combat
    combat =Preprocess(c_smooth,
                       write_hdf5_file_root='{site_code}_{group}_featurematrix_combat.hdf5',
                       data_dir=output_dir)
    #features names
    for feature in features_smooth:
        print(feature)
        combat.combat_new_subject(feature, combat_params_file)
    

    ###  INTRA, INTER & ASYMETRY ###
    #-----------------------------------------------------------------------------------------------
    print('PROCESS 3 : INTRA, INTER & ASYMETRY')
    #create cohort to normalise
    c_combat = MeldCohort(hdf5_file_root='{site_code}_{group}_featurematrix_combat.hdf5', dataset=dataset_newSubject, data_dir=output_dir)
    # provide mean and std parameter for normalisation by controls
    param_norms_file = os.path.join(BASE_PATH, NORM_CONTROLS_PARAMS_FILE)
    # create object normalisation
    norm = Preprocess(c_combat,
                        write_hdf5_file_root='{site_code}_{group}_featurematrix_combat.hdf5',
                        data_dir=output_dir)
    # call functions to normalise data
    for feature in features_combat:
        print(feature)
        norm.intra_inter_subject(feature, params_norm = param_norms_file)
        norm.asymmetry_subject(feature, params_norm = param_norms_file )
