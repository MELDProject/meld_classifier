#!/bin/sh

"""
Pipeline to compute combat parameters for new site using distributed combat
"""


# Import packages
import os
import argparse
import pandas as pd
import numpy as np
from meld_classifier.meld_cohort import MeldCohort, MeldSubject
from meld_classifier.data_preprocessing import Preprocess, Feature
from meld_classifier.paths import BASE_PATH, COMBAT_PARAMS_FILE, NEWSUBJECTS_DATASET

def create_dataset_file(subjects, output_path):
    df=pd.DataFrame()
    subjects_id = [subject for subject in subjects]
    df['subject_id']=subjects_id
    df['split']=['test' for subject in subjects]
    df.to_csv(output_path)

if __name__ == "__main__":
    #parse commandline arguments 
    parser = argparse.ArgumentParser(description='compute combat parameters for new site ')
    parser.add_argument('-ids','--list_ids',
                        help='Subject ID.',
                        required=True,)
    parser.add_argument('-demos','--demographic_file',
                        help='path to the new site demographic file',
                        required=True,)
    parser.add_argument('-d', '--output_dir', 
                        type=str, 
                        help='path to store hdf5 files',
                        default=BASE_PATH)
    args = parser.parse_args()
    subject_ids = np.array(np.loadtxt(args.list_ids, dtype='str',ndmin=1))
    output_dir = args.output_dir   
    demographic_file = args.demographic_file
    dataset_newSubject = os.path.join(BASE_PATH, NEWSUBJECTS_DATASET)

    # Set features and smoothed values
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
   
    ### COMBAT DISTRIBUTED DATA ###
    #-----------------------------------------------------------------------------------------------
    print('PROCESS : COMPUTE COMBAT PARAMETERS NEW SITE')
        
    #create cohort for the new subject
    c_smooth= MeldCohort(hdf5_file_root='{site_code}_{group}_featurematrix_smoothed.hdf5', 
                       dataset=dataset_newSite)
    #create object combat
    combat =Preprocess(c_smooth,
                           site_codes=[new_site_code],
                           write_hdf5_file_root="{site_code}_combat_parameters.hdf5",
                           data_dir=site_combat_path)
    #features names
    for feature in features_smooth:
        print(feature)
        combat.get_combat_new_site_parameters(feature, demographic_file)
    

   