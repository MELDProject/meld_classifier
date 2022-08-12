"""
Pipeline to prepare data from new patients : 
1) combat harmonise (make sure you have computed the combat harmonisation parameters for your site prior)
2) inter & intra normalisation
3) Save data in the "combat" hdf5 matrix
"""

## To run : python run_script_preprocessing.py -site <site_code> -ids <text_file_with_subject_ids> 

import os
import sys
import argparse
import pandas as pd
import numpy as np
import tempfile
from os.path import join as opj
from meld_classifier.meld_cohort import MeldCohort
from meld_classifier.data_preprocessing import Preprocess, Feature
from meld_classifier.paths import BASE_PATH, MELD_PARAMS_PATH, MELD_DATA_PATH, NORM_CONTROLS_PARAMS_FILE, COMBAT_PARAMS_FILE, MELD_SITE_CODES

def create_dataset_file(subjects_ids, save_file):
    df=pd.DataFrame()
    if  isinstance(subjects_ids, str):
        subjects_ids=[subjects_ids]
    df['subject_id']=subjects_ids
    df['split']=['test' for subject in subjects_ids]
    df.to_csv(save_file)

def which_combat_file(site_code):
    file_site=os.path.join(BASE_PATH, f'MELD_{site_code}', f'{site_code}_combat_parameters.hdf5')
    if site_code=='TEST':
        site_code = 'H4'
    if site_code in MELD_SITE_CODES:
        print('INFO: Use combat parameters from MELD cohort')
        return os.path.join(MELD_PARAMS_PATH,COMBAT_PARAMS_FILE)
    elif os.path.isfile(file_site):
        print(f'INFO: Use combat parameters from site {site_code} : {file_site}')
        return file_site
    else:
        print(f'INFO: Could not find combat parameters for {site_code}')
        return 'None'

def check_demographic_file(demographic_file, subject_ids):
    #check demographic file has the right columns
    try:
        df = pd.read_csv(demographic_file)
        df[['ID', 'Sex', 'Age at preoperative']]
    except Exception as e:
        print(f"ERROR: Error with the demographic file provided for the harmonisation. \n {e}")
        os.sys.exit(-1)
    #check demographic file has the right subjects
    if set(subject_ids).issubset(set(np.array(df['ID']))):
        return demographic_file
    else:
        print('ERROR: Missing subject in the demographic file')
        os.sys.exit(-1)


def run_data_processing_new_subjects(subject_ids, site_code, combat_params_file=None, output_dir=BASE_PATH, withoutflair=False):
 
    # Set features and smoothed values
    if withoutflair:
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
    tmp = tempfile.NamedTemporaryFile(mode="w")
    create_dataset_file(subject_ids, tmp.name)  

    if combat_params_file==None:
        combat_params_file = which_combat_file(site_code)
    if combat_params_file=='need_harmonisation':
        sys.exit('ERROR: You need to compute the combat harmonisation parameters for this site before to run combat')

    ### COMBAT DATA ###
    #-----------------------------------------------------------------------------------------------
    print('STEP: Combat harmonise')
    print(f'INFO: process subjects {subject_ids}')
    #create cohort for the new subject
    c_smooth = MeldCohort(hdf5_file_root='{site_code}_{group}_featurematrix_smoothed.hdf5', dataset=tmp.name)
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
    print('STEP : Intra-inter normalisation & asymmetry')
    print(f'INFO: process subjects {subject_ids}')
    #create cohort to normalise
    c_combat = MeldCohort(hdf5_file_root='{site_code}_{group}_featurematrix_combat.hdf5', dataset=tmp.name)
    # provide mean and std parameter for normalisation by controls
    param_norms_file = os.path.join(MELD_PARAMS_PATH, NORM_CONTROLS_PARAMS_FILE)
    # create object normalisation
    norm = Preprocess(c_combat,
                        write_hdf5_file_root='{site_code}_{group}_featurematrix_combat.hdf5',
                        data_dir=output_dir)
    # call functions to normalise data
    for feature in features_combat:
        print(feature)
        norm.intra_inter_subject(feature, params_norm = param_norms_file)
        norm.asymmetry_subject(feature, params_norm = param_norms_file )

    tmp.close()


def new_site_harmonisation(subject_ids, site_code, demographic_file, output_dir=BASE_PATH, withoutflair=False):

    # Set features and smoothed values
    if withoutflair:
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
    
    ### INITIALISE ###
    #check enough subjects for harmonisation
    if len(np.unique(subject_ids))<30:
        print("ERROR: There are not enough subjects to proceed to an acurate harmonisation of the data... Rerun with at least 30 subjects.")
        sys.exit(-1)

    #create dataset
    tmp = tempfile.NamedTemporaryFile(mode="w")
    create_dataset_file(subject_ids, tmp.name)

    check_demographic_file(demographic_file, subject_ids)
   
    ### COMBAT DISTRIBUTED DATA ###
    #-----------------------------------------------------------------------------------------------
    print('STEP: Compute combat harmonisation parameters for new site')
        
    #create cohort for the new subject
    c_smooth= MeldCohort(hdf5_file_root='{site_code}_{group}_featurematrix_smoothed.hdf5', 
                       dataset=tmp.name)
    #create object combat
    combat =Preprocess(c_smooth,
                           site_codes=[site_code],
                           write_hdf5_file_root="MELD_{site_code}/{site_code}_combat_parameters.hdf5",
                           data_dir=output_dir)
    #features names
    for feature in features_smooth:
        print(feature)
        combat.get_combat_new_site_parameters(feature, demographic_file)

    tmp.close()

def run_script_preprocessing(site_code, list_ids=None, sub_id=None, output_dir=BASE_PATH, demographic_file=None, harmonisation_only=False, withoutflair=False):
    print(site_code)
    site_code = str(site_code)
    subject_id=None
    subject_ids=None
    if list_ids != None:
        list_ids=opj(MELD_DATA_PATH, list_ids)
        try:
            sub_list_df=pd.read_csv(list_ids)
            subject_ids=np.array(sub_list_df.ID.values)
        except:
            subject_ids=np.array(np.loadtxt(list_ids, dtype='str', ndmin=1)) 
        else:
                print(f"ERROR: Could not open {subject_ids}")
                sys.exit(-1)                
    elif sub_id != None:
        subject_id=sub_id
        subject_ids=np.array([sub_id])
    else:
        print('ERROR: No ids were provided')
        print("ERROR: Please specify both subject(s) and site_code ...")
        sys.exit(-1) 
       
    #check that combat parameters exist for this site or compute it
    combat_params_file = which_combat_file(site_code)
    if combat_params_file=='None':
        print(f'INFO: Compute combat parameters for {site_code} with subjects {subject_ids}')
        if demographic_file == None:
            print('ERROR: Please provide a demographic file using the flag "-demos" to harmonise your data')
            os.sys.exit(-1)
        else:
            #check that demographic file exist and is adequate
            demographic_file = os.path.join(MELD_DATA_PATH, demographic_file) 
            if os.path.isfile(demographic_file):
                print(f'INFO: Use demographic file {demographic_file}')
                demographic_file = check_demographic_file(demographic_file, subject_ids) 
            else:
                print(f'ERROR: Could not find demographic file provided {demographic_file}')
                os.sys.exit(-1)
        #compute the combat parameters for a new site
        new_site_harmonisation(subject_ids, site_code=site_code, demographic_file=demographic_file, output_dir=output_dir, withoutflair=withoutflair)

    if not harmonisation_only:
        run_data_processing_new_subjects(subject_ids, site_code=site_code, output_dir=output_dir, withoutflair=withoutflair)

if __name__ == '__main__':

    #parse commandline arguments 
    parser = argparse.ArgumentParser(description='data-processing on new subject')
    #TODO think about how to best pass a list
    parser.add_argument("-id","--id",
                        help="Subject ID.",
                        default=None,
                        required=False,
                        )
    parser.add_argument("-ids","--list_ids",
                        default=None,
                        help="File containing list of ids. Can be txt or csv with 'ID' column",
                        required=False,
                        )
    parser.add_argument("-site",
                        "--site_code",
                        help="Site code",
                        required=True,
                        )
    parser.add_argument('-demos', '--demographic_file', 
                        type=str, 
                        help='provide the demographic files for the harmonisation',
                        required=False,
                        default=None,
                        )
    parser.add_argument('--harmo_only', 
                        action="store_true", 
                        help='only compute the harmonisation combat parameters, no further process',
                        required=False,
                        default=False,
                        )
    parser.add_argument("--withoutflair",
                        action="store_true",
                        default=False,
                        help="do not use flair information",
                        )
    args = parser.parse_args()
    print(args)

    run_script_preprocessing(
                    site_code=args.site_code,
                    list_ids=args.list_ids,
                    sub_id=args.id,
                    demographic_file=args.demographic_file,
                    harmonisation_only = args.harmo_only,
                    withoutflair=args.withoutflair,
                    )