## This script runs the MELD surface-based FCD classifier on the patient using the output features from script 2.
## The predicted clusters are then saved as file " " in the /output/<pat_id>/xhemi/classifier folder
## The predicted clusters are then registered back to native space and saved as a .mgh file in the /output/<pat_id>/classifier folder
## The predicted clusters are then registered back to the nifti volume and saved as nifti in the input/<pat_id>/predictions folder
## Individual reports for each identified cluster are calculated and saved in the input/<pat_id>/predictions/reports folder
## These contain images of the clusters on the surface and on the volumetric MRI as well as saliency reports
## The saliency reports include the z-scored feature values and how "salient" they were to the classifier

## To run : python run_script_prediction.py -ids <text_file_with_ids> -site <site_code>


import os
import sys
import numpy as np
import pandas as pd
import argparse
import tempfile
from os.path import join as opj
from meld_classifier.paths import FS_SUBJECTS_PATH, MELD_DATA_PATH, DEFAULT_HDF5_FILE_ROOT, EXPERIMENT_PATH, MODEL_PATH, MODEL_NAME
from meld_classifier.evaluation import Evaluator
from meld_classifier.experiment import Experiment
from meld_classifier.meld_cohort import MeldCohort
from scripts.manage_results.register_back_to_xhemi import register_subject_to_xhemi
from scripts.manage_results.move_predictions_to_mgh import move_predictions_to_mgh
from scripts.manage_results.plot_prediction_report import generate_prediction_report
from meld_classifier.tools_commands_prints import get_m

def create_dataset_file(subjects_ids, save_file):
    df=pd.DataFrame()
    if  isinstance(subjects_ids, str):
        subjects_ids=[subjects_ids]
    df['subject_id']=subjects_ids
    df['split']=['test' for subject in subjects_ids]
    df.to_csv(save_file)

def predict_subjects(subject_ids, output_dir, plot_images = False, saliency=False,
experiment_path=EXPERIMENT_PATH, experiment_name=MODEL_NAME, hdf5_file_root= DEFAULT_HDF5_FILE_ROOT,):       
    ''' function to predict on new subject using trained MELD classifier'''
    
    # create dataset csv
    tmp = tempfile.NamedTemporaryFile(mode="w")
    create_dataset_file(subject_ids, tmp.name)

    # load models
    exp = Experiment(experiment_path=experiment_path, experiment_name=experiment_name)
    exp.init_logging()

    #update experiment with subjects to predict
    exp.cohort = MeldCohort(hdf5_file_root=hdf5_file_root, dataset=tmp.name)
    
    #create sub-folders if do not exist
    os.makedirs(output_dir , exist_ok=True )
    os.makedirs(os.path.join(output_dir, "results"),  exist_ok=True)
    if plot_images:
        os.makedirs(os.path.join(output_dir, "results", "images"), exist_ok=True)
    
    # launch evaluation
    eva = Evaluator(exp, mode="inference", subject_ids=subject_ids, save_dir=output_dir)
    for subject in subject_ids:
        eva.load_predict_single_subject(
                subject, fold="", plot=plot_images, saliency=saliency, suffix=""
        )

def run_script_prediction(site_code, list_ids=None, sub_id=None, no_prediction_nifti=False, no_report=False, split=False, verbose=False):
    
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
            sys.exit(get_m(f'Could not open {subject_ids}', None, 'ERROR'))       
    elif sub_id != None:
        subject_id=sub_id
        subject_ids=np.array([sub_id])
    else:
        print(get_m(f'No ids were provided', None, 'ERROR'))
        print(get_m(f'Please specify both subject(s) and site_code ...', None, 'ERROR'))
        sys.exit(-1) 
    
    # initialise variables
    experiment_name = MODEL_NAME
    experiment_path = os.path.join(EXPERIMENT_PATH, MODEL_PATH)
    fold = 'all'
    subjects_dir = FS_SUBJECTS_PATH
    classifier_output_dir = opj(MELD_DATA_PATH,'output','classifier_outputs')
    prediction_file = opj(classifier_output_dir, "results", f"predictions_{experiment_name}.hdf5")
    data_dir = opj(MELD_DATA_PATH,'input')
    predictions_output_dir = opj(MELD_DATA_PATH,'output','predictions_reports')
    
    #split the subject in group of 5 if big number of subjects
    chunked_subject_list = list()
    if (split) and (len(subject_ids))>5 : 
        chunk_size = min(len(subject_ids), 5)
        for i in range(0, len(subject_ids), chunk_size):
            chunked_subject_list.append(subject_ids[i : i + chunk_size])
        print(get_m(f'{len(subject_ids)} subjects splitted in {len(chunked_subject_list)}', None, 'INFO'))
    else:
        subject_ids_chunk = chunked_subject_list.append(subject_ids)

    # for each chunk of subjects
    for subject_ids_chunk in chunked_subject_list:
        print(get_m(f'Run predictions', subject_ids_chunk, 'STEP 1'))
        
        #predict on new subjects 
        predict_subjects(subject_ids=subject_ids_chunk, 
                       output_dir=classifier_output_dir,  
                       plot_images=True, 
                       saliency=True,
                       experiment_path=experiment_path, 
                       experiment_name=experiment_name, 
                       hdf5_file_root= DEFAULT_HDF5_FILE_ROOT)
        
        if not no_prediction_nifti:        
            #Register predictions to native space
            print(get_m(f'Move predictions into volume', subject_ids_chunk, 'STEP 2'))
            move_predictions_to_mgh(subject_ids=subject_ids_chunk, 
                                    subjects_dir=subjects_dir, 
                                    prediction_file=prediction_file,
                                    verbose=verbose)

            #Register prediction back to nifti volume
            print(get_m(f'Move prediction back to native space', subject_ids_chunk, 'STEP 3'))
            register_subject_to_xhemi(subject_ids=subject_ids_chunk, 
                                      subjects_dir=subjects_dir, 
                                      output_dir=predictions_output_dir, 
                                      verbose=verbose)
        
        if not no_report:
            # Create individual reports of each identified cluster
            print(get_m(f'Create pdf report', subject_ids_chunk, 'STEP 4'))
            generate_prediction_report(
                subject_ids = subject_ids_chunk,
                data_dir = data_dir,
                hdf_predictions=prediction_file,
                experiment_path=experiment_path, 
                experiment_name=experiment_name, 
                output_dir = predictions_output_dir,
                hdf5_file_root = DEFAULT_HDF5_FILE_ROOT
            )

if __name__ == '__main__':

    #parse commandline arguments 
    parser = argparse.ArgumentParser(description='')
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
    parser.add_argument('--no_prediction_nifti',
                        action="store_true",
                        help='Only predict. Does not produce prediction on native T1, nor report',
                        )
    parser.add_argument('--no_report',
                        action="store_true",
                        help='Predict and map back into native T1. Does not produce report',)
    parser.add_argument('--split',
                        action="store_true",
                        help='Split subjects list in chunk to avoid data overload',
                        )
    parser.add_argument("--debug_mode", 
                        help="mode to debug error", 
                        required=False,
                        default=False,
                        action="store_true",
                        )
    args = parser.parse_args()
    print(args)    

    run_script_prediction(
                        site_code = args.site_code,
                        list_ids=args.list_ids,
                        sub_id=args.id,
                        no_prediction_nifti = args.no_prediction_nifti,
                        no_report = args.no_report,
                        split = args.split, 
                        verbose = args.debug_mode
                        )
                
