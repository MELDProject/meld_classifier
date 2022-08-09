## This script runs the MELD surface-based FCD classifier on the patient using the output features from script 2.
## The predicted clusters are then saved as file " " in the /output/<pat_id>/xhemi/classifier folder
## The predicted clusters are then registered back to native space and saved as a .mgh file in the /output/<pat_id>/classifier folder
## The predicted clusters are then registered back to the nifti volume and saved as nifti in the input/<pat_id>/predictions folder
## Individual reports for each identified cluster are calculated and saved in the input/<pat_id>/predictions/reports folder
## These contain images of the clusters on the surface and on the volumetric MRI as well as saliency reports
## The saliency reports include the z-scored feature values and how "salient" they were to the classifier

## To run : python new_pt_pirpine_script3.py -ids <text_file_with_ids> -site <site_code>


import os
import numpy as np
import pandas as pd
import argparse
import tempfile
from os.path import join as opj
from meld_classifier.paths import BASE_PATH, FS_SUBJECTS_PATH, MELD_DATA_PATH, SCRIPTS_DIR, DEFAULT_HDF5_FILE_ROOT, EXPERIMENT_PATH, MELD_DATA_PATH, MODEL_PATH, MODEL_NAME
from meld_classifier.evaluation import Evaluator
from meld_classifier.experiment import Experiment
from meld_classifier.meld_cohort import MeldCohort
from scripts.manage_results.register_back_to_xhemi import register_subject_to_xhemi
from scripts.manage_results.move_predictions_to_mgh import move_predictions_to_mgh
from scripts.manage_results.plot_prediction_report import generate_prediction_report

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
         
if __name__ == '__main__':

    #parse commandline arguments 
    parser = argparse.ArgumentParser(description='')
    #TODO think about how to best pass a list
    parser.add_argument('-id','--id',
                        help='Subjects ID',
                        required=False,
                        default=None)
    parser.add_argument('-ids','--list_ids',
                        help='Subjects IDs',
                        required=False,
                        default=None)
    parser.add_argument('-site','--site_code',
                        help='Site code',
                        required=True,)
    parser.add_argument('--no_prediction_nifti',
                        action="store_true",
                        help='Only predict. Does not produce prediction on native T1, nor report',)
    parser.add_argument('--no_report',
                        action="store_true",
                        help='Predict and map back into native T1. Does not produce report',)
    parser.add_argument('--split',
                        action="store_true",
                        help='Split subjects list in chunk to avoid data overload',)

    args = parser.parse_args()
    site_code=str(args.site_code)   
    if args.list_ids:
        try:
            sub_list_df = pd.read_csv(args.list_ids)
            subject_ids=np.array(sub_list_df.participant_id.values)
        except:
            subject_ids=np.array(np.loadtxt(args.list_ids, dtype='str', ndmin=1))     
    elif args.id:
        subject_ids=np.array([args.id])
    else:
        print('No ids were provided')
        subject_ids=None
    
    # initialise variables
    scripts_dir = os.path.join(SCRIPTS_DIR,'scripts')
    experiment_name = MODEL_NAME
    experiment_path = os.path.join(EXPERIMENT_PATH, MODEL_PATH)
    fold = 'all'
    subjects_dir = FS_SUBJECTS_PATH
    classifier_output_dir = opj(MELD_DATA_PATH,'output','classifier_outputs')
    prediction_file = opj(classifier_output_dir, "results", f"predictions_{experiment_name}.hdf5")
    predictions_output_dir = opj(MELD_DATA_PATH,'input')
    
    #split the subject in group of 5 if big number of subjects
    chunked_subject_list = list()
    if args.split: 
        chunk_size = min(len(subject_ids), 5)
        for i in range(0, len(subject_ids), chunk_size):
            chunked_subject_list.append(subject_ids[i : i + chunk_size])
        print(f'INFO: {len(subject_ids)} subjects splitted in {len(chunked_subject_list)} chunks')
    else:
        subject_ids_chunk = chunked_subject_list.append(subject_ids)
    
    # for each chunk of subjects
    for subject_ids_chunk in chunked_subject_list:
        print(f'INFO: run prediction on {subject_ids_chunk}')
        
        # predict on new subjects 
        predict_subjects(subject_ids=subject_ids_chunk, 
                        output_dir=classifier_output_dir,  
                        plot_images=True, 
                        saliency=True,
                        experiment_path=experiment_path, 
                        experiment_name=experiment_name, 
                        hdf5_file_root= DEFAULT_HDF5_FILE_ROOT)
        
        if (not args.no_prediction_nifti) & (not args.no_report):        
            # Register predictions to native space
            print('STEP1: move predictions into volume')
            move_predictions_to_mgh(subject_ids=subject_ids_chunk, 
                                    subjects_dir=subjects_dir, 
                                    prediction_file=prediction_file)

            # Register prediction back to nifti volume
            print('STEP2: move prediction back to native space')
            register_subject_to_xhemi(subject_ids=subject_ids_chunk, 
                                      subjects_dir=subjects_dir, 
                                      output_dir=predictions_output_dir)
        
        if (not args.no_report):
            # Create individual reports of each identified cluster
            print('STEP3: Create pdf report')
            generate_prediction_report(
                subject_ids = subject_ids_chunk,
                data_dir = predictions_output_dir,
                hdf_predictions=prediction_file,
                experiment_path=experiment_path, 
                experiment_name=experiment_name, 
                output_dir = predictions_output_dir,
                hdf5_file_root = DEFAULT_HDF5_FILE_ROOT
            )
                
