## This script runs the MELD surface-based FCD classifier on the patient using the output features from script 2.
## The predicted clusters are then saved as file " " in the /output/<pat_id>/xhemi/classifier folder
## The predicted clusters are then registered back to native space and saved as a .mgh file in the /output/<pat_id>/classifier folder
## The predicted clusters are then registered back to the nifti volume and saved as nifti in the input/<pat_id>/predictions folder
## Individual reports for each identified cluster are calculated and saved in the input/<pat_id>/predictions/reports folder
## These contain images of the clusters on the surface and on the volumetric MRI as well as saliency reports
## The saliency reports include the z-scored feature values and how "salient" they were to the classifier

## To run : python new_pt_pirpine_script3.py -ids <text_file_with_ids> -site <site_code>


import os
import sys
import argparse
import subprocess as sub
import glob
from meld_classifier.paths import BASE_PATH, FS_SUBJECTS_PATH, MELD_DATA_PATH, SCRIPTS_DIR, NEWSUBJECTS_DATASET, DEFAULT_HDF5_FILE_ROOT
from meld_classifier.predict_newsubject import predict_subjects
        
if __name__ == '__main__':

    #parse commandline arguments 
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('-ids','--list_ids',
                        help='Subjects IDs in a text file',
                        required=True,)
    parser.add_argument('-site','--site_code',
                        help='Site code',
                        required=True,)
    args = parser.parse_args()
    list_ids=str(args.list_ids)
    site_code=str(args.site_code)
    scripts_dir = os.path.join(SCRIPTS_DIR,'scripts')
    exp_fold = "output/classifier_outputs"
       
    # Run MELD surface-based FCD classifier on the patient
    new_data_parameters = {
        "hdf5_file_root": DEFAULT_HDF5_FILE_ROOT,
        "dataset": NEWSUBJECTS_DATASET,
        "saved_hdf5_dir": f"{MELD_DATA_PATH}/output/classifier_outputs",}    
    predict_subjects(list_ids, new_data_parameters, plot_images=True, saliency=True)
                     
    # Register predictions to native space
    command = format(f"python {scripts_dir}/manage_results/move_predictions_to_mgh.py --experiment-folder {exp_fold} --subjects_dir {FS_SUBJECTS_PATH} --list_ids {list_ids}")
    sub.check_call(command, shell=True)
    
    # Register prediction back to nifti volume
    output_dir = os.path.join(MELD_DATA_PATH,'input')
    command = format(f"bash {scripts_dir}/manage_results/register_back_to_xhemi.sh {FS_SUBJECTS_PATH} {list_ids} {output_dir}")
    sub.check_call(command, shell=True)
      
    # Create individual reports of each identified cluster
    command = format(f"python {scripts_dir}/manage_results/plot_prediction_report.py --experiment-folder {exp_fold} --subjects_dir {MELD_DATA_PATH} --list_ids {list_ids}")
    sub.check_call(command, shell=True)

    

    
