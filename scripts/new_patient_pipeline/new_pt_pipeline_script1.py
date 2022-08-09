## This script runs a FreeSurfer reconstruction on a participant
## Within your  MELD folder should be an input folder that contains folders 
## for each participant. Within each participant folder should be a T1 folder 
## that contains the T1 in nifti format ".nii" and where available a FLAIR 
## folder that contains the FLAIR in nifti format ".nii"

## To run : python new_pt_pipeline_script1.py -id <sub_id> -site <site_code>


import os
import numpy as np
from sqlite3 import paramstyle
import sys
import argparse
import subprocess as sub
import threading
import multiprocessing
from functools import partial
import tempfile
import glob
from meld_classifier.paths import BASE_PATH, SCRIPTS_DIR, MELD_DATA_PATH, FS_SUBJECTS_PATH, CLIPPING_PARAMS_FILE
import pandas as pd
from scripts.data_preparation.extract_features.create_xhemi import run_parallel_xhemi, create_xhemi
from scripts.data_preparation.extract_features.create_training_data_hdf5 import create_training_data_hdf5
from scripts.data_preparation.extract_features.sample_FLAIR_smooth_features import sample_flair_smooth_features
from scripts.data_preparation.extract_features.lesion_labels import lesion_labels
from scripts.data_preparation.extract_features.move_to_xhemi_flip import move_to_xhemi_flip
from meld_classifier.meld_cohort import MeldCohort
from meld_classifier.data_preprocessing import Preprocess
from os.path import join as opj

def init(lock):
    global starting
    starting = lock

def fastsurfer_subject(subject, fs_folder):
    # run fastsurfer segmentation on 1 subject

    #TODO: enable BIDS format
    if type(subject) == dict:
        subject_id = subject['id']
        subject_t1_path = subject['t1_path']
    else:
        subject_id = subject
        subject_t1_path =''

    # get subject folder
    # if freesurfer outputs already exist for this subject, continue running from where it stopped
    # else, find inputs T1 and FLAIR and run FS
    if os.path.isdir(opj(fs_folder, subject_id)):
        print(f"STEP 1: Freesurfer outputs already exists for subject {subject_id}. Freesurfer will be skipped")
        return

    # select inputs files T1 and FLAIR
    if subject_t1_path == '':
        # assume meld data structure
        subject_dir = opj(MELD_DATA_PATH, "input", subject_id)
        subject_t1_path = glob.glob(opj(subject_dir, "T1", "*T1*.nii*"))

        # check T1 and FLAIR exist
        if len(subject_t1_path) > 1:
            raise FileNotFoundError(
                "Find too much volumes for T1. Check and remove the additional volumes with same key name"
            )
        elif not subject_t1_path:
            raise FileNotFoundError(f"Could not find T1 volume. Check if name follow the right nomenclature")
        else:
            subject_t1_path = subject_t1_path[0]

    # setup cortical segmentation command
    print(f"STEP 1: Segmentation using T1 only with FastSurfer for {subject_id}")
    command = format(
        "$FASTSURFER_HOME/run_fastsurfer.sh --sd {} --sid {} --t1 {} --parallel --batch 1 --run_viewagg_on gpu".format(fs_folder, subject_id, subject_t1_path)
    )

    # call fastsurfer
    print(f"INFO : Start cortical parcellation for {subject_id} (up to 2h). Please wait")
    print(f"INFO : Results will be stored in {fs_folder}")
    starting.acquire()  # no other process can get it until it is released
    proc = sub.Popen(command, shell=True, stdout=sub.DEVNULL)  
    threading.Timer(120, starting.release).start()  # release in two minutes
    proc.wait()
    print(f"INFO : Finished cortical parcellation for {subject_id} !")

def fastsurfer_flair(subject, fs_folder):
    #improve fastsurfer segmentation with FLAIR on 1 subject

    #TODO: enable BIDS format
    if type(subject) == dict:
        subject_id = subject['id']
        subject_flair_path = subject['flair_path']
    else:
        subject_id = subject
        subject_flair_path =''

    if os.path.isfile(opj(fs_folder, subject_id, "mri", "FLAIR.mgz")):
        print(f"STEP 1.2: Freesurfer outputs already exists for subject {subject_id}. \nFreesurfer will be skipped")
        return

    if subject_flair_path == '':
        # get subject folder
        #assume meld data structure
        subject_dir = opj(MELD_DATA_PATH, "input", subject_id)
        subject_flair_path = glob.glob(opj(subject_dir, "FLAIR", "*FLAIR*.nii*"))

        if len(subject_flair_path) > 1:
            raise FileNotFoundError(
                "Find too much volumes for FLAIR. Check and remove the additional volumes with same key name"
            )

        if not subject_flair_path:
            print("No FLAIR file has been found for subject", subject_id)
            return 

        subject_flair_path = subject_flair_path[0]


    print("Starting FLAIRpial for subject", subject_id)
    command = format(
        "recon-all -sd {} -subject {} -FLAIR {} -FLAIRpial -autorecon3".format(fs_folder, subject_id, subject_flair_path)
    )
    proc = sub.Popen(command, shell=True, stdout=sub.DEVNULL)  
    proc.wait()
    print("finished FLAIRpial for subject", subject_id)

def freesurfer_subject(subject, fs_folder):
    #run freesurfer recon-all segmentation on 1 subject

    #TODO: enable BIDS format
    if type(subject) == dict:
        subject_id = subject['id']
        subject_t1_path = subject['t1_path']
        subject_flair_path = subject['flair_path']
    else:
        subject_id = subject
        subject_t1_path =''
        subject_flair_path =''

    # get subject folder
    # If freesurfer outputs already exist for this subject, continue running from where it stopped
    # Else, find inputs T1 and FLAIR and run FS
    if os.path.isdir(opj(fs_folder, subject_id)):
        print(f"STEP 1: Freesurfer outputs already exists for subject {subject_id}. \nFreesurfer will be skipped")
        return

    # select inputs files T1 and FLAIR
    if subject_t1_path == '':
        # assume meld data structure
        subject_dir = opj(MELD_DATA_PATH, "input", subject_id)
        subject_t1_path = glob.glob(opj(subject_dir, "T1", "*T1*.nii*"))
        # check T1 exists
        if len(subject_t1_path) > 1:
            raise FileNotFoundError(
                "Find too much volumes for T1. Check and remove the additional volumes with same key name"
            )
        elif not subject_t1_path:
            raise FileNotFoundError(f"Could not find T1 volume. Check if name follow the right nomenclature")
        else:
            subject_t1_path = subject_t1_path[0]

    if subject_flair_path == '':
        # assume meld data structure
        subject_dir = opj(MELD_DATA_PATH, "input", subject_id)   
        subject_flair_path = glob.glob(opj(subject_dir, "FLAIR", "*FLAIR*.nii*"))
        # check FLAIR exists
        if len(subject_flair_path) > 1:
            raise FileNotFoundError(
                "Find too much volumes for FLAIR. Check and remove the additional volumes with same key name"
            )
        elif not subject_flair_path:
            print("No FLAIR file has been found for subject", subject_id)
            isflair = False
        else:
            subject_flair_path = subject_flair_path[0]
            isflair = True

    # setup cortical segmentation command
    if isflair == True:
        print(f"STEP 1: Segmentation using T1 and FLAIR with Freesurfer for {subject_id}")
        command = format(
            "$FREESURFER_HOME/bin/recon-all -sd {} -s {} -i {} -FLAIR {} -FLAIRpial -all".format(
                fs_folder, subject_id, subject_t1_path, subject_flair_path
            )
        )
    else:
        print("STEP 1: Segmentation using T1 only with Freesurfer for {subject_id}")
        command = format(
            "$FREESURFER_HOME/bin/recon-all -sd {} -s {} -i {} -all".format(fs_folder, subject_id, subject_t1_path)
        )

    # call Freesurfer
    print(f"INFO : Start cortical parcellation for {subject_id} (up to 36h). Please wait")
    print(f"INFO : Results will be stored in {fs_folder}")
    starting.acquire()  # no other process can get it until it is released
    proc = sub.Popen(command, shell=True, stdout=sub.DEVNULL)  
    threading.Timer(120, starting.release).start()  # release in two minutes
    proc.wait()
    print(f"INFO : Finished cortical parcellation for {subject_id} !")

def extract_features(subject_id, fs_folder, output_dir):
    # Launch script to extract surface-based features from freesurfer outputs
    print("STEP 2: Extract surface-based features", subject_id)
    
    #### EXTRACT SURFACE-BASED FEATURES #####
    # Create the output directory to store the surface-based features processed
    os.makedirs(output_dir, exist_ok=True)
    
    #register to symmetric fsaverage xhemi
    print("INFO: Creating registration to template surface")
    create_xhemi(subject_id, fs_folder)

    #create basic features
    print("INFO: Sampling features in native space")
    sample_flair_smooth_features(subject_id, fs_folder)
       
    #move features and lesions to template
    print("INFO: Moving features to template surface")
    move_to_xhemi_flip(subject_id, fs_folder)
    
    print("INFO: Moving lesion masks to template surface")
    lesion_labels(subject_id, fs_folder)

    #create training_data matrix for all patients and controls.
    print("INFO: Creating final training data matrix")
    create_training_data_hdf5(subject_id, fs_folder, output_dir )
     
def create_dataset_file(subjects, output_path):
    df=pd.DataFrame()
    subjects_id = [subject for subject in subjects]
    df['subject_id']=subjects_id
    df['split']=['test' for subject in subjects]
    df.to_csv(output_path)

def smooth_features_new_subjects(subject_ids, output_dir):
    features = {
        ".on_lh.thickness.mgh": 10,
        ".on_lh.w-g.pct.mgh": 10,
        ".on_lh.pial.K_filtered.sm20.mgh": None,
        ".on_lh.sulc.mgh": 5,
        ".on_lh.curv.mgh": 5,
        ".on_lh.gm_FLAIR_0.25.mgh": 10,
        ".on_lh.gm_FLAIR_0.5.mgh": 10,
        ".on_lh.gm_FLAIR_0.75.mgh": 10,
        ".on_lh.gm_FLAIR_0.mgh": 10,
        ".on_lh.wm_FLAIR_0.5.mgh": 10,
        ".on_lh.wm_FLAIR_1.mgh": 10,
    }

    if isinstance(subject_ids, str):
        subject_ids=[subject_ids]

    tmp = tempfile.NamedTemporaryFile(mode="w")
    create_dataset_file(subject_ids, tmp.name)
    
    c_raw = MeldCohort(hdf5_file_root="{site_code}_{group}_featurematrix.hdf5", dataset=tmp.name, data_dir=BASE_PATH)
    smoothing = Preprocess(c_raw, write_hdf5_file_root="{site_code}_{group}_featurematrix_smoothed.hdf5", data_dir=output_dir)
    
    print('INFO: smoothing features')
    for feature in np.sort(list(set(features))):
        print(feature)
        smoothing.smooth_data(feature, features[feature], clipping_params=CLIPPING_PARAMS_FILE)

    tmp.close()
    
def run_subjects_segmentation_and_smoothing_parallel(subject_list, num_procs=20, site_code="", use_fastsurfer=False):
    # parallel version of the pipeline, finish each stage for all subjects first

    ### SEGMENTATION ###
    ini_freesurfer = format("$FREESURFER_HOME/SetUpFreeSurfer.sh")
    sub.check_call(ini_freesurfer, shell=True, stdout=sub.DEVNULL)

    ## Make a directory for the outputs
    fs_folder = FS_SUBJECTS_PATH
    os.makedirs(fs_folder, exist_ok=True)

    if use_fastsurfer:
        ## first processing stage with fastsurfer: segmentation
        pool = multiprocessing.Pool(processes=num_procs, initializer=init, initargs=[multiprocessing.Lock()])
        for _ in pool.imap_unordered(partial(fastsurfer_subject, fs_folder=fs_folder), subject_list):
            pass

        ## flair pial correction
        pool = multiprocessing.Pool(processes=num_procs)
        for _ in pool.imap_unordered(partial(fastsurfer_flair, fs_folder=fs_folder), subject_list):
            pass
    else:
        ## processing with freesurfer: segmentation
        pool = multiprocessing.Pool(processes=num_procs, initializer=init, initargs=[multiprocessing.Lock()])
        for _ in pool.imap_unordered(partial(freesurfer_subject, fs_folder=fs_folder), subject_list):
            pass


    ### EXTRACT SURFACE-BASED FEATURES ###
    output_dir = opj(BASE_PATH, f"MELD_{site_code}")

    # parallelize create xhemi because it takes a while!
    run_parallel_xhemi(subject_list, fs_folder, num_procs=num_procs)

    # Launch script to extract features
    for subject in subject_list:
        extract_features(subject, fs_folder=fs_folder, output_dir=output_dir)

    #### SMOOTH FEATURES #####
    #TODO: parallelise here
    for subject in subject_list:
        smooth_features_new_subjects(subject, output_dir=output_dir)

def run_subject_segmentation_and_smoothing(subject, site_code="", use_fastsurfer=False):
    # pipeline to segment the brain, exract surface-based features and smooth features for 1 subject
    
    ### SEGMENTATION ###
    ini_freesurfer = format("$FREESURFER_HOME/SetUpFreeSurfer.sh")
    sub.check_call(ini_freesurfer, shell=True, stdout=sub.DEVNULL)

    ## Make a directory for the outputs
    fs_folder = FS_SUBJECTS_PATH
    os.makedirs(fs_folder, exist_ok=True)

    if use_fastsurfer:
        ## first processing stage with fastsurfer: segmentation
        init(multiprocessing.Lock())
        fastsurfer_subject(subject,fs_folder)

        ## flair pial correction
        init(multiprocessing.Lock())
        fastsurfer_flair(subject,fs_folder)
    else:
        ## processing with freesurfer: segmentation
        init(multiprocessing.Lock())
        freesurfer_subject(subject,fs_folder)
    
    ### EXTRACT SURFACE-BASED FEATURES ###
    output_dir = opj(BASE_PATH, f"MELD_{site_code}")
    extract_features(subject, fs_folder=fs_folder, output_dir=output_dir)

    ### SMOOTH FEATURES ###
    smooth_features_new_subjects(subject, output_dir=output_dir)


if __name__ == "__main__":
    # parse commandline arguments
    parser = argparse.ArgumentParser(description="perform cortical parcellation using recon-all from freesurfer")
    parser.add_argument(
        "-id",
        "--id",
        help="Subject ID.",
        default="",
        required=False,
    )
    parser.add_argument(
        "-ids",
        "--list_ids",
        default="",
        help="Relative path to subject List containing id and site_code.",
        required=False,
    )
    parser.add_argument(
        "-site",
        "--site_code",
        help="Site code",
        default="",
        required=True,
    )
    parser.add_argument(
        "--fastsurfer", 
        help="use fastsurfer instead of freesurfer", 
        required=False, 
        default=False,
        action="store_true",
    )
    parser.add_argument(
        "--parallelise", 
        help="parallelise segmentation", 
        required=False,
        default=False,
        action="store_true",
        )
    args = parser.parse_args()
    site_code = str(args.site_code)
    use_fastsurfer = args.fastsurfer
    use_parallel = args.parallelise
    subject_id=None
    subject_ids=None
    print(args)

    if args.list_ids:
        try:
            sub_list_df=pd.read_csv(args.list_ids)
            subject_ids=np.array(sub_list_df.ID.values)
        except:
            subject_ids=np.array(np.loadtxt(args.list_ids, dtype='str', ndmin=1)) 
        else:
                print(f"ERROR: Could not open {subject_ids}")
                sys.exit(-1)                
    elif args.id:
        subject_id=args.id
        subject_ids=np.array([args.id])
    else:
        print('ERROR: No ids were provided')
        print("ERROR: Please specify both subject(s) and site_code ...")
        sys.exit(-1) 
    
    if subject_id != None:
        #launch segmentation and feature extraction for 1 subject
        run_subject_segmentation_and_smoothing(subject_id,  site_code = site_code, use_fastsurfer = use_fastsurfer)
    else:
        if use_parallel:
            #launch segmentation and feature extraction in parallel
            print('Run subjects in parallel') 
            run_subjects_segmentation_and_smoothing_parallel(subject_ids, site_code = site_code, use_fastsurfer = use_fastsurfer)
        else:
            #launch segmentation and feature extraction for each subject one after another
            print('Run subjects one after another')
            for subj in subject_ids:
                run_subject_segmentation_and_smoothing(subj,  site_code = site_code, use_fastsurfer = use_fastsurfer)

    # if list_ids!="":
    #     if id != "":
    #         print("Ignoring  subject id because list provided...")
    #     try:
    #         sub_list_df = pd.read_csv(list_ids)
    #         subject_ids=np.array(sub_list_df.ID.values)
    #     except:
    #         subject_ids=np.array(np.loadtxt(list_ids, dtype='str', ndmin=1))
    #     else:
    #         print("Could not open, subject_list")
    #         sys.exit(-1)     
    #     if use_parallel:
    #         #launch segmentation and feature extraction in parallel
    #         print('Run subjects in parallel') 
    #         run_subjects_segmentation_and_smoothing_parallel(subject_ids, site_code = site_code, use_fastsurfer = use_fastsurfer)
    #     else:
    #         #launch segmentation and feature extraction for each subject one after another
    #         print('Run subjects one after another')
    #         for subject_id in subject_ids:
    #             run_subject_segmentation_and_smoothing(subject_id,  site_code = site_code, use_fastsurfer = use_fastsurfer)
    # else:
    #     if id == "":
    #         print("Please specify both subject and site_code...")
    #     else:
    #         #launch segmentation and feature extraction for 1 subject
    #         subject_id=id
    #         run_subject_segmentation_and_smoothing(subject_id,  site_code = site_code, use_fastsurfer = use_fastsurfer)
















