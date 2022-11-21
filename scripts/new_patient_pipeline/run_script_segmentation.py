## This script runs a FreeSurfer reconstruction on a participant
## Within your  MELD folder should be an input folder that contains folders 
## for each participant. Within each participant folder should be a T1 folder 
## that contains the T1 in nifti format ".nii" and where available a FLAIR 
## folder that contains the FLAIR in nifti format ".nii"

## To run : python run_script_segmentation.py -id <sub_id> -site <site_code>


import os
from tabnanny import verbose
import numpy as np
from sqlite3 import paramstyle
import sys
import argparse
import subprocess
# from subprocess import Popen, DEVNULL, STDOUT, check_call
import threading
import multiprocessing
from functools import partial
import tempfile
import glob
from meld_classifier.paths import BASE_PATH, MELD_DATA_PATH, FS_SUBJECTS_PATH, CLIPPING_PARAMS_FILE
import pandas as pd
from scripts.data_preparation.extract_features.create_xhemi import run_parallel_xhemi, create_xhemi
from scripts.data_preparation.extract_features.create_training_data_hdf5 import create_training_data_hdf5
from scripts.data_preparation.extract_features.sample_FLAIR_smooth_features import sample_flair_smooth_features
from scripts.data_preparation.extract_features.lesion_labels import lesion_labels
from scripts.data_preparation.extract_features.move_to_xhemi_flip import move_to_xhemi_flip
from meld_classifier.meld_cohort import MeldCohort
from meld_classifier.data_preprocessing import Preprocess
from os.path import join as opj
from meld_classifier.tools_commands_prints import get_m, run_command


def init(lock):
    global starting
    starting = lock



def fastsurfer_subject(subject, fs_folder, verbose=False):
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
        print(get_m(f'Freesurfer outputs already exists for subject {subject_id}. Freesurfer will be skipped', subject_id, 'STEP 1'))
        return

    # select inputs files T1 and FLAIR
    if subject_t1_path == '':
        # assume meld data structure
        subject_dir = opj(MELD_DATA_PATH, "input", subject_id)
        subject_t1_path = glob.glob(opj(subject_dir, "T1", "*T1*.nii*"))

        # check T1 and FLAIR exist
        if len(subject_t1_path) > 1:
            raise FileNotFoundError(
                get_m(f'Find too much volumes for T1. Check and remove the additional volumes with same key name', subject, 'ERROR'))
        elif not subject_t1_path:
            raise FileNotFoundError(get_m(f'Could not find T1 volume. Check if name follow the right nomenclature', subject,'ERROR'))
        else:
            subject_t1_path = subject_t1_path[0]

    # setup cortical segmentation command
    print(get_m(f'Segmentation using T1 only with FastSurfer', subject_id, 'INFO'))
    command = format(
        "$FASTSURFER_HOME/run_fastsurfer.sh --sd {} --sid {} --t1 {} --parallel --batch 1 --run_viewagg_on gpu".format(fs_folder, subject_id, subject_t1_path)
    )

    # call fastsurfer
    print(get_m('Start cortical parcellation (up to 2h). Please wait', subject_id, 'INFO'))
    print(get_m(f'Results will be stored in {fs_folder}', subject_id, 'INFO'))
    starting.acquire()  # no other process can get it until it is released
    proc = run_command(command, verbose=verbose) 
    threading.Timer(120, starting.release).start()  # release in two minutes
    proc.wait()
    print(get_m(f'Finished cortical parcellation', subject_id, 'INFO'))

def fastsurfer_flair(subject, fs_folder, verbose=False):
    #improve fastsurfer segmentation with FLAIR on 1 subject

    #TODO: enable BIDS format
    if type(subject) == dict:
        subject_id = subject['id']
        subject_flair_path = subject['flair_path']
    else:
        subject_id = subject
        subject_flair_path =''

    if os.path.isfile(opj(fs_folder, subject_id, "mri", "FLAIR.mgz")):
        print(get_m(f'Freesurfer outputs already exists. Freesurfer will be skipped', subject_id, 'STEP 1.1'))
        return

    if subject_flair_path == '':
        # get subject folder
        #assume meld data structure
        subject_dir = opj(MELD_DATA_PATH, "input", subject_id)
        subject_flair_path = glob.glob(opj(subject_dir, "FLAIR", "*FLAIR*.nii*"))

        if len(subject_flair_path) > 1:
            raise FileNotFoundError(
                get_m("Find too much volumes for FLAIR. Check and remove the additional volumes with same key name", subject_id, 'ERROR')
            )

        if not subject_flair_path:
            print(get_m('No FLAIR file has been found', subject_id, 'ERROR'))
            return 

        subject_flair_path = subject_flair_path[0]


    print(get_m("Starting FLAIRpial", subject_id, 'INFO'))
    command = format(
        "recon-all -sd {} -subject {} -FLAIR {} -FLAIRpial -autorecon3".format(fs_folder, subject_id, subject_flair_path)
    )
    # proc = Popen(command, shell=True, stdout = DEVNULL, stderr=STDOUT) 
    proc = run_command(command, verbose=verbose)
    proc.wait()
    print(get_m("Finished FLAIRpial", subject_id, "INFO"))

def freesurfer_subject(subject, fs_folder, verbose=False):
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
        print(get_m('Freesurfer outputs already exists. Freesurfer will be skipped', subject_id, "STEP 1"))
        return

    # select inputs files T1 and FLAIR
    if subject_t1_path == '':
        # assume meld data structure
        subject_dir = opj(MELD_DATA_PATH, "input", subject_id)
        subject_t1_path = glob.glob(opj(subject_dir, "T1", "*T1*.nii*"))
        # check T1 exists
        if len(subject_t1_path) > 1:
            raise FileNotFoundError(
                get_m('Find too much volumes for T1. Check and remove the additional volumes with same key name', subject_id, 'ERROR'))
        elif not subject_t1_path:
            raise FileNotFoundError(get_m('Could not find T1 volume. Check if name follow the right nomenclature', subject_id, 'ERROR'))
        else:
            subject_t1_path = subject_t1_path[0]

    if subject_flair_path == '':
        # assume meld data structure
        subject_dir = opj(MELD_DATA_PATH, "input", subject_id)
        subject_flair_path = glob.glob(opj(subject_dir, "FLAIR", "*FLAIR*.nii*"))
        # check FLAIR exists
        if len(subject_flair_path) > 1:
            raise FileNotFoundError(
                get_m('Find too much volumes for FLAIR. Check and remove the additional volumes with same key name', subject_id, 'ERROR'))
        elif not subject_flair_path:
            print(get_m('No FLAIR file has been found for subject', subject_id, 'INFO'))
            isflair = False
        else:
            subject_flair_path = subject_flair_path[0]
            isflair = True

    # setup cortical segmentation command
    if isflair == True:
        print(get_m('Segmentation using T1 and FLAIR with Freesurfer', subject_id, 'STEP 1'))
        command = format(
            "$FREESURFER_HOME/bin/recon-all -sd {} -s {} -i {} -FLAIR {} -FLAIRpial -all".format(
                fs_folder, subject_id, subject_t1_path, subject_flair_path
            )
        )
    else:
        print(get_m('Segmentation using T1 only with Freesurfer', subject_id, 'STEP 1'))
        command = format(
            "$FREESURFER_HOME/bin/recon-all -sd {} -s {} -i {} -all".format(fs_folder, subject_id, subject_t1_path)
        )

    # call Freesurfer
    print(get_m('Start cortical parcellation (up to 36h). Please wait', subject_id, 'INFO'))
    print(get_m(f'Results will be stored in {fs_folder}', subject_id, 'INFO'))
    starting.acquire()  # no other process can get it until it is released
    # proc = Popen(command, shell=True, stdout = DEVNULL, stderr=STDOUT)
    proc = run_command(command, verbose=verbose)
    if proc.returncode == 0 :
        print(get_m('Finished cortical parcellation', subject_id, 'INFO'))
    else:
        print(get_m('Something went wrong during segmentation. Check the recon-all log', subject_id, 'WARNING'))
    threading.Timer(120, starting.release).start()  # release in two minutes
    proc.wait()
    

def extract_features(subject_id, fs_folder, output_dir, verbose=False):
    # Launch script to extract surface-based features from freesurfer outputs
    print(get_m('Extract surface-based features', subject_id, 'INFO'))
    
    #### EXTRACT SURFACE-BASED FEATURES #####
    # Create the output directory to store the surface-based features processed
    os.makedirs(output_dir, exist_ok=True)
    
    #register to symmetric fsaverage xhemi
    print(get_m(f'Creating registration to template surface', subject_id, 'INFO'))
    create_xhemi(subject_id, fs_folder, verbose=verbose)

    #create basic features
    print(get_m(f'Sampling features in native space', subject_id, 'INFO'))
    sample_flair_smooth_features(subject_id, fs_folder, verbose=verbose)
       
    #move features and lesions to template
    print(get_m(f'Moving features to template surface', subject_id, 'INFO'))
    move_to_xhemi_flip(subject_id, fs_folder, verbose = verbose )
    
    print(get_m(f'Moving lesion masks to template surface', subject_id, 'INFO'))
    lesion_labels(subject_id, fs_folder, verbose=verbose)

    #create training_data matrix for all patients and controls.
    print(get_m(f'Creating final training data matrix', subject_id, 'INFO'))
    create_training_data_hdf5(subject_id, fs_folder, output_dir  )
     
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

    print(get_m(f'Start smoothing features', subject_ids, 'STEP 3'))


    if isinstance(subject_ids, str):
        subject_ids=[subject_ids]

    tmp = tempfile.NamedTemporaryFile(mode="w")
    create_dataset_file(subject_ids, tmp.name)

    c_raw = MeldCohort(hdf5_file_root="{site_code}_{group}_featurematrix.hdf5", dataset=tmp.name, data_dir=BASE_PATH)
    smoothing = Preprocess(c_raw, write_hdf5_file_root="{site_code}_{group}_featurematrix_smoothed.hdf5", data_dir=output_dir)
    
    #file to store subject with outliers vertices
    outliers_file=opj(output_dir, 'list_subject_extreme_vertices.csv')
    
    for feature in np.sort(list(set(features))):
        print(feature)
        smoothing.smooth_data(feature, features[feature], clipping_params=CLIPPING_PARAMS_FILE, outliers_file=outliers_file)

    tmp.close()
    
def run_subjects_segmentation_and_smoothing_parallel(subject_ids, num_procs=10, site_code="", use_fastsurfer=False, verbose=False):
    # parallel version of the pipeline, finish each stage for all subjects first

    ### SEGMENTATION ###
    ini_freesurfer = format("$FREESURFER_HOME/SetUpFreeSurfer.sh")
    proc = run_command(ini_freesurfer)
    proc.wait()

    ## Make a directory for the outputs
    fs_folder = FS_SUBJECTS_PATH
    os.makedirs(fs_folder, exist_ok=True)

    if use_fastsurfer:
        ## first processing stage with fastsurfer: segmentation
        pool = multiprocessing.Pool(processes=num_procs, initializer=init, initargs=[multiprocessing.Lock()])
        for _ in pool.imap_unordered(partial(fastsurfer_subject, fs_folder=fs_folder, verbose=verbose), subject_ids):
            pass

        ## flair pial correction
        pool = multiprocessing.Pool(processes=num_procs)
        for _ in pool.imap_unordered(partial(fastsurfer_flair, fs_folder=fs_folder, verbose=verbose), subject_ids):
            pass
    else:
        ## processing with freesurfer: segmentation
        pool = multiprocessing.Pool(processes=num_procs, initializer=init, initargs=[multiprocessing.Lock()])
        for _ in pool.imap_unordered(partial(freesurfer_subject, fs_folder=fs_folder, verbose=verbose), subject_ids):
            pass


    ### EXTRACT SURFACE-BASED FEATURES ###
    print(get_m(f'Extract surface-based features', subject_ids, 'STEP 2'))
    output_dir = opj(BASE_PATH, f"MELD_{site_code}")

    # parallelize create xhemi because it takes a while!
    print(get_m(f'Run create xhemi in parallel', subject_ids, 'INFO'))
    run_parallel_xhemi(subject_ids, fs_folder, num_procs=num_procs, verbose=verbose)

    # Launch script to extract features
    for subject in subject_ids:
        print(get_m(f'Extract features in hdf5', subject, 'INFO'))
        extract_features(subject, fs_folder=fs_folder, output_dir=output_dir, verbose=verbose)

    #### SMOOTH FEATURES #####
    #TODO: parallelise here
    for subject in subject_ids:
        smooth_features_new_subjects(subject, output_dir=output_dir)

def run_subject_segmentation_and_smoothing(subject, site_code="", use_fastsurfer=False, verbose=False):
    # pipeline to segment the brain, exract surface-based features and smooth features for 1 subject
    
    ### SEGMENTATION ###
    ini_freesurfer = format("$FREESURFER_HOME/SetUpFreeSurfer.sh")
    # check_call(ini_freesurfer, shell=True, stdout = DEVNULL, stderr=STDOUT)
    proc = run_command(ini_freesurfer)
    proc.wait()
    
    ## Make a directory for the outputs
    fs_folder = FS_SUBJECTS_PATH
    os.makedirs(fs_folder, exist_ok=True)

    if use_fastsurfer:
        ## first processing stage with fastsurfer: segmentation
        init(multiprocessing.Lock())
        fastsurfer_subject(subject,fs_folder, verbose=verbose)

        ## flair pial correction
        init(multiprocessing.Lock())
        fastsurfer_flair(subject,fs_folder, verbose=verbose)
    else:
        ## processing with freesurfer: segmentation
        init(multiprocessing.Lock())
        freesurfer_subject(subject,fs_folder, verbose=verbose)
    
    ### EXTRACT SURFACE-BASED FEATURES ###
    output_dir = opj(BASE_PATH, f"MELD_{site_code}")
    extract_features(subject, fs_folder=fs_folder, output_dir=output_dir, verbose=verbose)

    ### SMOOTH FEATURES ###
    smooth_features_new_subjects(subject, output_dir=output_dir)

def run_script_segmentation(site_code, list_ids=None,sub_id=None, use_parallel=False, use_fastsurfer=False, verbose=False ):
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
    
    if subject_id != None:
        #launch segmentation and feature extraction for 1 subject
        run_subject_segmentation_and_smoothing(subject_id,  site_code = site_code, use_fastsurfer = use_fastsurfer, verbose=verbose)
    else:
        if use_parallel:
            #launch segmentation and feature extraction in parallel
            print(get_m(f'Run subjects in parallel', None, 'INFO'))
            run_subjects_segmentation_and_smoothing_parallel(subject_ids, site_code = site_code, use_fastsurfer = use_fastsurfer, verbose=verbose)
        else:
            #launch segmentation and feature extraction for each subject one after another
            print(get_m(f'Run subjects one after another', None, 'INFO'))
            for subj in subject_ids:
                run_subject_segmentation_and_smoothing(subj,  site_code = site_code, use_fastsurfer = use_fastsurfer, verbose=verbose)

if __name__ == "__main__":
    # parse commandline arguments
    parser = argparse.ArgumentParser(description="perform cortical parcellation using recon-all from freesurfer")
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
    parser.add_argument("--fastsurfer", 
                        help="use fastsurfer instead of freesurfer", 
                        required=False, 
                        default=False,
                        action="store_true",
                        )
    parser.add_argument("--parallelise", 
                        help="parallelise segmentation", 
                        required=False,
                        default=False,
                        action="store_true",
                        )
    parser.add_argument("--debug_mode", 
                        help="mode to debug error", 
                        required=False,
                        default=False,
                        action="store_true",
                        )
    args = parser.parse_args()
    print(args)

    run_script_segmentation(
                        site_code = args.site_code,
                        list_ids=args.list_ids,
                        sub_id=args.id, 
                        use_parallel=args.parallelise, 
                        use_fastsurfer=args.fastsurfer,
                        verbose = args.debug_mode
                        )
    












