## This script runs a FreeSurfer reconstruction on a participant
## Within your  MELD folder should be an input folder that contains folders 
## for each participant. Within each participant folder should be a T1 folder 
## that contains the T1 in nifti format ".nii" and where available a FLAIR 
## folder that contains the FLAIR in nifti format ".nii"

## To run : python new_pt_pipeline_script1.py -id <sub_id> -site <site_code>


import os
from sqlite3 import paramstyle
import sys
import argparse
import subprocess as sub
import threading
import multiprocessing
from functools import partial
import glob
import tempfile
from meld_classifier.paths import BASE_PATH, SCRIPTS_DIR, MELD_DATA_PATH, FS_SUBJECTS_PATH
import pandas as pd
from scripts.data_preparation.extract_features.create_xhemi import run_parallel


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
    if os.path.isdir(os.path.join(fs_folder, subject_id)):
        print(f"STEP 1:Freesurfer outputs already exists for subject {subject_id}. \nFreesurfer will be skipped")
        return

    # select inputs files T1 and FLAIR
    if subject_t1_path == '':
        # assume meld data structure
        subject_dir = os.path.join(MELD_DATA_PATH, "input", subject_id)
        subject_t1_path = glob.glob(os.path.join(subject_dir, "T1", "*T1*.nii*"))

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
    print("STEP 1: Segmentation using T1 only with FastSurfer")
    command = format(
        "$FASTSURFER_HOME/run_fastsurfer.sh --sd {} --sid {} --t1 {} --parallel".format(fs_folder, subject_id, subject_t1_path)
    )

    # call fastsurfer
    print(f"INFO : Start cortical parcellation for {subject_id} (up to 2h). Please wait")
    print(f"INFO : Results will be stored in {fs_folder}")
    starting.acquire()  # no other process can get it until it is released
    try:
        proc = sub.check_call(command, shell=True, stdout=sub.DEVNULL)
    except sub.CalledProcessError as e:
        print(f'ERROR STEP 1 : Segmentation has failed for {subject_id}')
        return    
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

    if os.path.isfile(os.path.join(fs_folder, subject_id, "mri", "FLAIR.mgz")):
        print(f"STEP 1.2: Freesurfer outputs already exists for subject {subject_id}. \nFreesurfer will be skipped")
        return

    if subject_flair_path == '':
        # get subject folder
        #assume meld data structure
        subject_dir = os.path.join(MELD_DATA_PATH, "input", subject_id)
        subject_flair_path = glob.glob(os.path.join(subject_dir, "FLAIR", "*FLAIR*.nii*"))

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
    try:
        proc = sub.check_call(command, shell=True, stdout=sub.DEVNULL)
    except sub.CalledProcessError as e:
        print(f'ERROR STEP 1.2 : Segmentation has failed for {subject_id}')
        return 
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
    if os.path.isdir(os.path.join(fs_folder, subject_id)):
        print(f"STEP 1: Freesurfer outputs already exists for subject {subject_id}. \nFreesurfer will be skipped")
        return

    # select inputs files T1 and FLAIR
    if subject_t1_path == '':
        # assume meld data structure
        subject_dir = os.path.join(MELD_DATA_PATH, "input", subject_id)
        subject_t1_path = glob.glob(os.path.join(subject_dir, "T1", "*T1*.nii*"))
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
        subject_dir = os.path.join(MELD_DATA_PATH, "input", subject_id)   
        subject_flair_path = glob.glob(os.path.join(subject_dir, "FLAIR", "*FLAIR*.nii*"))
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
        print("STEP 1: Segmentation using T1 and FLAIR with Freesurfer for {subject_id}")
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
    try:
        proc = sub.check_call(command, shell=True, stdout=sub.PIPE)
    except sub.CalledProcessError as e:
        print(f'ERROR STEP 1 : Segmentation has failed for {subject_id}')
        return    
    threading.Timer(120, starting.release).start()  # release in two minutes
    proc.wait()
    print(f"INFO : Finished cortical parcellation for {subject_id} !")

def extract_features(subject, scripts_dir, fs_folder, site_code=""):
    # Launch script to extract surface-based features from freesurfer outputs
    
        #TODO: enable BIDS format
    if type(subject) == dict:
        subject_id = subject['id']
    else:
        subject_id = subject
    
    print("STEP 2: Extract surface-based features", subject_id)
    if site_code == "":
        try:
            site_code = subject_id.split("_")[1]  ##according to current MELD naming convention TODO
        except ValueError:
            print("Could not recover site code from", subject_id)
            sys.exit(-1)
    #### EXTRACT SURFACE-BASED FEATURES #####
    # Create the output directory to store the surface-based features processed
    output_dir = os.path.join(BASE_PATH, f"MELD_{site_code}")
    os.makedirs(output_dir, exist_ok=True)
    # Create temporary list of ids
    tmp = tempfile.NamedTemporaryFile(mode="w")
    with open(tmp.name, 'w') as f:
        f.write(subject_id) 
    
    command = format(
        f"bash {scripts_dir}/data_preparation/meld_pipeline.sh {fs_folder} {site_code} {tmp.name} {scripts_dir}/data_preparation/extract_features {output_dir}"
    )
    try:
        sub.check_call(command, shell=True)  # ,stdout=sub.DEVNULL)
    except sub.CalledProcessError as e:
        print(f'ERROR STEP 2 : feature extraction has failed for {subject_id}')
        return
     
    tmp.close()
    # os.remove(tmp.name)

def smooth_features(subject, scripts_dir):
    
    #TODO: enable BIDS format
    if type(subject) == dict:
        subject_id = subject['id']

    else:
        subject_id = subject

    # Create temporary list of ids
    tmp = tempfile.NamedTemporaryFile(mode="w")
    with open(tmp.name, 'w') as f:
        f.write(subject_id) 
        
    # Launch script to smooth features
    print("STEP 3: SMOOTH FEATURES")
    command = format(
        f"python {scripts_dir}/data_preparation/run_data_smoothing_new_subjects.py -ids {tmp.name} -d {BASE_PATH}"
    )
    try:
        sub.check_call(command, shell=True, stdout=sub.DEVNULL)  
    except sub.CalledProcessError as e:
        print(f'ERROR STEP 3 : smoothing has failed for {subject_id}')
        return 
    tmp.close()
    # os.remove(subject_ids.name)

def run_subjects_segmentation_and_smoothing_parallel(subject_list, num_procs=20, site_code="", use_fastsurfer=False):
    # parallel version of the pipeline, finish each stage for all subjects first

    ### SEGMENTATION ###
    ini_freesurfer = format("$FREESURFER_HOME/SetUpFreeSurfer.sh")
    sub.check_call(ini_freesurfer, shell=True, stdout=sub.DEVNULL)

    ## Make a directory for the outputs
    fs_folder = FS_SUBJECTS_PATH
    os.makedirs(fs_folder, exist_ok=True)
    arguments = []

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
    scripts_dir = os.path.join(SCRIPTS_DIR, "scripts")

    # parallelize create xhemi because it takes a while!
    run_parallel(subject_list, fs_folder, num_procs=num_procs)

    # Launch script to extract features
    for subject in subject_list:
        extract_features(subject, scripts_dir=scripts_dir, fs_folder=fs_folder, site_code=site_code)

    #### SMOOTH FEATURES #####
    # Launch script to smooth features
    ###TODO: parallelize here
    for subject in subject_list:
        smooth_features(subject, scripts_dir=scripts_dir,)

def run_subject_segmentation_and_smoothing(subject, site_code="", use_fastsurfer=False):
    # pipeline to segment the brain, exract surface-based features and smooth features for 1 subject

    ### SEGMENTATION ###
    ini_freesurfer = format("$FREESURFER_HOME/SetUpFreeSurfer.sh")
    sub.check_call(ini_freesurfer, shell=True, stdout=sub.DEVNULL)

    ## Make a directory for the outputs
    fs_folder = FS_SUBJECTS_PATH
    os.makedirs(fs_folder, exist_ok=True)
    arguments = []

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
    scripts_dir = os.path.join(SCRIPTS_DIR, "scripts")

    # Launch script to extract features
    extract_features(subject, scripts_dir=scripts_dir, fs_folder=fs_folder, site_code=site_code)

    #### SMOOTH FEATURES #####
    # Launch script to smooth features
    smooth_features(subject, scripts_dir=scripts_dir,)


if __name__ == "__main__":
    # parse commandline arguments
    parser = argparse.ArgumentParser(description="perform cortical parcellation using recon-all from freesurfer")
    parser.add_argument(
        "-id",
        "--id_subj",
        help="Subject ID.",
        default="",
        required=False,
    )
    parser.add_argument(
        "-sl",
        "--subject_list",
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
        "-fs", 
        "--fastsurfer", 
        help="use fastsurfer instead of freesurfer", 
        required=False, 
        default=False,
        action="store_true",
    )
    parser.add_argument(
        "-p", 
        "--parallel", 
        help="parallelise segmentation", 
        required=False,
        default=False,
        action="store_true",
        )
    args = parser.parse_args()
    subject = str(args.id_subj)
    site_code = str(args.site_code)
    subject_list = str(args.subject_list)
    use_fastsurfer = args.fastsurfer
    use_parallel = args.parallel
    print(args)

    if subject_list != "":
        if subject != "":
            print("Ignoring  subject id because list provided...")
        try:
            sub_list_df = pd.read_csv(subject_list)
        except ValueError:
            print("Could not open, subject_list")
            sys.exit(-1)
        if use_parallel:   
            print('Run subjects in parallel') 
            run_subjects_segmentation_and_smoothing_parallel(list(sub_list_df.participant_id.values), site_code = site_code, use_fastsurfer = use_fastsurfer)
        else:
            print('Run subjects one after another')
            for subject in list(sub_list_df.participant_id.values):
                run_subject_segmentation_and_smoothing(subject,  site_code = site_code, use_fastsurfer = use_fastsurfer)
    else:
        if subject == "":
            print("Please specify both subject and site_code...")
        else:
            run_subject_segmentation_and_smoothing(subject,  site_code = site_code, use_fastsurfer = use_fastsurfer)
















