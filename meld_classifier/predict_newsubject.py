from meld_classifier.evaluation import Evaluator
from meld_classifier.experiment import Experiment, load_config
from meld_classifier.meld_cohort import MeldCohort
import argparse
import numpy as np
import json
import os
import importlib
import sys
import h5py
from meld_classifier.paths import BASE_PATH, EXPERIMENT_PATH, MELD_DATA_PATH, MODEL_PATH, MODEL_NAME
import subprocess
import pandas as pd

def create_dataset_file(subjects, output_path):
    df=pd.DataFrame()
    subjects_id = [subject for subject in subjects]
    df['subject_id']=subjects_id
    df['split']=['test' for subject in subjects]
    df.to_csv(output_path)
    return df

def predict_subjects(list_ids, new_data_parameters, plot_images = False, saliency=False):       
    #read subjects 
    subjects_ids = np.loadtxt(list_ids, dtype="str", ndmin=1)
    # create dataset csv
    create_dataset_file(subjects_ids, os.path.join(BASE_PATH, new_data_parameters['dataset']))
    # load models 
    experiment_path = os.path.join(EXPERIMENT_PATH, MODEL_PATH)
    exp = Experiment(experiment_path=experiment_path, experiment_name=MODEL_NAME)
    exp.init_logging()
    # load information to predict on new subjects
    exp.cohort = MeldCohort(
                            hdf5_file_root=new_data_parameters["hdf5_file_root"], dataset=new_data_parameters["dataset"]
                            )
    subject_ids = exp.cohort.get_subject_ids(**new_data_parameters, lesional_only=False)
    print(subject_ids)
    save_dir = new_data_parameters["saved_hdf5_dir"]
    #create sub-folders if do not exist
    os.makedirs(save_dir , exist_ok=True )
    os.makedirs(os.path.join(save_dir, "results"),  exist_ok=True)
    if plot_images:
        os.makedirs(os.path.join(save_dir, "results", "images"), exist_ok=True)
    # launch evaluation
    eva = Evaluator(exp, mode="inference", subject_ids=subject_ids, save_dir=save_dir)
    for subject in subject_ids:
        eva.load_predict_single_subject(
                subject, fold="", plot=plot_images, saliency=saliency, suffix=""
        )