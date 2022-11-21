# Import library
import os
import numpy as np
import json
from meld_classifier.meld_cohort import MeldCohort, MeldSubject
from meld_classifier.data_preprocessing import Preprocess, Feature
from meld_classifier.paths import BASE_PATH, NORM_CONTROLS_PARAMS_FILE, FINAL_SCALING_PARAMS
import time


site_codes = [
    "H2",
    "H3",
    "H4",
    "H5",
    "H6",
    "H7",
    "H9",
    "H10",
    "H11",
    "H12",
    "H14",
    "H15",
    "H16",
    "H17",
    "H18",
    "H19",
    "H21",
    "H23",
    "H24",
    "H26",
]

features = [
    ".on_lh.thickness.mgh",
    ".on_lh.w-g.pct.mgh",
    ".on_lh.pial.K_filtered.sm20.mgh",
    ".on_lh.sulc.mgh",
    ".on_lh.curv.mgh",
    ".on_lh.gm_FLAIR_0.25.mgh",
    ".on_lh.gm_FLAIR_0.5.mgh",
    ".on_lh.gm_FLAIR_0.75.mgh",
    ".on_lh.gm_FLAIR_0.mgh",
    ".on_lh.wm_FLAIR_0.5.mgh",
    ".on_lh.wm_FLAIR_1.mgh",
]
file_clip_params='clip_params_MELD_6.json'

# create cohort to smooth
cohort= MeldCohort(hdf5_file_root="{site_code}_{group}_featurematrix.hdf5", dataset='MELD_dataset_V6.csv')

"""get mean and std of all brain for the given cohort and save parameters"""
cohort_ids = cohort.get_subject_ids(group="both")

for feature in features:
    print(feature)
    # Give warning if list of controls empty
    if len(cohort_ids) == 0:
        print("WARNING: there is no subject in this cohort")
    vals_array = []
    included_subj = []
    for id_sub in cohort_ids:
        # create subject object
        subj = MeldSubject(id_sub, cohort=cohort)
        # append data to compute mean and std if feature exist and for FLAIR=0
        if subj.has_features(feature):
            # load feature's value for this subject
            vals_lh = subj.load_feature_values(feature, hemi="lh")
            vals_rh = subj.load_feature_values(feature, hemi="rh")
            vals = np.array(np.hstack([vals_lh[cohort.cortex_mask], vals_rh[cohort.cortex_mask]]))
            if (feature == ".on_lh.sulc.mgh") & (np.mean(vals, axis=0) > 0.2):
                vals = vals / 10
            vals_array.append(vals)
            included_subj.append(id_sub)                
    print("Compute mean and std from {} subject".format(len(included_subj)))
    # get min and max percentile
    vals_array = np.matrix(vals_array)
    min_p = np.percentile(vals_array.flatten(),0.1)
    print(f'min percentile {min_p}')
    print(f'vertices below min : {(vals_array.flatten()<min_p).sum()}')
    max_p = np.percentile(vals_array.flatten(),99.9)
    print(f'max percentile {max_p}')
    print(f'vertices above min : {(vals_array.flatten()>max_p).sum()}')
    # save in json
    data = {}
    data["{}".format(feature)] = {
        "min_percentile": str(min_p),
        "max_percentile": str(max_p),
    }
    # create or re-write json file
    file = os.path.join(BASE_PATH, file_clip_params)
    if os.path.isfile(file):
        # open json file and get dictionary
        with open(file, "r") as f:
            x = json.loads(f.read())
        # update dictionary with new dataset version
        x.update(data)
    else:
        x = data
    # save dictionary in json file
    with open(file, "w") as outfile:
        json.dump(x, outfile, indent=4)
    print(f"parameters saved in {file}")