""" This script run the data preprocessing on a whole cohort using Preprocess class 
Whole script has 3 steps
1. smooth data - write to new hdf5 file
2a. new object based on raw data to find outliers & write in csv
2b. new object with cohort based on smooth hdf5 - outlier & combat, write to new hdf5 file
3. new object with combat hdf5 normalisation steps, adds these files to the same combat hdf5.
"""

# Import library
import os
import numpy as np
from meld_classifier.meld_cohort import MeldCohort, MeldSubject
from meld_classifier.data_preprocessing import Preprocess, Feature
from meld_classifier.paths import BASE_PATH, NORM_CONTROLS_PARAMS_FILE, FINAL_SCALING_PARAMS
import time

print(time.asctime(time.localtime(time.time())))
start_time = time.clock()


# Initialisation
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
feat = Feature()
features_smooth = [feat.smooth_feat(feature, features[feature]) for feature in features]
features_combat = [feat.combat_feat(feature) for feature in features_smooth]

### SMOOTH DATA ###
# -----------------------------------------------------------------------------------------------
print("PROCESS 1 : SMOOTHING")
# create cohort to smooth
c_prepro = MeldCohort(hdf5_file_root="{site_code}_{group}_featurematrix.hdf5", dataset=None)
# create object smoothing
smoothing = Preprocess(
    c_prepro,
    site_codes=site_codes,
    write_hdf5_file_root="{site_code}_{group}_featurematrix_smoothed_6.hdf5",
    data_dir=BASE_PATH,
)

# call function to smooth features
for feature in np.sort(list(set(features))):
    print(feature)
    smoothing.smooth_data(feature, features[feature])

# Transfer lesions in the new hdf5
smoothing.transfer_lesion()


### FIND OUTLIERS ###
# -----------------------------------------------------------------------------------------------
print("PROCESS 2a : find outliers")
# create cohort to find outlier
c_preproc = MeldCohort(hdf5_file_root="{site_code}_{group}_featurematrix.hdf5", dataset=None)
# create object
outliers = Preprocess(c_preproc, site_codes=site_codes, write_hdf5_file_root=None, data_dir=BASE_PATH)
# call function to find and save outliers
features = [
    ".on_lh.curv.mgh",
    ".on_lh.gm_FLAIR_0.5.mgh",
    ".on_lh.pial.K_filtered.sm20.mgh",
    ".on_lh.sulc.mgh",
    ".on_lh.thickness.mgh",
    ".on_lh.w-g.pct.mgh",
    ".on_lh.wm_FLAIR_0.5.mgh",
]
list_outliers = outliers.find_outliers(features, "list_outliers_qc_6.csv")


### COMBAT DATA ###
# -----------------------------------------------------------------------------------------------
print("PROCESS 2b : COMBAT")

# create cohort to combat
c_smooth = MeldCohort(hdf5_file_root="{site_code}_{group}_featurematrix_smoothed_6.hdf5", dataset=None)
# create object combat
combat = Preprocess(
    c_smooth,
    site_codes=site_codes,
    write_hdf5_file_root="{site_code}_{group}_featurematrix_combat_6_shrink.hdf5",
    data_dir=BASE_PATH,
)
# call function to combat data
for feature in features_smooth:
    print(feature)
    combat.combat_whole_cohort(
        feature, outliers_file="list_outliers_qc_6.csv", combat_params_file=os.path.join(BASE_PATH,"Combat_parameters_6_shrink.hdf5")
    )

# Transfer lesions in the new hdf5
combat.transfer_lesion()


###  INTRA, INTER & ASYMETRY ###
# -----------------------------------------------------------------------------------------------
print("PROCESS 3 : INTRA, INTER & ASYMETRY")

# create cohort to normalise
c_combat = MeldCohort(hdf5_file_root="{site_code}_{group}_featurematrix_combat_6.hdf5", dataset=None)
# create cohort of controls for inter normalisation if differente
c_controls = MeldCohort(hdf5_file_root="{site_code}_{group}_featurematrix_combat_6.hdf5", dataset=None)
# create object normalisation
norm = Preprocess(
    c_combat,
    site_codes=site_codes,
    write_hdf5_file_root="{site_code}_{group}_featurematrix_combat_6.hdf5",
    data_dir=BASE_PATH,
)
# call functions to normalise data
for feature in features_combat:
    print(feature)
    #save parameters of normalisation by controls 
#     norm.compute_mean_std_controls(feature, cohort=c_controls, params_norm=os.path.join(BASE_PATH, NORM_CONTROLS_PARAMS_FILE))
#     norm.compute_mean_std_controls(feature, cohort=c_controls, asym= True, params_norm=os.path.join(BASE_PATH, NORM_CONTROLS_PARAMS_FILE))
    # intra-inter normalise and assymetry
    norm.intra_inter_subject(feature)
    norm.asymmetry_subject(feature)
    
end_time = time.clock()
print("execution time = " + str(end_time - start_time) + " seconds")
print(time.asctime(time.localtime(time.time())))


# ### CREATE BOUNDARIES ###
# #-----------------------------------------------------------------------------------------------
print("PROCESS 4 : BOUNDARIES")

# create cohort to combat
c = MeldCohort(hdf5_file_root="{site_code}_{group}_featurematrix_combat_6.hdf5", dataset=None)
# create object combat
boundary = Preprocess(
    c, site_codes=site_codes, write_hdf5_file_root="{site_code}_{group}_featurematrix_combat_6.hdf5", data_dir=BASE_PATH
)

# Create boundaries
boundary.make_boundary_zones()

# ### SAVE PARAMETERS FOR NORMALISATION BEFORE CLASSIFIER###
# #-----------------------------------------------------------------------------------------------
print("PROCESS 5 : GET FINAL PARAMETER FEATURE SCALING ")

# create cohort to combat
c = MeldCohort(hdf5_file_root="{site_code}_{group}_featurematrix_combat_6.hdf5", dataset=None)
# create object combat
normalise = Preprocess(
    c, site_codes=site_codes, write_hdf5_file_root=FINAL_SCALING_PARAMS, data_dir=BASE_PATH
)
# get all the features present for this cohort
features = c.get_features(features_to_exclude=[""])
# for each feature save the parameter in the file provided
for feature in sorted(features):
    print(feature)
    normalise.compute_mean_std(feature, c)
