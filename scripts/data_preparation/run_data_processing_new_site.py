""" This script run the data preprocessing on a new site using Preprocess class 
Whole script has 3 steps
1. smooth data - write to new hdf5 file
2. combat the new site with a reference cohort
3. intranormalise, internormlise new site by controls from a reference cohort  hdf5, and asymetry
"""

# Import library
import os
import numpy as np
from meld_classifier.meld_cohort import MeldCohort, MeldSubject
from meld_classifier.data_preprocessing import Preprocess, Feature
from meld_classifier.paths import BASE_PATH
import time

print(time.asctime(time.localtime(time.time())))
start_time = time.clock()


# Initialisation
# site from cohort
sites = [
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
# new site to process
new_sites = "H27"

features = {
        ".on_lh.thickness.mgh": 10,
        '.on_lh.w-g.pct.mgh' : 10,
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

### SMOOTH DATA ###
#-----------------------------------------------------------------------------------------------
print('PROCESS 1 : SMOOTHING')
#create cohort to smooth
c_raw = MeldCohort(hdf5_file_root='{site_code}_{group}_featurematrix.hdf5', dataset=None)
#create object smoothing
smoothing = Preprocess(c_raw,
                       site_codes=new_sites,
                       write_hdf5_file_root='{site_code}_{group}_featurematrix_smoothed_NewSite.hdf5',
                       data_dir=BASE_PATH)
#call function to smooth features
for feature in np.sort(list(set(features))):
    print(feature)
    smoothing.smooth_data(feature, features[feature])

#Transfer lesions in the new hdf5
smoothing.transfer_lesion()


### COMBAT DATA ###
#-----------------------------------------------------------------------------------------------
print('PROCESS 2 : COMBAT')

#create cohort to combat
c_smooth = MeldCohort(hdf5_file_root='{site_code}_{group}_featurematrix_smoothed_NewSite.hdf5', dataset=None)
#create cohort for reference
c_combat = MeldCohort(hdf5_file_root='{site_code}_{group}_featurematrix_combat_6.hdf5', dataset=None)
#create object smoothing
combat =Preprocess(c_smooth,
                    site_codes=new_sites,
                    write_hdf5_file_root='{site_code}_{group}_featurematrix_combat_NewSite.hdf5',
                    data_dir=BASE_PATH)
#call function to combat data
features_smooth = [feat.smooth_feat(feature, features[feature]) for feature in features]
for feature in features_smooth:
    print(feature)
    combat.combat_new_site(feature, new_site_code=new_sites,
                       ref_cohort = c_combat,
                       new_outliers_file = '/rds/project/kw350/rds-kw350-meld/meld_data/Data/list_outliers_qc_6.csv',
                       )

#Transfer lesions in the new hdf5
combat.transfer_lesion()


###  INTRA, INTER & ASYMETRY ###
#-----------------------------------------------------------------------------------------------
print('PROCESS 3 : INTRA, INTER & ASYMETRY')

#create cohort to normalise
c_combat = MeldCohort(hdf5_file_root='{site_code}_{group}_featurematrix_combat_NewSite.hdf5', dataset=None)
# create cohort of controls for inter normalisation
# provide a dataset to make sure only wanted subject are selected for normalisation, or it will take all existing sujects from cohort
c_controls = MeldCohort(hdf5_file_root='{site_code}_{group}_featurematrix_combat_6.hdf5', dataset='MELD_dataset_V6.csv')
# create object normalisation
norm = Preprocess(c_combat,
                    site_codes=new_sites,
                    write_hdf5_file_root='{site_code}_{group}_featurematrix_combat_NewSite.hdf5',
                    data_dir=BASE_PATH)
# call functions to normalise data
features_combat = [feat.combat_feat(feature) for feature in features_smooth]
for feature in features_combat:
    print(feature)
    norm.intra_inter_subject(feature, cohort_for_norm=c_controls)
    norm.asymmetry_subject(feature, cohort_for_norm=c_controls)

# #transfer lesion if not already done during combat
# combat.transfer_lesion()


# ### CREATE BOUNDARIES ###
# #-----------------------------------------------------------------------------------------------
print("PROCESS 4 : BOUNDARIES")

# create cohort to combat
c = MeldCohort(hdf5_file_root="{site_code}_{group}_featurematrix_combat_NewSite.hdf5", dataset=None)
# create object combat
boundary = Preprocess(
    c,
    site_codes=new_sites,
    write_hdf5_file_root="{site_code}_{group}_featurematrix_combat_NewSite.hdf5",
    data_dir=BASE_PATH,
)

# Create boundaries
boundary.make_boundary_zones()

end_time = time.clock()
print("execution time = " + str(end_time - start_time) + " seconds")
print(time.asctime(time.localtime(time.time())))
