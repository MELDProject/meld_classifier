"""
This scripts is used to retrieved the demographics information from patients and controls
and correct information from site specific mistakes

output : homogeneous demographic information csv file
"""

# import necessary packages
import meld_classifier.paths as paths
import pandas as pd

pd.options.mode.chained_assignment = None  # default='warn'
import numpy as np
import os
from meld_classifier.meld_cohort import MeldCohort, MeldSubject
import logging

logging.basicConfig(level=logging.DEBUG)


def histology_per_subject(histo_id):
    """return histological classification for subjects, in order of input subjects"""
    try:
        histo_id = histo_id[0]
    except IndexError:
        # catch ones with a wrong 000
        spl = ids.split("_")
        spl[4] = "0" + spl[4]
        n_ids = "_".join(spl)
        histo_id = get_demographic_feature(n_ids, "Histo")
    if "mild cortical dysplasia" in histo_id:
        histo = "NaN"
    elif "3" in histo_id or "III" in histo_id:
        histo = "FCD_3"
    elif "2A" in histo_id or "IIA" in histo_id:
        histo = "FCD_2A"
    elif "2B" in histo_id or "IIB" in histo_id:
        histo = "FCD_2B"
    elif "1" in histo_id or "I" in histo_id:
        histo = "FCD_1"
    else:
        histo = "NaN"

    return histo


def tidy_features(feature_list):
    """function to tidy up values of feature"""
    filter_values = [555, "555", 666, "666", "NO", "No", "No f/u"]
    for value in filter_values:
        feature_list[feature_list == value] = float("NaN")
    # specific change for H27 because Engel outcome in roman letters
    roman2int = {"I": 1, "IA": 1, "IB": 1, "IC": 1, "ID": 1, "II": 2, "III": 3, "IIIA": 3, "IV": 4}
    for value in roman2int:
        feature_list[feature_list == value] = float(roman2int[value])

    return np.array(feature_list, dtype=np.float)


# 1) Prepare data
# -------------------------------------
# hospital sites to include

# sites = ['H28']
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
    "H27",
    "H28",
]

c = MeldCohort(hdf5_file_root="{site_code}_{group}_featurematrix.hdf5", dataset=None)


listids = c.get_subject_ids(site_codes=sites, lesional_only=False)

## patients confirmed as to be excluded by collaborating sites
subjects_excluded = list(pd.read_csv(os.path.join(paths.BASE_PATH, 'list_subjects_excluded_from_sites.csv'), header=0)["ID"])
for subject in subjects_excluded: 
    try:
        listids.remove(subject)
        print(subject)
    except:
        pass
data_path = paths.BASE_PATH

# 2) Retrieve data
# -------------------------------------
# load in features. Engel outcome will be replaced by seizure freedom further down
features = [
    "Age of onset",
    "Duration",
    "Age at preoperative",
    "Sex",
    "Ever reported MRI negative",
    "Engel Outcome",
    "Surgery",
    "f/u",
    "urfer",
]
data = []
for ids in listids:
    # create dictionaries of the features, then make into pandas dataframe
    # this retrieves the demographic features
    subj = MeldSubject(ids, cohort=c)
    features_dict = dict(
        zip(
            features,
            subj.get_demographic_features(
                features, csv_file=f"MELD_{subj.site_code}/MELD_{subj.site_code}_participants.csv"
            ),
        )
    )
    features_dict["ID"] = subj.subject_id
    features_dict["group"] = subj.group
    features_dict["Lesion area"], features_dict["Hemisphere"], features_dict["Lobe"] = subj.get_lesion_area()
    features_dict["lesion"] = subj.has_lesion()
    features_dict["Site"] = subj.site_code
    features_dict["Scanner"] = subj.scanner
    features_dict["FLAIR"] = subj.has_flair
    features_dict["Histology"] = histology_per_subject(
        subj.get_demographic_features(
            ["Histo"], csv_file=f"MELD_{subj.site_code}/MELD_{subj.site_code}_participants.csv"
        )
    )
    data.append(features_dict)


df = pd.DataFrame(data)
df = df.rename(columns={"urfer": "FreeSurfer"})

column_order = [
    "ID",
    "Site",
    "group",
    "Age of onset",
    "Duration",
    "Age at preoperative",
    "Sex",
    "Ever reported MRI negative",
    "Engel Outcome",
    "Histology",
    "Surgery",
    "f/u",
    "FreeSurfer",
    "Scanner",
    "lesion",
    "Hemisphere",
    "Lobe",
    "FLAIR",
]
df = df[column_order]

# 3) Tidy data
# -------------------------------------
# This tidies the features - e.g. replacing NO (not operated) with NaN
features_to_tidy = [
    "Age of onset",
    "Duration",
    "Age at preoperative",
    "Sex",
    "Ever reported MRI negative",
    "Engel Outcome",
    "Surgery",
    "f/u",
]

for feature in features_to_tidy:
    print(feature)
    try:
        df[feature] = tidy_features(np.array(df[feature]))
    except ValueError:
        #  print(np.array(df[feature]))
        df[feature] = tidy_features(np.array(df[feature]).astype(float))
df = df.replace("NaN", np.nan, regex=True)
df[df["Sex"] > 1]["Sex"] = np.nan

# switch engel for seizure free
df["Seizure free"] = (df["Engel Outcome"] == 1).astype(int)
df["Seizure free"][pd.isna(df["Engel Outcome"])] = np.nan

# remove 2.0 from surgery (some sites misscoded no surgery as 2 in the .csv files)
df["Surgery"][df["Surgery"] == 2.0] = 0
df["Sex"][df["Sex"] == 2.0] = 0
df["Ever reported MRI negative"][df["Ever reported MRI negative"] == 2.0] = 0

# problem - age of onset > age at preoperative
problem_2 = df["Age at preoperative"] < df["Age of onset"]
# swap round data for two patients in H26, where entered age of onset and age at preop are in the wrong columns
swap_columns = ["MELD_H26_15T_FCD_0010", "MELD_H26_3T_FCD_0011"]
for ids in df[problem_2]["ID"]:
    if ids in swap_columns:
        df["Age of onset"][df["ID"] == ids] = 9.0
        df["Age at preoperative"][df["ID"] == ids] = 12.0
    else:
        df["Age of onset"][df["ID"] == ids] = np.nan
        df["Age at preoperative"][df["ID"] == ids] = np.nan

# replace recorded duration with difference between onset and age at scan
df["Duration"] = df["Age at preoperative"] - df["Age of onset"]


# 4) Save data
# -------------------------------------
df.to_csv(os.path.join(data_path, "demographics_qc_allgroups_TEST.csv"), index=False)
