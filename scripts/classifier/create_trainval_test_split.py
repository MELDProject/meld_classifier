from meld_classifier.paths import BASE_PATH
import os
from meld_classifier.meld_cohort import MeldCohort
import argparse
import numpy as np
import pandas as pd
import sys


def get_subject_counts_per_site(listids):
    sites = []
    scanners = []
    subject_types = []

    for subj_id in listids:
        res = subj_id.split("_")
        sites.append("_".join(res[:2]))
        scanners.append(res[2])
        subject_types.append(res[3])
    df = pd.DataFrame({"site": sites, "scanner": scanners, "subject_type": subject_types})
    return df


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Create trainval and test splits. Should only be executed once per major dataset version. Resulting file is saved in BASE_PATH. Is run on a hardcoded list of sites and hdf5_file_root. If you would like to change this, edit this script."
    )
    parser.add_argument("--outname", default="MELD_dataset_V6.csv")
    parser.add_argument("--test_frac", type=float, default=0.5)
    parser.add_argument("--outliers", default=False)
    parser.add_argument("-f", "--force", default=False, action="store_true")
    args = parser.parse_args()

    # set subjects to consider for dataset
    data_parameters = {
        "site_codes": [
            "H1",
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
        ],
        "group": "both",
        "hdf5_file_root": "{}_{}_featurematrix_combat_6.hdf5",
    }
    # check if outname already exists
    if os.path.exists(os.path.join(BASE_PATH, args.outname)):
        if args.force:
            print("WARNING: overwriting existing output file {}".format(args.outname))
        else:
            print("output file {} exists. Use --force to overwrite. Exiting.".format(args.outname))
            sys.exit(0)
    print("test fraction is {}".format(args.test_frac))

    c = MeldCohort(hdf5_file_root=data_parameters["hdf5_file_root"])
    # load subject ids
    listids = c.get_subject_ids(site_codes=data_parameters["site_codes"], group=data_parameters["group"])
    np.random.seed(42)  # seed the dataset split generation
    np.random.shuffle(listids)

    # split the data
    num_test = int(len(listids) * args.test_frac)
    testids = listids[:num_test]
    trainvalids = listids[num_test:]

    # remove eventual outliers
    if args.outliers != False:
        # get outliers
        csv = pd.read_csv(os.path.join(BASE_PATH, args.outliers), header=0, encoding="unicode_escape")
        list_outliers = csv["ID"].values
        # remove outliers from different list
        listids = np.setdiff1d(listids, list_outliers)
        testids = np.setdiff1d(testids, list_outliers)
        trainvalids = np.setdiff1d(trainvalids, list_outliers)

    # print results of split
    df_all = get_subject_counts_per_site(listids)
    df_test = get_subject_counts_per_site(testids)
    for group, values in df_all.groupby(["site", "scanner", "subject_type"]):
        num_all = len(values)
        try:
            num_test = df_test.groupby(["site", "scanner", "subject_type"]).get_group(group)["site"].count()
        except KeyError:
            num_test = 0

        print("{}: {} of {} in test ({}%)".format(group, num_test, num_all, num_test / num_all * 100))

    # save test / trainval ids
    # create matrix with test / trainval ids
    subject_ids = pd.DataFrame(
        {"subject_id": listids, "split": ["test" if subj_id in testids else "trainval" for subj_id in listids]}
    )
    subject_ids.to_csv(os.path.join(BASE_PATH, args.outname))
