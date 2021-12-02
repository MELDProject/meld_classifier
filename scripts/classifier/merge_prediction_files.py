import h5py
import pandas as pd
import argparse
import os
from meld_classifier.paths import EXPERIMENT_PATH

# merge_prediction_files
def merge_prediction_files(experiment_folder, experiment_name, fold="all", n_splits=100):
    experiment_path = os.path.join(EXPERIMENT_PATH, experiment_folder, f"fold_{fold}")

    # predictions hdf5
    fname = os.path.join(experiment_path, "results", "predictions_" + experiment_name + "_{}.hdf5")
    outfile = os.path.join(experiment_path, "results", f"predictions_{experiment_name}.hdf5")
    if not os.path.isfile(outfile):
        mode = "a"
    else:
        mode = "r+"

    print("merging hdf5")
    with h5py.File(outfile, mode=mode) as f:
        for split in range(0, n_splits):
            try:
                with h5py.File(fname.format(split), mode="r") as f_split:
                    for subj in f_split.keys():
                        print(subj)
                        f_split.copy(subj, f)
            except OSError as e:
                print(f"error in split {split}")
                pass

    # test results csv
    print("merging csv")
    test_results = []
    csv_fname = os.path.join(experiment_path, "results", "test_results_{}.csv")
    csv_outfile = os.path.join(experiment_path, "results", f"test_results.csv")
    for split in range(0, n_splits):
        try:
            df = pd.read_csv(csv_fname.format(split), index_col=False)
            test_results.append(df)
        except FileNotFoundError as f:
            print(f"error in split {split}")
            pass
    df = pd.concat(test_results, axis="index")
    df.to_csv(csv_outfile, index=False)


if __name__ == "__main__":
    # Set up experiment
    parser = argparse.ArgumentParser(description="Merges per-split hdf5 and csv results files")
    parser.add_argument(
        "--n-splits",
        type=int,
        default=20,
        help="number of splits that were used",
    )
    parser.add_argument(
        "--experiment-folder",
        help="Experiments folder",
    )
    parser.add_argument(
        "--experiment-name",
        help="subfolder to use, typically the ensemble model",
        default="ensemble_0",
    )
    parser.add_argument("--fold", default="all", help="fold number to use (by default all)")
    args = parser.parse_args()

    merge_prediction_files(
        experiment_folder=args.experiment_folder,
        experiment_name=args.experiment_name,
        fold=args.fold,
        n_splits=args.n_splits,
    )
