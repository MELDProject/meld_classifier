import numpy as np
import h5py
import pandas as pd
import meld_classifier.matplotlib_surface_plotting as msp
from meld_classifier.meld_cohort import MeldCohort
import os
import meld_classifier.paths as paths
import nibabel as nb
import argparse
import matplotlib.pyplot as plt


def load_prediction(subject, hdf5):
    results = {}
    with h5py.File(hdf5, "r") as f:
        for hemi in ["lh", "rh"]:
            results[hemi] = f[subject][hemi]["prediction"][:]
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plot all subjects' predictions on inflated brain surface.")
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
    experiment_folder = args.experiment_folder
    experiment_name = args.experiment_name
    fold = args.fold

    hemis = ["lh", "rh"]
    c = MeldCohort()
    vertices = c.surf_partial["coords"]
    faces = c.surf_partial["faces"]

    experiment_path = os.path.join(paths.EXPERIMENT_PATH, experiment_folder, f"fold_{fold}")
    result_file = os.path.join(experiment_path, "results", "test_results.csv")
    prediction_file = os.path.join(experiment_path, "results", f"predictions_{experiment_name}.hdf5")

    if os.path.isfile(result_file):
        df = pd.read_csv(result_file, index_col=False)
        subjids = np.array(df["ID"])
    else:
        subjids = []

    if os.path.isfile(prediction_file):
        for subject in subjids:
            print(subject)
            predictions = load_prediction(subject, prediction_file)
            for hemi in hemis:
                prediction_h = predictions[hemi]
                overlay = np.zeros_like(c.cortex_mask, dtype=int)
                overlay[c.cortex_mask] = prediction_h
                msp.plot_surf(
                    vertices,
                    faces,
                    overlay,
                    rotate=[90, 270],
                    filename=os.path.join(
                        experiment_path, "results", "images", "prediction_ims_{}_{}.png".format(subject, hemi)
                    ),
                    vmin=0,
                    vmax=1,
                )
            plt.close("all")
