import numpy as np
import meld_classifier.matplotlib_surface_plotting as msp
from meld_classifier.meld_cohort import MeldCohort
import os
import meld_classifier.paths as paths
import nibabel as nb
import argparse
import matplotlib.pyplot as plt

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plot all subjects' lesion masks")
    parser.add_argument("--split", default=0, help="index of split", type=int)
    args = parser.parse_args()
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
    c = MeldCohort()
    subjids = c.get_subject_ids(site_codes=site_codes)
    subjids_split = np.array_split(np.array(subjids), 4)

    vertices = c.surf_partial["coords"]
    faces = c.surf_partial["faces"]

    for subject in subjids_split[args.split]:
        if not os.path.isfile(os.path.join(paths.EXPERIMENT_PATH, "lesion_ims", "{}.png".format(subject))):
            try:
                print(subject)
                subj = MeldSubject(subject, cohort=c)
                hemisphere = subj.get_lesion_hemisphere()
                lesion = np.ceil(subj.load_feature_values(hemi=hemisphere, feature=".on_lh.lesion.mgh")).astype(int)
                msp.plot_surf(
                    vertices,
                    faces,
                    lesion,
                    rotate=[90, 270],
                    filename=os.path.join(paths.EXPERIMENT_PATH, "lesion_ims", "{}.png".format(subject)),
                    vmin=0,
                    vmax=1,
                )
                plt.close("all")
            except TypeError:
                pass
