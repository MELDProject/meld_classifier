## make a plot of a single patient, lesional hemisphere
import os
import json
from meld_classifier.network_tools import build_model
from meld_classifier.experiment import get_subject_ids
from meld_classifier.meld_plotting import trim,rotate90
import meld_classifier.hdf5_io as io
import meld_classifier.paths as paths
import numpy as np
import seaborn as sns
import json
import pandas as pd
from meld_classifier import hdf5_io as hio
import matplotlib.pyplot as plt
import meld_classifier.data_prep as dp
import nibabel as nb
import matplotlib_surface_plotting as msp
from PIL import Image
from PIL import ImageFont
from PIL import ImageDraw
import subprocess
import argparse


# either 3x3 or 2x2 (FLAIR)
def plot_single_subject(data_to_plots, lesion, feature_names=None, out_filename="tmp.png"):
    """create a grid of flatmap plots"""
    # load in meshes
    flat = nb.load(os.path.join(paths.BASE_PATH, "fsaverage_sym", "surf", "lh.full.patch.flat.gii"))

    vertices, faces = flat.darrays[0].data, flat.darrays[1].data
    cortex = np.sort(
        nb.freesurfer.io.read_label(os.path.join(paths.BASE_PATH, "fsaverage_sym", "label", "lh.cortex.label"))
    )
    cortex_bin = np.zeros(len(vertices)).astype(bool)
    cortex_bin[cortex] = 1
    # round up to get the square grid size
    gridsize = np.ceil(np.sqrt(len(data_to_plots))).astype(int)
    ims = np.zeros((gridsize, gridsize), dtype=object)
    random = np.random.choice(100000)
    for k, data_to_plot in enumerate(data_to_plots):
        msp.plot_surf(
            vertices,
            faces,
            data_to_plot,
            flat_map=True,
            base_size=10,
            mask=~cortex_bin,
            pvals=np.ones_like(cortex_bin),
            parcel=lesion,
            vmin=np.percentile(data_to_plot[cortex_bin], 1),
            vmax=np.percentile(data_to_plot[cortex_bin], 99),
            cmap="viridis",
            colorbar=False,
            filename=out_filename,
        )
        plt.close()
        #subprocess.call(f"convert {out_filename} -trim ./tmp{random}1.png", shell=True)
        #subprocess.call(f"convert ./tmp{random}1.png -rotate 90 {out_filename}", shell=True)
        #os.remove(f"./tmp{random}1.png")
        im = Image.open(out_filename)
        im = trim(im)
        im = rotate90(im)
        im = im.convert("RGBA")
        fnt = ImageFont.truetype("Pillow/Tests/fonts/FreeSansBold.ttf", 25)
        f_name = ""
        if feature_names is not None:
            if k == 0:
                f_name = feature_names[k]
                base = np.array(im.convert("RGBA"))
            else:
                f_name = feature_names[k][35:-9]
        draw = ImageDraw.Draw(im)
        draw.text((100, 0), f_name, (255, 0, 0), font=fnt)
        arr_im = np.array(im.convert("RGBA"))
        s0 = np.min([base.shape[0], arr_im.shape[0]])
        s1 = np.min([base.shape[1], arr_im.shape[1]])
        base[:s0, :s1, :3] = arr_im[:s0, :s1, :3]

        # make transparent white
        # cropped[cropped[:,:,3]==0]=255
        base = base[:, :, :3]
        ims[k // gridsize, k % gridsize] = base.copy()

    rows = np.zeros(1 + k // gridsize, dtype=object)
    for j in np.arange(1 + k // gridsize):
        try:
            rows[j] = np.hstack(ims[j])
        except ValueError:
            ims[j, k % gridsize + 1] = np.ones_like(base) * 255
            ims[j, k % gridsize + 2] = np.ones_like(base) * 255
            rows[j] = np.hstack(ims[j])
    grid_ims = np.vstack(rows)
    im = Image.fromarray(grid_ims)
    im.save(out_filename)


def get_n_features(data_parameters, features):
    """considers universal_features, self.features, and neighbours"""
    # features
    n_features = len(features)
    # universal features
    preloaded_features = io.load_universal_features(data_parameters["universal_features"])
    if preloaded_features is not None:
        n_features += preloaded_features.shape[-1]
    # neighbours
    n_features = n_features * (data_parameters["num_neighbours"] + 1)
    return n_features


def plot_reports_for_experiment(experiments_folder, experiment, date, fold, param, subset="val_ids"):
    """plot reports for all subjects in experiment fold for given parameter"""
    features_to_plot = [
        ".inter_z.asym.intra_z.combat.on_lh.gm_FLAIR_0.25.sm10.mgh",
        ".inter_z.asym.intra_z.combat.on_lh.gm_FLAIR_0.5.sm10.mgh",
        ".inter_z.asym.intra_z.combat.on_lh.gm_FLAIR_0.75.sm10.mgh",
        ".inter_z.asym.intra_z.combat.on_lh.gm_FLAIR_0.sm10.mgh",
        ".inter_z.asym.intra_z.combat.on_lh.pial.K_filtered.sm20.mgh",
        ".inter_z.asym.intra_z.combat.on_lh.thickness.sm10.mgh",
        ".inter_z.asym.intra_z.combat.on_lh.w-g.pct.sm10.mgh",
        ".inter_z.asym.intra_z.combat.on_lh.wm_FLAIR_0.5.sm10.mgh",
        ".inter_z.asym.intra_z.combat.on_lh.wm_FLAIR_1.sm10.mgh",
    ]
    experiment_path = os.path.join(experiments_folder, f"{experiment}_{date}", f"fold_{fold}")
    experiment_name = experiment + "_" + param
    data_parameters = json.load(open(os.path.join(experiment_path, "data_parameters_{}.json".format(experiment_name))))
    listids, features = get_subject_ids(data_parameters, verbose=False)
    cortex = np.sort(
        nb.freesurfer.io.read_label(os.path.join(paths.BASE_PATH, "fsaverage_sym", "label", "lh.cortex.label"))
    )
    _, _, features_to_ignore = dp.filter_features(
        data_parameters.get("features_to_replace_with_0", []),
        return_excluded=True,
        combat="combat" in data_parameters["hdf5_file_root"],
    )
    subject_ids = data_parameters[subset]

    # setup network
    if "test" not in subset:
        network_parameters = json.load(
            open(os.path.join(experiment_path, "network_parameters_{}.json".format(experiment_name)))
        )
        n_features = get_n_features(data_parameters, features)
        checkpoint_path = os.path.join(experiment_path, "models", experiment_name)
        model = build_model(
            n_features=n_features // (data_parameters["num_neighbours"] + 1),
            n_neighbours=data_parameters["num_neighbours"],
            **network_parameters,
        )
        model.load_weights(checkpoint_path)
    else:
        subject_ids = np.array_split(subject_ids, 10)[fold]

    for subject in subject_ids:

        if "FCD" in subject:
            print(subject)
            sub_features, labels = io.load_subject_combined_hemisphere_data(
                subject,
                features,
                cortex=cortex,
                normalise=False,
                hdf5_file_root=data_parameters["hdf5_file_root"],
                neighbours=None,
                preloaded_features=None,
                demographic_features=None,
                features_to_ignore=data_parameters.get("features_to_replace_with_0", []),
            )

            les_hemi = ["lh", "rh"].index(io.get_les_hemi(subject))
            lesion = dp.split_hemispheres(labels)[["left", "right"][les_hemi]]
            boundary_label = io.load_boundary_zone(subject, cortex=cortex)
            boundary_hemi = dp.split_hemispheres(boundary_label)[["left", "right"][les_hemi]]
            lesion_combi = lesion + boundary_hemi
            lesional_hemi = dp.split_hemispheres(sub_features)[["left", "right"][les_hemi]]
            if "test" not in subset:
                prediction = model.predict(sub_features, batch_size=data_parameters["batch_size"]).ravel()
                pred_hemi = dp.split_hemispheres(prediction)[["left", "right"][les_hemi]]
                data_to_plots = [pred_hemi > network_parameters["optimal_threshold"]]
            else:
                data_to_plots = [lesion_combi]

            # split to just plot lesional hemisphere

            feature_list = io.get_feature_list(subject, hdf5_file_root=data_parameters["hdf5_file_root"])

            overlapping_features = np.intersect1d(feature_list, features_to_plot)
            for f in overlapping_features:
                data_to_plots.append(lesional_hemi[:, features.index(f)])
            feature_names = ["prediction"] + list(overlapping_features)
            plot_single_subject(
                data_to_plots,
                lesion=lesion_combi,
                feature_names=feature_names,
                out_filename=os.path.join(paths.EXPERIMENT_PATH, "qc_ims", "{}.jpeg".format(subject)),
            )

    return


def get_n_features(data_parameters, features):
    """considers universal_features, self.features, and neighbours"""
    # features
    n_features = len(features)
    # universal features
    preloaded_features = io.load_universal_features(data_parameters["universal_features"])
    if preloaded_features is not None:
        n_features += preloaded_features.shape[-1]
    # neighbours
    n_features = n_features * (data_parameters["num_neighbours"] + 1)
    return n_features


if __name__ == "__main__":
    # Set up experiment
    parser = argparse.ArgumentParser(description="Plot flat maps on subjects")
    parser.add_argument("--fold", type=int, default=0)
    parser.add_argument("--subset", type=str, default="val_ids")
    args = parser.parse_args()
    experiments_folder = "/home/kw350/rds/rds-kw350-meld/experiments/kw350/"
    experiment = "iteration"
    date = "21-02-12"
    param = "0"
    fold = args.fold
    subset = args.subset
    # Set up model
    experiment_path = os.path.join(experiments_folder, f"{experiment}_{date}", f"fold_{fold}")
    experiment_name = experiment + "_" + param
    data_parameters = json.load(open(os.path.join(experiment_path, "data_parameters_{}.json".format(experiment_name))))
    plot_reports_for_experiment(experiments_folder, experiment, date, fold, param, subset=subset)
