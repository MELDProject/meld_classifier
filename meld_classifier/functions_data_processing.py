import os
import numpy as np
import numpy.linalg as la
import pandas as pd
import nibabel as nb
import scipy.stats as sp
import h5py
import pickle
import meld_classifier.paths as paths
import meld_classifier.hdf5_io as hio
import meld_classifier.meld_io as io
import meld_classifier.mesh_tools as mesh_tools
from meld_classifier.neuroCombat_meld import neuroCombat
from meld_classifier.define_features import Feature


def normalise(data):
    if len(data.shape) == 1:
        data[:, np.newaxis]
    mean_intra = np.mean(data, axis=0)
    std_intra = np.std(data, axis=0)
    intra_norm = (data - mean_intra) / std_intra
    return intra_norm


def compute_asym(intra_norm):
    intra_lh = intra_norm[: int(len(intra_norm) / 2)]
    intra_rh = intra_norm[int(len(intra_norm) / 2) :]
    lh_asym = intra_lh - intra_rh
    rh_asym = intra_rh - intra_lh
    asym = np.hstack([lh_asym, rh_asym])
    return asym


def get_combat_values(parameter_name, feature_name, file_path):
    """Outputs the values and site list of a particular parameter from a feature that has been combat normalised"""
    with open(file_path, "rb") as file:
        f = pickle.load(file)
        combat_dir = f[feature_name]
        parameter = combat_dir[parameter_name]
        f.close()
    return parameter


def to_categorical(y, nb_classes=None):
    if not nb_classes:
        nb_classes = np.max(y) + 1
    Y = np.zeros((len(y), nb_classes))
    for i in range(len(y)):
        Y[i, y[i]] = 1.0
    return Y


def save_combat_parameters(
    parameter,
    feature_name,
    parameter_name,
    dtype="float32",
    filename="Combat_parameters.hdf5",
    base_path=paths.BASE_PATH,
):
    if os.path.isfile(os.path.join(base_path, "combat_data", filename)):
        mode = "r+"
    else:
        mode = "a"
    f = h5py.File(os.path.join(base_path, "combat_data", filename), mode)
    group = f.require_group(feature_name)
    dset = group.require_dataset(
        parameter_name, shape=np.shape(parameter), dtype=dtype, compression="gzip", compression_opts=9
    )
    dset[:] = parameter
    f.close()
    return


### MAIN FUNCTIONS ###
def smooth_data(subject, features, hdf5_filename, smooth_hdf5_filename, base_path=paths.BASE_PATH):
    """loads in features for subject and smooths"""
    # Load cortex label
    cortex = np.sort(nb.freesurfer.read_label(os.path.join(base_path, "fsaverage_sym/label/lh.cortex.label")))

    for f, feature in enumerate(features):
        feat = Feature(feature, features[feature])  # feature obct
        print(feat.raw)
        print("loading" + feat.raw)

        try:
            vals_lh = hio.get_feature_values(
                subject, hemi="lh", feature=feat.raw, hdf5_file_root=hdf5_filename, base_path=base_path
            )
            vals_rh = hio.get_feature_values(
                subject, hemi="rh", feature=feat.raw, hdf5_file_root=hdf5_filename, base_path=base_path
            )
            print(vals_lh)
            # Smooth raw features.
            if feat.smoother != None:
                print("smoothing...")
                smoothed_vals_lh = mesh_tools.smoothing_fs(vals_lh, feat.smoother)
                smoothed_vals_rh = mesh_tools.smoothing_fs(vals_rh, feat.smoother)
                raw_vals = np.hstack([smoothed_vals_lh[cortex], smoothed_vals_rh[cortex]])
            else:
                raw_vals = np.hstack([vals_lh[cortex], vals_rh[cortex]])

            raw_vals = np.array(raw_vals)

            if feature == ".on_lh.sulc.mgh":
                mean_raw_vals = np.mean(raw_vals)
                print(mean_raw_vals)
                if mean_raw_vals < 0.2:
                    pass
                elif mean_raw_vals > 0.2:
                    raw_vals = raw_vals / 10
            else:
                pass

            print("saving subjects' smoothed data for the following feature:" + feat.raw)
            hio.save_subject(
                subject, feat.smooth, cortex, raw_vals, hdf5_file_root=smooth_hdf5_filename, base_path=base_path
            )

        except KeyError as e:
            print("unable to load feature")


def combat_data(
    subject, features, smooth_hdf5_filename, combat_hdf5_filename, filename_combat_param, base_path=paths.BASE_PATH
):
    """combat normalise data and save out combat parameters"""
    # initialise file
    combat_param_file = os.path.join(base_path, "combat_data", filename_combat_param)

    # Load cortex label
    cortex = np.sort(nb.freesurfer.read_label(os.path.join(base_path, "fsaverage_sym/label/lh.cortex.label")))

    for f, feature in enumerate(features):
        feat = Feature(feature, features[feature])
        try:
            vals_lh = hio.get_feature_values(
                subject, hemi="lh", feature=feat.smooth, base_path=base_path, hdf5_file_root=smooth_hdf5_filename
            )
            vals_rh = hio.get_feature_values(
                subject, hemi="rh", feature=feat.smooth, base_path=base_path, hdf5_file_root=smooth_hdf5_filename
            )
            p_data = np.hstack([vals_lh[cortex], vals_rh[cortex]])
            isfeat = True
        except KeyError as e:
            print("unable to load feature {}".format(feature))
            isfeat = False

        if isfeat == True:

            print("Loading Combat parameters")
            with open(combat_param_file, "rb") as file:
                d = pickle.load(file)
                d_feat = d[feat.smooth]
                gamma_star = d_feat["gamma"][:]
                delta_star = d_feat["delta"][:]
                s_mean_saved = d_feat["s_mean"][:]
                v_pool_saved = d_feat["v_pool"][:]
                site_scanner_code_list = d_feat["site_scanner_codes_sorted"][:]
                info_dict = d_feat["info_dict"]
                design = d_feat["design"]
                file.close()

            print("find batch indx for subject")
            # site code for test_subjects
            site_code = io.get_sitecode(subject)
            scanner_code = io.get_scanner(subject)
            site_scanner_code = site_code + "_" + scanner_code
            batch_indx = np.where(np.sort(site_scanner_code_list) == site_scanner_code)[0][0]

            print("find site index for subject")
            site_scanner_code_list = site_scanner_code_list.astype("str")
            p_site_scanner_index = np.where(site_scanner_code_list == site_scanner_code)[0][0]

            print("standardise test patient data")
            # s_mean & v_pool have the same vlaues for all participants
            s_p_data = (p_data.T - s_mean_saved[:, p_site_scanner_index]) / np.sqrt(v_pool_saved.flatten())

            print("adjust new patient data by combat parameters")
            j = batch_indx  # batch index of patient to have combat
            dsq = np.sqrt(delta_star[j, :])
            denom = dsq
            batch_info = info_dict["batch_info"]
            n_batch = info_dict["n_batch"]
            batch_design = design[:, :n_batch]
            numer = np.array(s_p_data - np.dot(batch_design[batch_info[batch_indx], :], gamma_star))
            numer = numer[0, :]

            bayesdata = numer / denom

            vpsq = np.sqrt(v_pool_saved).reshape((len(v_pool_saved), 1))

            bayesdata = bayesdata.T * vpsq.ravel()
            bayesdata = bayesdata.flatten() + s_mean_saved[:, p_site_scanner_index]

            print("saving subjects' combat normalised data for the following feature:" + feat.raw)
            hio.save_subject(
                subject, feat.combat, cortex, bayesdata, hdf5_file_root=combat_hdf5_filename, base_path=base_path
            )


def normalise_data(
    subject, features, listids_c, combat_c_hdf5_filename, combat_hdf5_filename, base_path=paths.BASE_PATH
):
    """carry out intrasubject, and interhemispheric (asym) and intersubject normalisations"""
    # Load cortex label
    cortex = np.sort(nb.freesurfer.read_label(os.path.join(base_path, "fsaverage_sym/label/lh.cortex.label")))

    ## Intra-subject normalise & Calculate asymmetries between hemispheres
    for f, feature in enumerate(features):
        feat = Feature(feature, features[feature])
        # Load combat normalised features from hdf5 for test participant
        try:
            vals_lh = hio.get_feature_values(
                subject, hemi="lh", feature=feat.combat, hdf5_file_root=combat_hdf5_filename, base_path=base_path
            )
            vals_rh = hio.get_feature_values(
                subject, hemi="rh", feature=feat.combat, hdf5_file_root=combat_hdf5_filename, base_path=base_path
            )
            p_data = np.hstack([vals_lh[cortex], vals_rh[cortex]])
            # Intra subject normalise test participant
            intra_norm_p = normalise(p_data)
            # Calculate asymmetry of test participant
            asym_p = compute_asym(intra_norm_p)
            ## Load combat data of controls and intra subject normalise, calculate asymmetry
            raw_vals = []
            raw_vals_asym = []
            print("Calculate asymmetry between hemispheres")
            for k, control in enumerate(listids_c):
                try:
                    vals_lh = hio.get_feature_values(
                        control,
                        hemi="lh",
                        feature=feat.combat,
                        hdf5_file_root=combat_c_hdf5_filename,
                        base_path=base_path,
                    )
                    vals_rh = hio.get_feature_values(
                        control,
                        hemi="rh",
                        feature=feat.combat,
                        hdf5_file_root=combat_c_hdf5_filename,
                        base_path=base_path,
                    )
                    vals_both = np.hstack([vals_lh[cortex], vals_rh[cortex]])
                    # Intrasubject normalise
                    intra_norm = normalise(vals_both)
                    raw_vals.append(intra_norm)
                    # Calculate asymmetry
                    asym = compute_asym(intra_norm)
                    raw_vals_asym.append(asym)
                except KeyError as e:
                    pass
            raw_vals = np.array(raw_vals)
            raw_vals_asym = np.array(raw_vals_asym)
            print(feat.raw)
            print("Normalise participant by controls")
            mean_c = np.mean(raw_vals, axis=0)
            std_c = np.std(raw_vals, axis=0)
            norm_combat = (intra_norm_p - mean_c) / std_c

            mean_c = np.mean(raw_vals_asym, axis=0)
            std_c = np.std(raw_vals_asym, axis=0)
            norm_combat_asym = (asym_p - mean_c) / std_c
            print(
                "saving participant "
                + subject
                + " intra-internormalisation and asymmetry combat data for the following feature:"
                + feat.raw
            )
            hio.save_subject(
                subject,
                ".inter_z.intra_z" + feat.combat,
                cortex,
                norm_combat,
                hdf5_file_root=combat_hdf5_filename,
                base_path=base_path,
            )
            hio.save_subject(
                subject,
                ".inter_z.asym.intra_z" + feat.combat,
                cortex,
                norm_combat_asym,
                hdf5_file_root=combat_hdf5_filename,
                base_path=base_path,
            )
        except KeyError as e:
            print("unable to load feature")
            print(feat.raw)
