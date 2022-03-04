from meld_classifier.paths import (
    DEMOGRAPHIC_FEATURES_FILE,
    CORTEX_LABEL_FILE,
    SURFACE_FILE,
    DEFAULT_HDF5_FILE_ROOT,
    BOUNDARY_ZONE_FILE,
    NVERT,
    BASE_PATH,
    DK_ATLAS_FILE,
    SMOOTH_CALIB_FILE,
    COMBAT_PARAMS_FILE,
    FINAL_SCALING_PARAMS, 
    NORM_CONTROLS_PARAMS_FILE
)
import pandas as pd
import numpy as np
import nibabel as nb
import os
import h5py
import glob
import logging
import random
import json
from itertools import chain
import potpourri3d as pp3d
import meld_classifier.mesh_tools as mt
from meld_classifier.meld_cohort import MeldCohort, MeldSubject
from neuroCombat import neuroCombat, neuroCombatFromTraining


class Preprocess:
    def __init__(self, cohort, site_codes=None, write_hdf5_file_root=None, data_dir=BASE_PATH):
        self.cohort = cohort
        self._covars = None
        self.write_hdf5_file_root = write_hdf5_file_root
        self.data_dir = data_dir
        self.site_codes = site_codes
        # filter subject ids based on site codes
        if self.site_codes is None:
            self.site_codes = self.cohort.get_sites()
        self.subject_ids = self.cohort.get_subject_ids(site_codes=self.site_codes, lesional_only=False)
        self.feat = Feature()
        # calibration_smoothing : curve to calibrate smoothing on surface mesh
        self._calibration_smoothing = None

    @property
    def covars(self):
        if self._covars is None:
            self._covars = self.load_covars()
        return self._covars

    def add_lesion(self, subj):
        hemi = subj.get_lesion_hemisphere()
        print(hemi)
        if hemi is not None:
            print("transfer lesion for {}".format(subj.subject_id))
            lesion = subj.load_feature_values(".on_lh.lesion.mgh", hemi)
            subj.write_feature_values(
                ".on_lh.lesion.mgh",
                lesion[self.cohort.cortex_mask],
                hemis=[hemi],
                hdf5_file_root=self.write_hdf5_file_root,
            )

    def transfer_lesion(self):
        new_cohort = MeldCohort(hdf5_file_root=self.write_hdf5_file_root)
        new_listids = new_cohort.get_subject_ids(lesional_only=False)
        for subject in self.subject_ids:
            if subject in new_listids:
                print("exist")
                subj = MeldSubject(subject, self.cohort)
                self.add_lesion(subj)

    def flatten(self, t):
        return [item for sublist in t for item in sublist]

    def make_boundary_zones(self, smoothing=0, boundary_feature_name=".on_lh.boundary_zone.mgh"):
        # preload geodesic distance solver
        solver = pp3d.MeshHeatMethodDistanceSolver(self.cohort.surf["coords"], self.cohort.surf["faces"])

        for ids in self.subject_ids:
            subj = MeldSubject(ids, cohort=self.cohort)
            hemi = subj.get_lesion_hemisphere()
            if hemi:
                if subj.has_lesion:
                    print(ids)
                    overlay = np.round(subj.load_feature_values(hemi=hemi, feature=".on_lh.lesion.mgh")[:])
                    # smooth a bit for registration, conservative masks etc.
                    if smoothing > 0:
                        overlay = np.ceil(mt.smoothing_fs(overlay, fwhm=smoothing)).astype(int)
                    non_lesion_and_neighbours = self.flatten(np.array(self.cohort.neighbours)[overlay == 0])
                    lesion_boundary_vertices = np.setdiff1d(non_lesion_and_neighbours, np.where(overlay == 0)[0])
                    boundary_distance = solver.compute_distance_multisource(lesion_boundary_vertices)
                    # include lesion
                    boundary_distance[overlay == 1] = 0
                    # mask medial wall
                    boundary_distance = boundary_distance[self.cohort.cortex_mask]
                    # write in hdf5
                    subj.write_feature_values(
                        boundary_feature_name, boundary_distance, hemis=[hemi], hdf5_file_root=self.write_hdf5_file_root
                    )
                else:
                    print("skipping ", ids)

    def load_covars(self, subject_ids=None):
        if subject_ids is None:
            subject_ids = self.subject_ids
        covars = pd.DataFrame()
        ages = []
        sex = []
        group = []
        sites_scanners = []
        for subject in subject_ids:
            subj = MeldSubject(subject, cohort=self.cohort)
            a, s = subj.get_demographic_features(["Age at preop", "Sex"])
            ages.append(a)
            sex.append(s)
            group.append(subj.is_patient)
            sites_scanners.append(subj.site_code + "_" + subj.scanner)

        covars["ages"] = ages
        covars["sex"] = sex
        covars["group"] = group
        covars["site_scanner"] = sites_scanners
        covars["ID"] = subject_ids

        #         #clean missing values in demographics
        covars["ages"] = covars.groupby("site_scanner").transform(lambda x: x.fillna(x.mean()))["ages"]
        covars["sex"] = covars.groupby("site_scanner").transform(lambda x: x.fillna(random.choice([0, 1])))["sex"]
        return covars

    def save_norm_combat_parameters(self, feature, estimates, hdf5_file):
        """Save estimates from combat and normalisation parameters on hdf5"""
        if not os.path.isfile(hdf5_file):
            hdf5_file_context = h5py.File(hdf5_file, "a")
        else:
            hdf5_file_context = h5py.File(hdf5_file, "r+")

        with hdf5_file_context as f:
            list_params = list(set(estimates))
            for parameter_name in list_params:
                parameter = estimates[parameter_name]
                parameter = np.array(parameter)
                dtype = parameter.dtype
                dtype = parameter.dtype

                group = f.require_group(feature)
                if dtype == "O":
                    dset = group.require_dataset(
                        parameter_name, shape=np.shape(parameter), dtype="S10", compression="gzip", compression_opts=9
                    )
                    dset.attrs["values"] = list(parameter)
                else:
                    dset = group.require_dataset(
                        parameter_name, shape=np.shape(parameter), dtype=dtype, compression="gzip", compression_opts=9
                    )
                    dset[:] = parameter

    def read_norm_combat_parameters(self, feature, hdf5_file):
        """reconstruct estimates dictionnary from the combat parameters hdf5 file"""
        hdf5_file_context = h5py.File(hdf5_file, "r+")
        estimates = {}
        with hdf5_file_context as f:
            feat_dir = f[feature]
            parameters = feat_dir.keys()
            for param in parameters:
                if feat_dir[param].dtype == "S10":
                    estimates[param] = feat_dir[param].attrs["values"].astype(np.str)
                else:
                    estimates[param] = feat_dir[param][:]
        return estimates
    
    def shrink_combat_estimates(self, estimates):
        """ shrink combat estimates to reduce size file"""
        #combined mod.mean with stand.mean
        stand_mean =  estimates['stand.mean'][:, 0] + estimates['mod.mean'].mean(axis=1)
        estimates['stand.mean'] = stand_mean
        #save the number of subjects to un-shrink later
        estimates['num_subjects']= np.array([estimates['mod.mean'].shape[1]])
        #remove mod.mean to reduce estimates size
        del estimates['mod.mean']
        return estimates

    def unshrink_combat_estimates(self, estimates):
        """ unshrink combat estimates to use as input in neuroCombatFromTraining"""
        num_subjects = estimates['num_subjects'][0]
        mod_mean = np.zeros((len(estimates['stand.mean']),num_subjects ))
        estimates['mod.mean'] = mod_mean
        estimates['stand.mean'] = np.tile(estimates['stand.mean'], (num_subjects,1)).T
        return estimates

    def combat_whole_cohort(self, feature_name, outliers_file=None, combat_params_file=None):
        """Harmonise data between site/scanner with age, sex and disease status as covariate
        using neuroComBat (Fortin et al., 2018, Neuroimage) and save in hdf5
        Args:
            feature_name (str): name of the feature, usually smoothed data.
            outliers_file (str): file name of the csv containing subject ID to exclude from harmonisation

        Returns:
            estimates : Combat parameters used for the harmonisation. Need to save for new patient harmonisation.
            info : dictionary of information from combat
        """
        # read morphological outliers from cohort.
        if outliers_file is not None:
            outliers = list(pd.read_csv(os.path.join(BASE_PATH, outliers_file), header=0)["ID"])
        else:
            outliers = []
        # load in features using cohort + subject class
        combat_subject_include = np.zeros(len(self.subject_ids), dtype=bool)
        precombat_features = []
        for k, subject in enumerate(self.subject_ids):
            subj = MeldSubject(subject, cohort=self.cohort)
            # exclude outliers and subject without feature
            if (subj.has_features(feature_name)) & (subject not in outliers):
                lh = subj.load_feature_values(feature_name, hemi="lh")[self.cohort.cortex_mask]
                rh = subj.load_feature_values(feature_name, hemi="rh")[self.cohort.cortex_mask]
                combined_hemis = np.hstack([lh, rh])
                precombat_features.append(combined_hemis)
                combat_subject_include[k] = True
            else:
                print("exclude")
                combat_subject_include[k] = False
        if precombat_features:
            precombat_features = np.array(precombat_features)
            # load in covariates - age, sex, group, site and scanner unless provided
            covars = self.covars[combat_subject_include].copy()
            # check for nan
            index_nan = pd.isnull(covars).any(1).to_numpy().nonzero()[0]
            if len(index_nan) != 0:
                print(
                    "There is missing information in the covariates for subjects {}. \
                Combat aborted".format(
                        np.array(covars["ID"])[index_nan]
                    )
                )
            else:
                # function to check for single subjects
                covars, precombat_features = self.remove_isolated_subs(covars, precombat_features)
                covars = covars.reset_index(drop=True)

                dict_combat = neuroCombat(
                    precombat_features.T,
                    covars,
                    batch_col="site_scanner",
                    categorical_cols=["sex", "group"],
                    continuous_cols="ages",
                )
                # save combat parameters
                if combat_params_file is not None:
                    shrink_estimates = self.shrink_combat_estimates(dict_combat["estimates"])
                    self.save_norm_combat_parameters(feature_name, shrink_estimates, combat_params_file)

                post_combat_feature_name = self.feat.combat_feat(feature_name)

                print("Combat finished \n Saving data")
                self.save_cohort_features(post_combat_feature_name, dict_combat["data"].T, np.array(covars["ID"]))
        else:
            print('no data to combat harmonised')
            pass
        
    def combat_new_site(
        self,
        feature_name,
        new_site_code,
        ref_cohort,
        new_outliers_file=None,
    ):
        """Harmonise new site data to post-combat whole cohort and save in
        new hdf5 file. New sites are run individually currently.
        assumes that the base cohort is the post-combat cohort
        Args:
            feature_name (str): name of the feature
            outliers_file : outliers file for the new cohort

        """
        # read morphological outliers from new cohort only
        if new_outliers_file is not None:
            outliers = list(pd.read_csv(os.path.join(BASE_PATH, new_outliers_file), header=0)["ID"])
        else:
            outliers = []

        # make empty for all subjects
        ref_subject_ids = ref_cohort.get_subject_ids(lesional_only=False)
        combined_ids = ref_subject_ids + self.subject_ids
        combat_subject_include = np.zeros(len(combined_ids), dtype=bool)
        new_site_codes = np.ones(len(combined_ids), dtype=int)
        new_site_codes[: len(ref_subject_ids)] = 0
        # load in both combat normalised and new cohort
        precombat_features = []
        cohorts = [ref_cohort, self.cohort]
        # need pre combat and post combat feature names, loading in post for the whole cohort,
        # pre for the new cohort.
        post_combat_feature_name = self.feat.combat_feat(feature_name)
        feature_names = [post_combat_feature_name, feature_name]
        for k, subject in enumerate(combined_ids):
            # get the reference index and cohort object for the site, 0 whole cohort, 1 new cohort
            site_code_index = new_site_codes[k]
            cohort = cohorts[site_code_index]
            subj = MeldSubject(subject, cohort=cohort)
            # exclude outliers and subject without feature
            if (subj.has_features(feature_names[site_code_index])) & (subject not in outliers):
                lh = subj.load_feature_values(feature_names[site_code_index], hemi="lh")[self.cohort.cortex_mask]
                rh = subj.load_feature_values(feature_names[site_code_index], hemi="rh")[self.cohort.cortex_mask]
                combined_hemis = np.hstack([lh, rh])
                precombat_features.append(combined_hemis)
                combat_subject_include[k] = True
            else:
                combat_subject_include[k] = False
        if precombat_features:
            precombat_features = np.array(precombat_features)
            # load in covariates - age, sex, group, site and scanner,
            # set site_scanner to 0 for existing cohort
            covars = pd.concat([self.load_covars(ref_subject_ids), self.covars])
            covars["site_scanner"][: len(ref_subject_ids)] = "H0"
            covars = covars[combat_subject_include].copy()

            # function to check for single subjects
            covars, precombat_features = self.remove_isolated_subs(covars, precombat_features)

            dict_combat = neuroCombat(
                precombat_features.T,
                covars,
                batch_col="site_scanner",
                categorical_cols=["sex", "group"],
                continuous_cols=["ages"],
                ref_batch="H0",
            )

            print("Combat finished \n Saving data")
            # only save out new subjects
            ids_to_save = np.array(covars[covars["site_scanner"] != "H0"]["ID"])
            self.save_cohort_features(
                post_combat_feature_name, dict_combat["data"].T[covars["site_scanner"] != "H0"], ids_to_save
            )
        else:
            print('No data to combat harmonised')
            pass

    def combat_new_subject(self, feature_name, combat_params_file):
        """Harmonise new subject data with Combat parameters from whole cohort
            and save in new hdf5 file
        Args:
            subjects (list of str): list of subjects ID to harmonise
            feature_name (str): name of the feature, usually smoothed data.
            combat_estimates (arrays): combat parameters used for the harmonisation
        """
        # load combat parameters        
        combat_estimates = self.read_norm_combat_parameters(feature_name, combat_params_file)
        combat_estimates = self.unshrink_combat_estimates(combat_estimates)
        precombat_features = []
        site_scanner = []
        for subject in self.subject_ids:
            subj = MeldSubject(subject, cohort=self.cohort)
            if subj.has_features(feature_name):
                lh = subj.load_feature_values(feature_name, hemi="lh")[self.cohort.cortex_mask]
                rh = subj.load_feature_values(feature_name, hemi="rh")[self.cohort.cortex_mask]
                combined_hemis = np.hstack([lh, rh])
                precombat_features.append(combined_hemis)
                site_scanner.append(subj.site_code + "_" + subj.scanner)
        #if matrix empty, pass
        if precombat_features:
            precombat_features = np.array(precombat_features)
            site_scanner = np.array(site_scanner)
            dict_combat = neuroCombatFromTraining(dat=precombat_features.T, batch=site_scanner, estimates=combat_estimates)

            post_combat_feature_name = self.feat.combat_feat(feature_name)
            print("Combat finished \n Saving data")
            self.save_cohort_features(post_combat_feature_name, dict_combat["data"].T, np.array(self.subject_ids))
        else:
            print('No data to combat harmonised')
            pass
        
    def save_cohort_features(self, feature_name, features, subject_ids, hemis=["lh", "rh"]):
        assert len(features) == len(subject_ids)
        for s, subject in enumerate(subject_ids):
            subj = MeldSubject(subject, cohort=self.cohort)
            subj.write_feature_values(feature_name, features[s], hemis=hemis, hdf5_file_root=self.write_hdf5_file_root)

    def remove_isolated_subs(self, covars, precombat_features):
        """remove subjects where they are sole examples from the site (for FLAIR)"""

        df = pd.DataFrame(covars.groupby("site_scanner").count()["ages"])
        single_subject_sites = list(df.index[covars.groupby("site_scanner").count()["ages"] == 1])
        mask = np.zeros(len(covars)).astype(bool)
        for site_scan in single_subject_sites:
            mask += covars.site_scanner == site_scan
        precombat_features = precombat_features[~mask]
        covars = covars[~mask]
        return covars, precombat_features

    def correct_sulc_freesurfer(self, vals):
        """this function normalized sulcul feature in cm when values are in mm (depending on Freesurfer version used)"""
        if np.mean(vals, axis=0) > 0.2:
            vals = vals / 10
        else:
            pass
        return vals

    @property
    def calibration_smoothing(self):
        """caliration curve for smoothing surface mesh'"""
        if self._calibration_smoothing is None:
            p = os.path.join(self.data_dir, SMOOTH_CALIB_FILE)
            coords, faces = nb.freesurfer.io.read_geometry(p)
            line, model = mt.calibrate_smoothing(coords, faces, start_v=125000, n_iter=70)
            self._calibration_smoothing = (line, model)
        return self._calibration_smoothing

    def smooth_data(self, feature, fwhm):
        """smooth features with given fwhm for all subject and save in new hdf5 file"""
        # create smooth name
        feature_smooth = self.feat.smooth_feat(feature, fwhm)
        # initialise
        neighbours = self.cohort.neighbours
        subject_include = []
        vals_matrix_lh = []
        vals_matrix_rh = []
        for id_sub in self.subject_ids:
            print(id_sub)
            # create subject object
            subj = MeldSubject(id_sub, cohort=self.cohort)
            # smooth data only if the feature exist
            if subj.has_features(feature):
                # load feature's value for this subject
                vals_lh = subj.load_feature_values(feature, hemi="lh")
                vals_rh = subj.load_feature_values(feature, hemi="rh")
                # harmonise sulcus data from freesurfer v5 and v6
                if feature == ".on_lh.sulc.mgh":
                    vals_lh = self.correct_sulc_freesurfer(vals_lh)
                    vals_rh = self.correct_sulc_freesurfer(vals_rh)
                vals_matrix_lh.append(vals_lh)
                vals_matrix_rh.append(vals_rh)
                subject_include.append(id_sub)
            else:
                print("feature {} does not exist for subject {}".format(feature, id_sub))
        #if matrix is empty, do nothing
        if not vals_matrix_lh:
            pass
        else:
            # smoothed data if fwhm
            vals_matrix_lh = np.array(vals_matrix_lh)
            vals_matrix_rh = np.array(vals_matrix_rh)
            if fwhm:
                # find number iteration from calibration smoothing
                x, y = self.calibration_smoothing
                idx = (np.abs(y - fwhm)).argmin()
                n_iter = int(np.round(x[idx]))
                print(f"smoothing with {n_iter} iterations ...")
                vals_matrix_lh = mt.smooth_array(
                    vals_matrix_lh.T, neighbours, n_iter=n_iter, cortex_mask=self.cohort.cortex_mask
                )
                vals_matrix_rh = mt.smooth_array(
                    vals_matrix_rh.T, neighbours, n_iter=n_iter, cortex_mask=self.cohort.cortex_mask
                )
            else:
                print("no smoothing")
                vals_matrix_lh = vals_matrix_lh.T
                vals_matrix_rh = vals_matrix_rh.T

            smooth_vals_hemis = np.array(
                np.hstack([vals_matrix_lh[self.cohort.cortex_mask].T, vals_matrix_rh[self.cohort.cortex_mask].T])
            )
            # write features in hdf5
            print("Smoothing finished \n Saving data")
            self.save_cohort_features(feature_smooth, smooth_vals_hemis, np.array(subject_include))
            return smooth_vals_hemis

    def define_atlas(self):
        atlas = nb.freesurfer.io.read_annot(os.path.join(BASE_PATH, DK_ATLAS_FILE))
        self.vertex_i = np.array(atlas[0]) - 1000  # subtract 1000 to line up vertex
        self.rois_prop = [
            np.count_nonzero(self.vertex_i == x) for x in set(self.vertex_i)
        ]  # proportion of vertex per rois
        rois = [x.decode("utf8") for x in atlas[2]]  # extract rois label from the atlas
        rois = dict(zip(rois, range(len(rois))))  # extract rois label from the atlas
        rois.pop("unknown")  # roi not part of the cortex
        rois.pop("corpuscallosum")  # roi not part of the cortex
        self.rois = rois

    def get_key(self, dic, val):
        # function to return key for any value in dictionnary
        for key, value in dic.items():
            if val == value:
                return key
        return "No key for value {}".format(val)

    def create_features_rois_matrix(self, feature, hemi, save_matrix=False):
        """Compute matrix with average feature values per ROIS for each subject"""
        self.define_atlas()
        matrix = pd.DataFrame()
        for id_sub in self.subject_ids:
            # create subject object
            subj = MeldSubject(id_sub, cohort=self.cohort)
            # create a dictionnary to store values for each row of the matrix
            row = {}
            row["ID"] = subj.subject_id
            row["site"] = subj.site_code
            row["scanner"] = subj.scanner
            row["group"] = subj.group
            row["FLAIR"] = subj.has_flair

            # remove rois where more than 25% of vertex are lesional
            rois_s = self.rois.copy()
            if subj.has_lesion == True:
                lesion = subj.load_feature_values(".on_lh.lesion.mgh", hemi)
                rois_lesion = list(self.vertex_i[lesion == 1])
                rois_lesion = [[x, rois_lesion.count(x)] for x in set(rois_lesion) if x != 0]
                for ind, num in rois_lesion:
                    if num / self.rois_prop[ind - 1] * 100 > 30:
                        print("remove {}".format(ind))
                        rois_s.pop(self.get_key(rois_s, ind))

            # compute average feature per rois
            if subj.has_features(feature):
                feat_values = subj.load_feature_values(feature, hemi)
                # correct sulcus values if in mm
                if feature == ".on_lh.sulc.mgh":
                    feat_values = self.correct_sulc_freesurfer(feat_values)
                # calculate mean thickness & std per ROI
                for roi, r in rois_s.items():
                    row[roi + "." + feature] = np.mean(feat_values[self.vertex_i == r])
            else:
                pass
            #                 print('feature {} does not exist for subject {}'.format(feature,id_sub))
            # add row to matrix
            matrix = matrix.append(pd.DataFrame([row]), ignore_index=True)
        # save matrix
        if save_matrix == True:
            file = os.path.join(BASE_PATH, "matrix_QC_{}.csv".format(hemi))
            matrix.to_csv(file)
            print("Matrix with average features/ROIs for all subject can be found at {}".format(file))

        return matrix

    def get_outlier_feature(self, feature, hemi):
        """return array of 1 (feature is outlier) and 0 (feature is not outlier) for list of subjects"""
        df = self.create_features_rois_matrix(feature, hemi, save_matrix=False)
        # define if feature is outlier or not
        ids = df.groupby(["site", "scanner"])
        outliers = []
        subjects = []
        for index, row in df.iterrows():
            print(row["ID"])
            subjects.append(row["ID"])
            group = ids.get_group((row["site"], row["scanner"]))
            # warning if not enough subjects per site/scanner
            if len(group.index) <= 6:
                print(
                    "WARNING : only {} subjects in site {} and scanner {}".format(
                        len(group.index), row["site"], row["scanner"]
                    )
                )
            # find upper and lower limit for each ROIS
            lower_lim = group.mean() - 2.698 * group.std()
            upper_lim = group.mean() + 2.698 * group.std()
            # check if subject out of specs
            keys_feat = [key for key in set(df) if feature in key]
            count_out_rois = 0
            for key in keys_feat:
                if (row[key] <= lower_lim[key]) or (row[key] >= upper_lim[key]):
                    count_out_rois += 1
                else:
                    pass
            # decide if feature is outliers based on number or outliers ROIs
            if count_out_rois >= 10:
                outliers.append(1)
            else:
                outliers.append(0)
        return outliers, df[["ID", "FLAIR"]].copy()

    def find_outliers(self, features, output_file=None):
        """return list of outliers pre-combat"""
        # Find how many features are outliers per subjec
        tot_out_feat = []
        for feature in features:
            print("Process outlier for feature {}".format(feature))
            out_feat_lh, df = self.get_outlier_feature(feature, "lh")
            out_feat_rh, _ = self.get_outlier_feature(feature, "rh")
            tot_out_feat.append(out_feat_lh)
            tot_out_feat.append(out_feat_rh)
        df["tot_out_feat"] = np.array(tot_out_feat).sum(axis=0)

        # different conditions to define if subject is an outlier
        outliers = df[(df["FLAIR"] == True) & (df["tot_out_feat"] >= 3)]["ID"]
        outliers = outliers.append(df[(df["FLAIR"] == False) & (df["tot_out_feat"] >= 2)]["ID"])
        # save outliers
        if output_file is not None:
            file_path = os.path.join(BASE_PATH, output_file)
            print("list of outliers saved at {}".format(file_path))
            outliers.to_csv(file_path, index=False)

        return outliers

    def compute_mean_std_controls(self, feature, cohort, asym=False, params_norm=None):
        """retrieve controls from given cohort, intra-normalise feature and return mean and std for inter-normalisation"""
        controls_ids = cohort.get_subject_ids(group="control")
        # Give warning if list of controls empty
        if len(controls_ids) == 0:
            print("WARNING: there is no controls in this cohort to do inter-normalisation")
        vals_array = []
        included_subj = []
        for id_sub in controls_ids:
            # create subject object
            subj = MeldSubject(id_sub, cohort=cohort)
            # append data to compute mean and std if feature exist
            if subj.has_features(feature):
                # load feature's value for this subject
                vals_lh = subj.load_feature_values(feature, hemi="lh")
                vals_rh = subj.load_feature_values(feature, hemi="rh")
                vals = np.array(np.hstack([vals_lh[cohort.cortex_mask], vals_rh[cohort.cortex_mask]]))
                # intra subject normalisation asym
                intra_norm = np.array(self.normalise(vals))
                # Calculate asymmetry
                if asym == True:
                    intra_norm = self.compute_asym(intra_norm)
                    names_save = [f'mean.asym',f'std.asym']
                else:
                    names_save = [f'mean',f'std']   
                vals_array.append(intra_norm)
                included_subj.append(id_sub)
            else:
                pass
        print("Compute mean and std from {} controls".format(len(included_subj)))
        # get mean and std from controls
        params = {}
        params[names_save[0]] = np.mean(vals_array, axis=0)
        params[names_save[1]] = np.std(vals_array, axis=0)
        # save parameters in hdf5
        if params_norm!=None:
            self.save_norm_combat_parameters(feature, params, params_norm)
        return params[names_save[0]], params[names_save[1]]
    
                    
    def normalise(self, data):
        if len(data.shape) == 1:
            data[:, np.newaxis]
        mean_intra = np.mean(data, axis=0)
        std_intra = np.std(data, axis=0)
        intra_norm = (data - mean_intra) / std_intra
        return intra_norm

    def compute_asym(self, intra_norm):
        intra_lh = intra_norm[: int(len(intra_norm) / 2)]
        intra_rh = intra_norm[int(len(intra_norm) / 2) :]
        lh_asym = intra_lh - intra_rh
        rh_asym = intra_rh - intra_lh
        asym = np.hstack([lh_asym, rh_asym])
        return asym

    def intra_inter_subject(self, feature, cohort_for_norm=None, params_norm=None):
        """perform intra normalisation (within subject) and
        inter-normalisation (between subjects relative to controls)"""
        feature_norm = self.feat.norm_feat(feature)
        # loop over subjects
        vals_array = []
        included_subjects = np.zeros(len(self.subject_ids), dtype=bool)
        controls_subjects = np.zeros(len(self.subject_ids), dtype=bool)
        for k, id_sub in enumerate(self.subject_ids):
            # create subject object
            subj = MeldSubject(id_sub, cohort=self.cohort)
            if subj.has_features(feature):
                included_subjects[k] = True
                if subj.group == "control":
                    controls_subjects[k] = True
                else:
                    controls_subjects[k] = False
                # load feature's value for this subject
                vals_lh = subj.load_feature_values(feature, hemi="lh")
                vals_rh = subj.load_feature_values(feature, hemi="rh")
                vals = np.array(np.hstack([vals_lh[self.cohort.cortex_mask], vals_rh[self.cohort.cortex_mask]]))
                # intra subject normalisation asym
                intra_norm = np.array(self.normalise(vals))
                vals_array.append(intra_norm)
            else:
                print("exlude subject {}".format(id_sub))
                included_subjects[k] = False
                controls_subjects[k] = False
        if vals_array:
            vals_array = np.array(vals_array)
            # remove exclude subjects
            controls_subjects = np.array(controls_subjects)[included_subjects]
            included_subjects = np.array(self.subject_ids)[included_subjects]
            # normalise by controls
            if cohort_for_norm is not None:
                print("Use other cohort for normalisation")
                mean_c, std_c = self.compute_mean_std_controls(feature, cohort=cohort_for_norm, 
                                                               params_norm=os.path.join(BASE_PATH, NORM_CONTROLS_PARAMS_FILE))
            else:
                if params_norm is not None:
                    params = self.read_norm_combat_parameters(feature, params_norm)
                    mean_c = params['mean']
                    std_c = params['std']
                else : 
                    print(
                        "Use same cohort for normalisation \n Compute mean and std from {} controls".format(
                            len(controls_subjects)
                        )
                    )
                    mean_c = np.mean(vals_array[controls_subjects], axis=0)
                    std_c = np.std(vals_array[controls_subjects], axis=0)
            vals_combat = (vals_array - mean_c) / std_c
            # save subject
            print("Normalisation finished \nSaving data")
            self.save_cohort_features(feature_norm, vals_combat, included_subjects)
        else:
            print('No data to normalise')
            pass
        
    def asymmetry_subject(self, feature, cohort_for_norm=None, params_norm=None):
        """perform intra normalisation (within subject) and
        inter-normalisation (between subjects relative to controls) and asymetry between hemispheres"""
        feature_asym = self.feat.asym_feat(feature)
        # loop over subjects
        vals_asym_array = []
        included_subjects = np.zeros(len(self.subject_ids), dtype=bool)
        controls_subjects = np.zeros(len(self.subject_ids), dtype=bool)
        for k, id_sub in enumerate(self.subject_ids):
            # create subject object
            subj = MeldSubject(id_sub, cohort=self.cohort)
            if subj.has_features(feature):
                included_subjects[k] = True
                if subj.group == "control":
                    controls_subjects[k] = True
                else:
                    controls_subjects[k] = False

                # load feature's value for this subject
                vals_lh = subj.load_feature_values(feature, hemi="lh")
                vals_rh = subj.load_feature_values(feature, hemi="rh")
                vals = np.array(np.hstack([vals_lh[self.cohort.cortex_mask], vals_rh[self.cohort.cortex_mask]]))
                # intra subject normalisation asym
                intra_norm = np.array(self.normalise(vals))
                # Calculate asymmetry
                vals_asym = self.compute_asym(intra_norm)
                vals_asym_array.append(vals_asym)
            else:
                print("exlude subject {}".format(id_sub))
                included_subjects[k] = False
                controls_subjects[k] = False
        if vals_asym_array :
            vals_asym_array = np.array(vals_asym_array)
            # remove exclude subjects
            controls_subjects = np.array(controls_subjects)[included_subjects]
            included_subjects = np.array(self.subject_ids)[included_subjects]
            # normalise by controls
            if cohort_for_norm is not None:
                print("Use other cohort for normalisation")
                mean_c, std_c = self.compute_mean_std_controls(feature, cohort=cohort_for_norm, asym=True, 
                                                               params_norm=os.path.join(BASE_PATH, NORM_CONTROLS_PARAMS_FILE))
            else:
                if params_norm is not None:
                    params = self.read_norm_combat_parameters(feature, params_norm)
                    mean_c = params['mean.asym']
                    std_c = params['std.asym']
                else:
                    print(
                        "Use same cohort for normalisation \n Compute mean and std from {} controls".format(
                            controls_subjects.sum()
                        )
                    )
                    mean_c = np.mean(vals_asym_array[controls_subjects], axis=0)
                    std_c = np.std(vals_asym_array[controls_subjects], axis=0)
            asym_combat = (vals_asym_array - mean_c) / std_c
            # save subject
            print("Asym finished \nSaving data")
            self.save_cohort_features(feature_asym, asym_combat, included_subjects)
        else:
            print('No data to do asym')
            pass
        
    def compute_mean_std(self, feature, cohort):
        """get mean and std of all brain for the given cohort and save parameters"""
        cohort_ids = cohort.get_subject_ids(group="both")
        # Give warning if list of controls empty
        if len(cohort_ids) == 0:
            print("WARNING: there is no subject in this cohort")
        vals_array = []
        included_subj = []
        for id_sub in cohort_ids:
            # create subject object
            subj = MeldSubject(id_sub, cohort=cohort)
            # append data to compute mean and std if feature exist
            if subj.has_features(feature):
                # load feature's value for this subject
                vals_lh = subj.load_feature_values(feature, hemi="lh")
                vals_rh = subj.load_feature_values(feature, hemi="rh")
                vals = np.array(np.hstack([vals_lh[cohort.cortex_mask], vals_rh[cohort.cortex_mask]]))
                vals_array.append(vals)
                included_subj.append(id_sub)
            else:
                pass
        #                 print('feature {} does not exist for subject {}'.format(feature,id_sub))
        print("Compute mean and std from {} subject".format(len(included_subj)))
        # get mean and std
        vals_array = np.matrix(vals_array)
        mean = (vals_array.flatten()).mean()
        std = (vals_array.flatten()).std()
        # save in json
        data = {}
        data["{}".format(feature)] = {
            "mean": str(mean),
            "std": str(std),
        }
        # create or re-write json file
        file = os.path.join(self.data_dir, self.write_hdf5_file_root)
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


class Feature:
    def __init__(self):
        """Class to define feature name"""
        pass

    def raw_feat(self, feature):
        self._raw_feat = feature
        return self._raw_feat

    def smooth_feat(self, feature, smoother=None):
        if smoother != None:
            smooth_part = "sm" + str(int(smoother))
            list_name = feature.split(".")
            new_name = list(chain.from_iterable([list_name[0:-1], [smooth_part, list_name[-1]]]))
            self._smooth_feat = ".".join(new_name)
        else:
            self._smooth_feat = feature
        return self._smooth_feat

    def combat_feat(self, feature):
        return "".join([".combat", feature])

    def norm_feat(self, feature):
        self._norm_feat = "".join([".inter_z.intra_z", feature])
        return self._norm_feat

    def asym_feat(self, feature):
        self._asym_feat = "".join([".inter_z.asym.intra_z", feature])
        return self._asym_feat

    def list_feat(self):
        self._list_feat = [self.smooth, self.combat, self.norm, self.asym]
        return self._list_feat
