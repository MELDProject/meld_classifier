from meld_classifier.paths import NVERT
import meld_classifier.mesh_tools as mt
from meld_classifier.meld_cohort import MeldSubject
import meld_classifier.meld_plotting as mpt
from meld_classifier.dataset import load_combined_hemisphere_data
from meld_classifier.saliency import integrated_gradients, vanilla_backprop

import nibabel as nb
import numpy as np
import os
import matplotlib.pyplot as plt
import matplotlib_surface_plotting.matplotlib_surface_plotting as msp
import pandas as pd
import scipy

import json
import logging
from functools import partial
import shutil
import tensorflow as tf
import h5py


class Evaluator:
    """
    Evaluate neural network model.
    Predicts train/val/test subjects, or selected single subjects.
    Creates evaluation plots.

    Args:
        experiment (Experiment object): experiment to evaluate
        mode (str, optional): train, val, or test. determines which subject_ids are loaded and which control_ids
        checkpoint_path (str, optional): path to the checkpoint file
            (if not given, is inferred from experiment path + experiment name)
        make_images (bool, optional): flag for plotting results of brain surface
        make_prediction_space (bool, optional): flag for making the prediction space
        no_load_predict_data (bool, optional): if set, do not automatically evaluate on given set of subjects. Useful for evaluation of single subjects and for quick creation of ensemble.
        subject_ids (optional, list): list of subject_ids to predict.
            Only used when mode == 'inference'
        save_dir = directory to save plots and results. If no directory passed it will be the experiment path.

    """

    def __init__(
        self,
        experiment,
        mode="test",
        checkpoint_path=None,
        make_images=False,
        make_prediction_space=False,
        subject_ids=None,
        save_dir=None,
    ):
        # set class params
        self.log = logging.getLogger(__name__)
        self.experiment = experiment
        assert mode in (
            "test",
            "val",
            "train",
            "inference",
        ), "mode needs to be either test or val or train or inference"
        self.mode = mode
        self.make_images = make_images
        self.make_prediction_space = make_prediction_space
        self.threshold = self.experiment.network_parameters["optimal_threshold"]
        # if threshold was not optimized, use 0.5
        if not isinstance(self.threshold, float):
            self.threshold = 0.5
        self.min_area_threshold = self.experiment.data_parameters["min_area_threshold"]
        self.log.info("Evalution {}, {}".format(self.mode, self.threshold))

        # ensure that features are saved in experiment.data_parameters
        experiment.get_features()

        # set subject_ids
        if mode == "inference":
            assert subject_ids is not None, "for mode inference need to define subject ids"
            self.combined_ids = subject_ids
            self.log.info(
                f"Initialized Evaluation with mode {self.mode}, thresh {self.threshold} on {len(self.combined_ids)} subjects"
            )
        else:
            train_ids, val_ids, test_ids = experiment.get_train_val_test_ids()
            if mode == "train":
                subject_ids = train_ids
            elif mode == "val":
                subject_ids = val_ids
            elif mode == "test":
                subject_ids = test_ids
            self.patient_ids, self.control_ids = self.divide_subjects(subject_ids, n_controls=5)
            self.combined_ids = list(self.patient_ids) + list(self.control_ids)
            if mode == "train":  
                 #if we're training, i.e. optimising threshold, this is only done using patients
                self.combined_ids = self.patient_ids
            self.log.info(
                f"Initialized Evaluation with mode {self.mode}, thresh {self.threshold} on {len(self.patient_ids)} patients and {len(self.control_ids)} controls"
            )

        # get initialised model instance
        # get checkpoint_path
        if checkpoint_path is None:
            checkpoint_path = os.path.join(experiment.path, "models", experiment.name)
        # build model - replace dropout with mc_dropout for building model. mc_dropout is later tested again for use mc prediction
        self.experiment.network_parameters["dropout"] = self.experiment.network_parameters.get("mc_dropout", 0)
        self.experiment.load_model(checkpoint_path=checkpoint_path)

        # data_dictionary, filled later with predicted data using load_predict_data()
        self.data_dictionary = None

        # Initialised directory to save results and plots
        if save_dir is None:
            self.save_dir = self.experiment.path
        else:
            self.save_dir = save_dir

    def evaluate(self):
        """
        Evaluate the model.
        Runs `self.get_metrics(); self.plot_prediction_space(); self.plot_subjects_prediction()`
        and saves images to results folder.
        """
        # need to load and predict data?
        if self.data_dictionary is None:
            self.load_predict_data()
        # per vertex and per subject metrics
        self.get_metrics()
        # make PCA plots
        if self.make_prediction_space:
            self.plot_prediction_space()
        # make images if asked for
        if self.make_images:
            self.plot_subjects_prediction()
        return

    def optimise_threshold(self, plot_curve=False):
        """optimise the per-patient decision threshold based on dice index.

        creates attributes: threshold, optimal_dice, dice_per_threshold, per_subject_dice
        plots threshold-dice curve if plot_curve

        Returns:
            threshold, optimal_dice
        """
        if self.data_dictionary is None:
            self.load_predict_data()
        # load surface data and create adjacency matrix
        self.log.info("calculating optimal threshold")
        # calculate scores for every threshold
        n_thresholds = 51
        thresholds = np.linspace(0, 1, n_thresholds)
        per_subject_dice = {}
        # just run this on patients
        for subj_id in self.patient_ids:
            subj = MeldSubject(subj_id, self.experiment.cohort)
            if subj.is_patient:
                self.log.debug("subject {}".format(subj_id))
                # import subject predictions
                predictions = self.experiment.cohort.split_hemispheres(self.data_dictionary[subj_id]["result"])
                labels = self.experiment.cohort.split_hemispheres(self.data_dictionary[subj_id]["input_labels"])
                stats = np.zeros((3, n_thresholds))
                dice_parameters = {
                    "TP": np.zeros(n_thresholds),
                    "FP": np.zeros(n_thresholds),
                    "FN": np.zeros(n_thresholds),
                }
                for hemi in ["left", "right"]:
                    # get predictions and labels for that hemisphere
                    pred = predictions[hemi]
                    label = labels[hemi].astype(bool)
                    for k, threshold in enumerate(thresholds):
                        # calculate predictions above threshold.
                        mask = pred >= threshold

                        dice_parameters["TP"][k] += np.sum(mask * label)
                        dice_parameters["FP"][k] += np.sum(mask * ~label)
                        dice_parameters["FN"][k] += np.sum(~mask * label)

                # per subject dice
                per_subject_dice[subj_id] = (2 * dice_parameters["TP"]) / (
                    (2 * dice_parameters["TP"]) + dice_parameters["FP"] + dice_parameters["FN"]
                )

        dice_per_threshold = np.array([list(el) for el in per_subject_dice.values()]).mean(axis=0)
        threshold, optimal_dice = thresholds[np.argmax(dice_per_threshold)], np.max(dice_per_threshold)
        # store results as attributes
        self.threshold = threshold
        self.optimal_dice = optimal_dice
        self.dice_per_threshold = dice_per_threshold
        self.per_subject_dice = per_subject_dice

        # plot
        if plot_curve:
            plt.figure(figsize=(6, 3))
            plt.subplot(1, 1, 1)
            plt.plot(self.dice_per_threshold)
            # vertical line at maximum dice index
            plt.plot(
                [np.argmax(self.dice_per_threshold), np.argmax(self.dice_per_threshold)],
                [0, np.max(self.dice_per_threshold)],
            )
            plt.xticks(
                [0, 25, np.argmax(self.dice_per_threshold), 50],
                ["0", "0.5", thresholds[np.argmax(self.dice_per_threshold)], "1.0"],
            )

            plt.savefig(
                os.path.join(
                    self.save_dir,
                    "results",
                    "images",
                    f"sensitivity_specificity_curve_{self.experiment.name}_2.png",
                )
            )
            plt.close("all")

        return self.threshold, self.optimal_dice

    def cluster_and_area_threshold(self, mask, island_count=0):
        """cluster predictions and threshold based on min_area_threshold

        Args:
            mask: boolean mask of the per-vertex lesion predictions to cluster"""
        n_comp, labels = scipy.sparse.csgraph.connected_components(self.experiment.cohort.adj_mat[mask][:, mask])
        islands = np.zeros(len(mask))
        # only include islands larger than minimum size.
        for island_index in np.arange(n_comp):
            include_vec = labels == island_index
            size = np.sum(include_vec)
            if size >= self.min_area_threshold:
                island_count += 1
                island_mask = mask.copy()
                island_mask[mask] = include_vec
                islands[island_mask] = island_count
        return islands

    def calculate_island_stats(self, all_islands, label):
        """calculate if overlap, any additional clusters,
        and how many additional clusters"""
        overlap = np.any((all_islands > 0) * label, axis=1).astype(int)
        # number of clusters is the maximum index. Only true clusters are ordered by size.
        n_clusters = np.max(all_islands, axis=1) - overlap
        return np.array([overlap, (n_clusters > 0).astype(int), n_clusters])

    def optimum_threshold_youden_index(self, sensitivity, specificity, thresholds):
        """calculate threshold for optimum index"""
        youden_index = (sensitivity + specificity) - 1
        optimum_threshold = thresholds[np.argmax(youden_index)]
        # option to estimate optimum threshold using interpolation.
        return optimum_threshold, np.max(youden_index)

    def get_metrics(self, return_values=False):
        """
        calculate per vertex statistics at 0.5 and at optimal threshold
        """
        self.log.info("calculating per vertex statistics at 0.5 and at optimal threshold")
        from sklearn.metrics import roc_curve

        labels = []
        predictions = []
        for subj_id in self.patient_ids:
            labels.append(self.data_dictionary[subj_id]["input_labels"])
            predictions.append(self.data_dictionary[subj_id]["result"])
        labels = np.array(labels).ravel().astype(np.float32)
        predictions = np.array(predictions).ravel().astype(np.float32)
        results_stats = np.zeros((2, 9))
        results_stats[0, 0] = 0.5
        results_stats[0, 1:4] = self.calculate_metrics(labels, predictions)
        results_stats[1, 0] = self.threshold
        results_stats[1, 1:4] = self.calculate_metrics(labels, (predictions > self.threshold).astype(float))
        # evaluate on both 0.5 and 'optimal threshold'
        for t, threshold in enumerate([0.5, self.threshold]):
            sensitivity = 0
            n_clusters_patients = []
            n_clusters_controls = []
            self.patient_results = {}
            self.control_results = {}
            for i, subj_id in enumerate(self.combined_ids):
                subj = MeldSubject(subj_id, self.experiment.cohort)
                predictions = self.experiment.cohort.split_hemispheres(self.data_dictionary[subj_id]["result"])
                labels = self.experiment.cohort.split_hemispheres(
                    self.data_dictionary[subj_id]["input_labels"].astype(int)
                )
                # check if number of correct lesional predictions is above min vertex overlap
                n_clusters = 0
                detected = 0
                self.data_dictionary[subj_id]["cluster_thresholded"] = {}
                patient_dice_vars = {"TP": 0, "FP": 0, "FN": 0}

                for h, hemi in enumerate(["left", "right"]):
                    mask = predictions[hemi] >= threshold
                    islands = self.cluster_and_area_threshold(mask)
                    if t == 1:
                        # save clustered and thresholded prediction
                        self.data_dictionary[subj_id]["cluster_thresholded"][hemi] = islands
                    if subj.is_patient:
                        sensitivity += (np.any((islands > 0) * labels[hemi])).astype(int)
                        # subtract number of islands overlapping lesion as being true positives
                        n_clusters -= (np.any((islands > 0) * labels[hemi])).astype(int)
                        label = labels[hemi].astype(bool)
                        patient_dice_vars["TP"] += np.sum(mask * label)
                        patient_dice_vars["FP"] += np.sum(mask * ~label)
                        patient_dice_vars["FN"] += np.sum(~mask * label)
                    n_clusters += np.max(islands)
                    detected += int(np.any((islands > 0) * labels[hemi]))
                if subj.is_patient:
                    patient_dice = (2 * patient_dice_vars["TP"]) / (
                        (2 * patient_dice_vars["TP"]) + patient_dice_vars["FP"] + patient_dice_vars["FN"]
                    )
                    self.patient_results[subj_id] = [detected, int(n_clusters), patient_dice]
                    n_clusters_patients.append(n_clusters)
                else:
                    self.control_results[subj_id] = [int(n_clusters > 0), int(n_clusters)]
                    n_clusters_controls.append(n_clusters)
            results_stats[t, 4] = sensitivity / len(n_clusters_patients)
            results_stats[t, 5] = 1 - np.mean((np.array(n_clusters_controls) > 0).astype(float))
            # Youden index
            results_stats[t, 6] = results_stats[t, 4] + results_stats[t, 5] - 1
            results_stats[t, 7] = np.median(np.array(n_clusters_patients))
            results_stats[t, 8] = np.median(np.array(n_clusters_controls))

            # save thresholded and clustered prediction for plotting
            # TODO filenames incorrect
            fnames = ["0.5", "optimal"]
            json_filename = os.path.join(
                self.save_dir, "results", "per_subject_{}_{}.json".format(self.experiment.name, fnames[t])
            )
            json_results = {"patients": self.patient_results, "controls": self.control_results}
            save_json(json_filename, json_results)

        df = pd.DataFrame(results_stats)
        df.columns = [
            "threshold",
            "recall",
            "precision",
            "specificity",
            "subject_sensitivity",
            "subject_specificity",
            "youden_index",
            "patient_median_cluster",
            "control_median_cluster",
        ]

        filename = os.path.join(self.save_dir, "results", "test_results_{}.csv".format(self.experiment.name))
        # np.savetxt(filename,per_vertex_results,fmt='%.2f')
        df.to_csv(filename, index=False, float_format="%.2f")
        if return_values:
            return df

    # now used for predict single subject
    def threshold_and_cluster(self, data_dictionary=None):
        return_dict = data_dictionary is not None
        if data_dictionary is None:
            data_dictionary = self.data_dictionary
        for subj_id, data in data_dictionary.items():
            data["cluster_thresholded"] = {}
            predictions = self.experiment.cohort.split_hemispheres(data["result"])
            island_count = 0
            for h, hemi in enumerate(["left", "right"]):
                mask = predictions[hemi] >= self.threshold
                islands = self.cluster_and_area_threshold(mask, island_count=island_count)
                data["cluster_thresholded"][hemi] = islands
                island_count += np.max(islands)
        if return_dict:
            return data_dictionary
        else:
            self.data_dictionary = data_dictionary

    def calculate_metrics(self, y_true, y_pred):
        true_positives = np.sum(y_true * np.round(y_pred))
        true_negatives = np.sum(np.logical_and(y_true == 0, (np.round(y_pred) == 0)))
        all_positives = np.sum(np.round(y_true))
        predicted_positives = np.sum(np.round(y_pred))
        all_negatives = np.sum(y_true == 0)
        recall = true_positives / all_positives
        precision = true_positives / predicted_positives
        specificity = true_negatives / all_negatives
        return recall, precision, specificity

    def plot_prediction_space(self):
        """plot PCA decomposition of features.
        Use inverse transform to map dense prediction from reduced space
        """
        from sklearn.decomposition import PCA
        import matplotlib.colors as mcolors

        filename = os.path.join(self.save_dir, "results", "images", f"prediction_space_{self.experiment.name}.png")
        pca_m = PCA(n_components=2)
        # ravel features into big matrix
        input_data = []
        labels = []
        subject_predictions = []
        for subject in self.data_dictionary.keys():
            input_data.append(self.data_dictionary[subject]["input_features"])
            labels.append(self.data_dictionary[subject]["input_labels"])
            subject_predictions.append(self.data_dictionary[subject]["result"])
        input_data = np.vstack(input_data)
        labels = np.stack(labels).ravel()
        subject_predictions = np.stack(subject_predictions).ravel()
        pca_outputs = pca_m.fit_transform(input_data)
        # get limits
        xmin, xmax, ymin, ymax = (
            np.min(pca_outputs[:, 0]),
            np.max(pca_outputs[:, 0]),
            np.min(pca_outputs[:, 1]),
            np.max(pca_outputs[:, 1]),
        )
        xsteps = np.linspace(xmin, xmax, 100)
        ysteps = np.linspace(ymin, ymax, 100)
        x_s, y_s = np.meshgrid(xsteps, ysteps)
        inverted = pca_m.inverse_transform(np.array([x_s.ravel(), y_s.ravel()]).T)
        predictions = self.experiment.model.predict(inverted, batch_size=self.experiment.data_parameters["batch_size"])
        subject_predictions = (subject_predictions >= self.threshold).astype(int)
        four_colours = subject_predictions * 2 + labels
        subset = np.random.choice(np.arange(len(four_colours)), 10000)
        subset_mask = np.zeros(len(four_colours)).astype(bool)
        subset_mask[subset] = 1
        plt.figure()
        # make colormap asymmetric about optimal threshold
        vcenter = np.clip(self.threshold, 0.01, 0.99)
        offset = mcolors.TwoSlopeNorm(vmin=0, vcenter=vcenter, vmax=1)
        plt.scatter(x_s.ravel(), y_s.ravel(), c=offset(predictions.ravel()), cmap="bwr", vmin=0, vmax=1)
        colours = ["brown", "pink", "gray", "olive"]
        labels = ["tn", "fn", "fp", "tp"]
        for k, colour in enumerate(colours):
            vertices = np.logical_and(subset_mask, four_colours == k)
            plt.scatter(pca_outputs[vertices, 0], pca_outputs[vertices, 1], c=colour, alpha=1, label=labels[k])
        plt.legend()
        plt.savefig(filename)
        plt.close()
        return

    def plot_subjects_prediction(self, rootfile=None):
        """plot predicted subjects"""
        plt.close("all")

        for subject in self.data_dictionary.keys():
            if rootfile is not None:
                filename = os.path.join(rootfile.format(subject))
            else:
                filename = os.path.join(
                    self.save_dir, "results", "images", "{}_{}.jpg".format(self.experiment.name, subject)
                )

            result = self.data_dictionary[subject]["result"]
            thresholded = self.data_dictionary[subject]["cluster_thresholded"]
            label = self.data_dictionary[subject]["input_labels"]
            result = np.reshape(result, len(result))

            result_hemis = self.experiment.cohort.split_hemispheres(result)
            label_hemis = self.experiment.cohort.split_hemispheres(label)
            msp.plot_surf(
                self.experiment.cohort.surf_partial["coords"],
                self.experiment.cohort.surf_partial["faces"],
                [
                    result_hemis["left"],
                    thresholded["left"],
                    label_hemis["left"],
                    result_hemis["right"],
                    thresholded["right"],
                    label_hemis["right"],
                ],
                rotate=[90, 270],
                filename=filename,
                vmin=0.4,
                vmax=0.6,
            )
            plt.close("all")
        return

    def load_predict_data(self, subject_ids=None):
        """
        load subject data in self.data_dictionary and predict subjects.

        If self.mode is train, input_features are not saved to save memory
        (are not needed for threshold optimization)

        Args:
            subject_ids: if specified, return data_dictionary for these subjects.
                Otherwise predict self.combined_ids and save in self.data_dictionary
        """
        self.log.info("loading data and predicting model")
        return_dict = subject_ids is not None
        data_dictionary = {}
        if subject_ids is None:
            subject_ids = self.combined_ids
        for subj_id in subject_ids:
            subj = MeldSubject(subj_id, self.experiment.cohort)
            features, features_to_ignore = self.experiment.get_features()
            features, labels = load_combined_hemisphere_data(
                subj,
                features=features,
                features_to_ignore=features_to_ignore,
                universal_features=self.experiment.data_parameters.get("universal_features", []),
                demographic_features=self.experiment.data_parameters.get("demographic_features", []),
                num_neighbours=self.experiment.data_parameters.get("num_neighbours", 0),
                normalise=self.experiment.data_parameters.get("normalise", False),
            )

            prediction = self.predict(features)
            data_dictionary[subj_id] = {"input_labels": labels, "result": np.reshape(prediction, len(prediction))}
            if self.mode != "train":
                data_dictionary[subj_id]["input_features"] = features
        if return_dict:
            return data_dictionary
        else:
            self.data_dictionary = data_dictionary

    def saliency(self, data_dictionary=None, method=["integrated_gradients"], target=["pred"], ig_kwargs={}):
        """
        saliency for lesional predictions and gt lesions.

        Can calculate several methods / targets at once
        Updates data_dictionary with keys "{method}_{target}" containing the saliency.

        Args:
            data_dictionary: dict with input_labels, result, and input_features as keys.
                If none, use self.data_dictionary
            method: str or list of str. Options: 'vanilla_backprop', 'integrated_gradients'
            target: str or list of str. Options: 'gt' or 'pred',
                defines with respect to what we calculate the gradients / saliency
            ig_kwargs: kwargs for IntergratedGradients class
        """
        if isinstance(method, str):
            method = [method]
        if isinstance(target, str):
            target = [target]
        return_dict = data_dictionary is not None
        if data_dictionary is None:
            data_dictionary = self.data_dictionary
        for subj_id, data in data_dictionary.items():
            self.log.info(f"calculating saliency for {subj_id}")
            # get features for which want an explanation
            mask = (data["input_labels"] == 1) | (data["result"] >= self.threshold)
            X = data["input_features"][mask]
            for t in target:
                if t == "gt":
                    targets = data["input_labels"][mask]
                elif t == "pred":
                    targets = (data["result"] >= self.threshold)[mask]

                for m in method:
                    data_dictionary[subj_id][f"{m}_{t}"] = np.zeros_like(data["input_features"])
                    if len(X) != 0:
                        if m == "integrated_gradients":
                            res = integrated_gradients(self.experiment.model, X, targets, kwargs=ig_kwargs)
                        elif m == "vanilla_backprop":
                            res = vanilla_backprop(self.experiment.model, X, targets)
                        else:
                            raise NotImplementedError(f"saliency method {m}")
                        data_dictionary[subj_id][f"{m}_{t}"][mask] = res
            break
        if return_dict:
            return data_dictionary
        else:
            self.data_dictionary = data_dictionary

    def load_predict_single_subject(self, subj_id, fold=None, plot=False, saliency=False, suffix=""):
        """"""
        # load and predict data
        data_dictionary = self.load_predict_data(subject_ids=[subj_id])
        subject_features = data_dictionary[subj_id]["input_features"]
        labels = data_dictionary[subj_id]["input_labels"]
        prediction = data_dictionary[subj_id]["result"]
        # threshold prediction
        thresholded = self.threshold_and_cluster(data_dictionary=data_dictionary)[subj_id]["cluster_thresholded"]
        prediction = np.concatenate(
            [thresholded[hemi][self.experiment.cohort.cortex_mask] for hemi in ["left", "right"]]
        )
        # saliency
        if saliency:
            # TODO might want to make method and target configureable parameters
            data_dictionary = self.saliency(
                data_dictionary=data_dictionary, method=["integrated_gradients"], target=["pred"]
            )

        # get stats
        self.per_subject_stats(subj_id, prediction, labels, fold=fold, suffix=suffix)

        # plot prediction
        if plot:
            self.plot_report_prediction(subj_id, subject_features, prediction, labels)

        # save data
        self.save_prediction(subj_id, prediction, suffix=suffix)
        if saliency:
            # save saliency
            dataset_str = "integrated_gradients_pred"
            self.save_prediction(
                subj_id,
                data_dictionary[subj_id][dataset_str],
                dataset_str=dataset_str,
                suffix=suffix,
                dtype=np.float32,
            )

        return

    def save_prediction(self, subject, prediction, dataset_str="prediction", suffix="", dtype=np.uint8):
        """
        saves prediction to {experiment_path}/results/predictions_{experiment_name}.hdf5.
        the hdf5 has the structure (subject_id/hemisphere/prediction).
        and contains predictions for all vertices inside the cortex mask

        dataset_str: name of the dataset to save prediction. If is 'prediction', also saves threshold
        dtype: dtype of the dataset. If none, use dtype of prediction.
        suffix: suffix for the filename for the prediction: "predictions_<self.experiment.name><suffix>.hdf5" is used
        """
        # make sure that give prediction has expected length
        nvert_hemi = len(self.experiment.cohort.cortex_label)
        assert len(prediction) == nvert_hemi * 2
        # get dtype
        if dtype is None:
            dtype = prediction.dtype

        filename = os.path.join(self.save_dir, "results", f"predictions_{self.experiment.name}{suffix}.hdf5")
        if not os.path.isfile(filename):
            mode = "a"
        else:
            mode = "r+"
        done = False
        while not done:
            try:
                with h5py.File(filename, mode=mode) as f:
                    self.log.info(f"saving {dataset_str} for {subject}")
                    for i, hemi in enumerate(["lh", "rh"]):
                        shape = tuple([nvert_hemi] + list(prediction.shape[1:]))
                        # create dataset
                        dset = f.require_dataset(f"{subject}/{hemi}/{dataset_str}", shape=shape, dtype=dtype)
                        # save prediction in dataset
                        dset[:] = prediction[i * nvert_hemi : (i + 1) * nvert_hemi]
                        if dataset_str == "prediction":
                            # save threshold as attribute in dataset
                            dset.attrs["threshold"] = self.threshold
                    done = True
            except OSError:
                done = False

    def per_subject_stats(self, subject, prediction, labels, fold=None, suffix=""):
        """calculate stats per subject.

        TODO: could improve code
        """
        boundary_label = MeldSubject(subject, self.experiment.cohort).load_boundary_zone(max_distance=20)

        # columns: ID, group, detected,  number extra-lesional clusters,border detected
        # calculate stats first
        id_ = subject
        group = "FCD" in subject
        detected = np.logical_and(prediction, labels).any()
        difference = np.setdiff1d(np.unique(prediction), np.unique(prediction[labels]))
        difference = difference[difference > 0]
        n_clusters = len(difference)
        # if not detected, does a cluster overlap boundary zone and if so, how big is the cluster?
        if not detected and prediction[np.logical_and(boundary_label, ~labels)].sum() > 0:
            border_verts = prediction[np.logical_and(boundary_label, ~labels)]
            i, counts = np.unique(border_verts, return_counts=True)
            counts = counts[i > 0]
            i = i[i > 0]
            cluster_index = i[np.argmax(counts)]
            border_detected = np.sum(prediction == cluster_index)
        else:
            border_detected = 0
        sub_df = pd.DataFrame(
            np.array([id_, group, detected, n_clusters, border_detected]).reshape(-1, 1).T,
            columns=["ID", "group", "detected", "n_clusters", "border"],
        )
        filename = os.path.join(self.save_dir, "results", f"test_results{suffix}.csv")
        if fold is not None:
            filename = os.path.join(self.save_dir, "results", f"test_results_{fold}{suffix}.csv")

        if os.path.isfile(filename):
            done = False
            while not done:
                try:
                    df = pd.read_csv(filename, index_col=False)
                    df = df.append(sub_df, ignore_index=True)
                    df.to_csv(filename, index=False)
                    done = True
                except pd.errors.EmptyDataError:
                    done = False
        else:
            sub_df.to_csv(filename, index=False)
        return

    # TODO need to go through
    def plot_report_prediction(self, subj_id, subject_features, prediction, labels):
        """plot prediction reports for a subject in a given output folder"""
        # TODO : needs to better define which features to display
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
        subj = MeldSubject(subj_id, self.experiment.cohort)
        boundary_label = subj.load_boundary_zone(max_distance=20)

        # predict lesion and make plot report for each hemisphere
        try:
            les_hemi = ["lh", "rh"].index(subj.get_lesion_hemisphere())
            les_hemi = ["left", "right"][les_hemi]
        except:
            les_hemi = None
        hemis = ["left", "right"]
        for hemi in hemis:
            if hemi == les_hemi:
                lesion = self.experiment.cohort.split_hemispheres(labels)[hemi]
                boundary_hemi = self.experiment.cohort.split_hemispheres(boundary_label)[hemi]
                lesion_combi = lesion + boundary_hemi
            else:
                lesion_combi = np.zeros(NVERT)
            predictions_hemi = self.experiment.cohort.split_hemispheres(prediction)[hemi]
            data_to_plots = [predictions_hemi]
            feats_hemi = self.experiment.cohort.split_hemispheres(subject_features)[hemi]
            # plot prediction and features
            feature_list = subj.get_feature_list()
            features, _ = self.experiment.get_features()
            overlapping_features = np.intersect1d(feature_list, features_to_plot)
            for f in overlapping_features:
                data_to_plots.append(feats_hemi[:, features.index(f)])
            feature_names = ["prediction"] + list(overlapping_features)

            mpt.plot_single_subject(
                data_to_plots,
                lesion=lesion_combi,
                feature_names=feature_names,
                out_filename=os.path.join(self.save_dir, "results", "images", f"qc_{subj_id}_{hemi}.jpeg"),
            )

    # TODO: is this function used outside of the class?
    # if not, could merge with load_predict_data - but not absolutely necessary
    def predict(self, features, n_predict=20):
        """create prediction for model.
        average of n_predict predictions if mc_dropout is >0"""
        if self.experiment.network_parameters.get("mc_dropout", 0) == 0:
            prediction = self.experiment.model.predict(
                features, batch_size=self.experiment.data_parameters["batch_size"]
            )
        else:
            prediction = np.zeros(features.shape[0])
            for n in np.arange(n_predict):
                single_prediction = self.experiment.model.predict(
                    features, batch_size=self.experiment.data_parameters["batch_size"]
                )
                prediction += single_prediction.ravel() / n_predict
        return prediction

    def divide_subjects(self, subject_ids, n_controls=5):
        """divide subject_ids into patients and controls
        if only trained on patients, controls are added.
        If self.mode is test, controls from test set (defined by dataset csv file) are added.
        If self.mode is train/val, the first/last n_controls are added.
        """
        if self.experiment.data_parameters["group"] == "patient":
            # get n_control ids (not in subject_ids, because training was only on patients)
            # get all valid control ids (with correct features etc)
            data_parameters_copy = self.experiment.data_parameters.copy()
            data_parameters_copy["group"] = "control"
            control_ids = self.experiment.cohort.get_subject_ids(**data_parameters_copy, verbose=False)
            # shuffle control ids
            np.random.seed(5)
            np.random.shuffle(control_ids)
            # filter controls by self.mode (make sure when mode is test, only test controls are used)
            if self.mode == "test":
                _, _, dataset_test_ids = self.experiment.cohort.read_subject_ids_from_dataset()
                control_ids = np.array(control_ids)[np.in1d(control_ids, dataset_test_ids)]
                # select n_controls
                control_ids = control_ids[:n_controls]
            elif self.mode in ("train", "val"):
                _, dataset_trainval_ids, _ = self.experiment.cohort.read_subject_ids_from_dataset()
                control_ids = np.array(control_ids)[np.in1d(control_ids, dataset_trainval_ids)]
                # select n_controls (first n if mode is train, last n if mode is val)
                if len(control_ids) < n_controls * 2:
                    n_controls_train = len(control_ids) // 2
                    n_controls_val = len(control_ids) - n_controls_train
                else:
                    n_controls_train = n_controls_val = n_controls
                if self.mode == "train":
                    control_ids = control_ids[:n_controls_train]
                else:  # mode is val
                    control_ids = control_ids[-n_controls_val:]
                control_ids = list(control_ids)
            if len(control_ids) < n_controls:
                self.log.warning(
                    "only {} controls available for mode {} (requested {})".format(
                        len(control_ids), self.mode, n_controls
                    )
                )
            patient_ids = subject_ids
        else:
            patient_ids = []
            control_ids = []
            for subj_id in subject_ids:
                if MeldSubject(subj_id, self.experiment.cohort).is_patient:
                    patient_ids.append(subj_id)
                else:
                    control_ids.append(subj_id)
        return patient_ids, control_ids


def save_json(json_filename, json_results):
    """
    Save dictionaries to json
    """
    # data_parameters
    json.dump(json_results, open(json_filename, "w"), indent=4)
    return
