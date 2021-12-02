from meld_classifier.meld_cohort import MeldSubject
import meld_classifier.mesh_tools as mt

import numpy as np
import os
import nibabel as nb
from tensorflow.keras.utils import Sequence
import math
import json
import io
import logging
import tensorflow as tf
from meld_classifier.paths import BASE_PATH


def normalise_data(data, features, file_name):
    """normalise all input features by mean and std"""
    import matplotlib.pyplot as plt

    file = os.path.join(BASE_PATH, file_name)
    with open(file, "r") as f:
        params_norm = json.loads(f.read())
    for i, feature in enumerate(features):
        mean = float(params_norm[feature]["mean"])
        std = float(params_norm[feature]["std"])
        data[:, i] = (data[:, i] - mean) / std
    return data


def load_combined_hemisphere_data(
    subj,
    features,
    features_to_ignore=[],
    universal_features=[],
    demographic_features=[],
    num_neighbours=0,
    normalise=False,
):
    """
    Combine features from both hemispheres into single matrix and
    mask cortex and non-lesional data if necessary

    Args:
        subj:
        features: list of features to be loaded
        features_to_ignore: list of features that should be replaced with 0 upon loading
        universal_features: non-patient specific features that should also be appended to features.
            currently, only 'coords' are implemented
        demographic_features: demographic features that should be appended to features
        num_neighbours: number of neighbours to load
        neighbours (np.array, optional): list of neighbours for each vertex.
        Neighbour information is appended to demographic_features:
            list of feature values to be repeated for each vertex (e.g. values for age of onset)
    """
    cortex_mask = subj.cohort.cortex_mask
    combined_data = []
    combined_label = []
    for hemi in ["lh", "rh"]:
        # get mri features
        hemi_data, hemi_label = subj.load_feature_lesion_data(
            features, hemi=hemi, features_to_ignore=features_to_ignore
        )
        # get universal features
        if universal_features is not None:
            for universal_feature in universal_features:
                if universal_feature == "coords":
                    hemi_data = np.hstack([hemi_data, subj.cohort.coords])
                else:
                    raise NotImplementedError(universal_feature)
        # get demographic features
        if demographic_features is not None and len(demographic_features) > 0:
            demographic_data = subj.get_demographic_features(demographic_features, default="random", normalize=True)
            # broadcast to hemi_data
            n_vert = hemi_data.shape[0]
            demographic_data = np.broadcast_to(
                np.array(demographic_data)[np.newaxis, :], (n_vert, len(demographic_data))
            )
            # add noise
            noise = np.random.normal(0, 0.1, size=demographic_data.shape)
            hemi_data = np.hstack([hemi_data, demographic_data + noise])
        # add neighbours
        if num_neighbours > 0:
            selected_neighbours = np.array([arr[:num_neighbours] for arr in subj.cohort.neighbours])
            hemi_neigh = hemi_data[selected_neighbours].reshape((len(hemi_data), -1))
            hemi_data = np.concatenate([hemi_data, hemi_neigh], axis=-1)
        # mask out non-cortical data
        hemi_data = hemi_data[cortex_mask]
        hemi_label = hemi_label[cortex_mask]

        combined_data.append(hemi_data)
        combined_label.append(hemi_label)
    combined_data = np.vstack(combined_data)
    combined_label = np.hstack(combined_label).astype(int)
    # normalise
    if normalise != False:
        combined_data = normalise_data(combined_data, features, normalise)
    return combined_data, combined_label


class Dataset(Sequence):
    """Dataset for training classifiers.

    This is an iterable returning batched (features, labels),
    i.e. you can iterate over a Dataset instance with:
    ```
    dataset = Dataset.from_experimet(experiment)
    for features, labels in dataset:
        print(labels)
    ```

    Attributes:
        subject_ids (list of str): subjects to use in this dataset
        cohort (MeldCohort): defines hdf5 file location to read data
        features (list or str): features to load for each subject
        is_val_dataset (optional, bool): switches of active selection and shuffling per epoch
            for a val dataset,default is False
        model (optional, Model instance): neural network model that should be used for active selection of vertices. Is not required if no active selection is desired.
        **kwargs: data parameters determinning sampling of vertices and additional features that should
            be passed to the model. Used parameters are:

            batch_size (int): number of samples per batch, default is 1024
            contra (bool): where to sample normal vertices from, default is True
                If True, only sample normal vertices from contralateral hemisphere
                If False, sample normal vertices from both hemispheres
            boundary (bool): exclude normal vertices that are close to lesional vertices,
                default is False.
                Creates an uncertainty boundary around lesions (due to low-quality annotations).
                Only has an effect is contra is False.
            num_per_subject (int): number of vertices to randomly select per subject, default is None
                If None, all vertices are selected (respecting `equalize` flag).
            equalize (bool): sample the same amount of lesional/normal vertices for each subject, default is True
                If True, subsample normal vertices to have equal numbers of lesional and non-lesional examples.
                If `num_per_subject` is defined, will select num_per_subject lesional and num_per_subject non-lesional
                vertices (total 2*num_per_subject), oversampling lesional vertices if necessary.
            equalize_factor (float): determines the factor between lesional and normal vertices, default is 1.
                A factor of 1 results in a balanced dataset.
                A factor of 2 results in a dataset with twice as many normal vertices than lesional
                vertices.
            active_selection (bool): dynamic selection of normal vertices according to how well the model performs
                (select more vertices that are harder to classify). Default is False.
            active_selection_pool_factor (float): how many more normal vertices should be loaded
                (to select training vertices from)? Default is 2
            active_selection_frac (float): fraction of vertices that should be selected according to the models performance.
                The remainder is selected randomly.
                active_selection_frac=0 means random resampling of vertices from the pool of vertices.
                Default is 0.5
            resample_each_epoch (bool): regenerate training set each epoch (to randomly select other normal samples). Default is False
            shuffle_each_epoch (bool): shuffle the dataset each epoch. Default is False
            num_neighbours (int): determines the number of neighbouring vertices that are added as features to the data for each vertex. Default is 0
            universal_features (list): features applied to all subjects eg location of vertex. Default is empty list.
            demographic_features (list): demographic features loaded from csv. Valid values are column names of the demographics csv. Default is empty list
            features_to_replace_with_0 (list): features that should be replaced with 0 for training.
                Default is empty list.
    """

    def __init__(self, subject_ids, cohort, features=[], is_val_dataset=False, model=None, **kwargs):
        """
        For general documentation, see Dataset.
        Detailed behavior of equalize and num_per_subject flags:
            `equalize==True and num_per_subject==None`: select num_lesional lesional + normal vertices from each brain
                (500 from controls)
            `equalize==True and num_per_subject!=None`: select num_per_subject lesional + normal vertices
                (total 2*num_per_subject) from each brain (num_per_subject for controls)
            `equalize==None and num_per_subject==None`: select all vertices
            `equalize==None and num_per_subject!=None`: randomly select num_per_subject vertices
        """
        self.log = logging.getLogger(__name__)
        # set flags
        self.subject_ids = subject_ids
        self.cohort = cohort
        self.features = features
        self.model = model
        # set all used kwargs
        self.features_to_replace_with_0 = kwargs.get("features_to_replace_with_0", [])
        self.universal_features = kwargs.get("universal_features", [])
        self.demographic_features = kwargs.get("demographic_features", [])
        self.equalize = kwargs.get("equalize", True)
        self.equalize_factor = kwargs.get("equalize_factor", 1)
        self.num_per_subject = kwargs.get("num_per_subject", None)
        self.contra = kwargs.get("contra", True)
        self.boundary = kwargs.get("boundary", False)
        self.batch_size = kwargs.get("batch_size", 1024)
        self.resample_each_epoch = kwargs.get("resample_each_epoch", False)
        self.shuffle_each_epoch = kwargs.get("shuffle_each_epoch", False)
        self.active_selection = kwargs.get("active_selection", False)
        self.active_selection_pool_factor = kwargs.get("active_selection_pool_factor", 2)
        self.active_selection_frac = kwargs.get("active_selection_frac", 0.5)
        self.num_neighbours = kwargs.get("num_neighbours", 0)
        self.normalise = kwargs.get("normalise", False)

        # switch options off that do not make sense for validation data
        if is_val_dataset:
            self.active_selection = False
            self.shuffle_each_epoch = False

        # store features + labels per subject, ordered by lesion + normal
        # structure is: data[subj_id] = {'lesion':features, 'normal':features}
        self.data_dict = self.load_and_sample_data(seed=0)
        self.data_list = self.prepare_data_list(self.data_dict, do_random_selection=True)
        if self.shuffle_each_epoch:
            self.shuffle_data_list()

    @classmethod
    def from_experiment(cls, experiment, mode="train"):
        """
        initialise dataset from Experiment instance.

        Args:
            experiment: Experiment instance to use for getting all relevant parameters
            mode: 'train', 'val', 'test'; determines the subject_ids that are used for this dataset.
                if mode is val or test, the flag is_val_dataset will be set to true.
        """
        is_val_dataset = mode != "train"
        # set subject_ids
        train_ids, val_ids, test_ids = experiment.get_train_val_test_ids()
        if mode == "train":
            subject_ids = train_ids
        elif mode == "val":
            subject_ids = val_ids
        elif mode == "test":
            subject_ids = test_ids
        else:
            raise NotImplementedError(mode)
        # ensure that features are saved in experiment.data_parameters
        experiment.get_features()

        return cls(
            subject_ids,
            experiment.cohort,
            is_val_dataset=is_val_dataset,
            model=experiment.model,
            **experiment.data_parameters,
        )

    def get_num_vertices_to_sample(self, subj_id, total_lesional, total_normal):
        """Calculate number of lesional and normal vertices to sample according to the set up of the Dataset

        Takes into account the equalize, equalize_factor, and num_per_subject flags

        Args:
            subj_id (str): subject
            total_lesional (int): total number of lesional vertices for this subject
            total_normal (int): total number of normal vertices for this subject
        Returns:
            tuple:
                num_lesional: number of lesional vertices to sample
                num_normal: number of normal vertices to sample
        """
        subj = MeldSubject(subj_id, self.cohort)
        # control or patient
        if not subj.is_patient:
            num_lesional = 0
            if self.num_per_subject is None:
                if self.equalize:
                    # take 500 vertices because there's no lesional vertices to equalize against.
                    num_normal = 500 * self.equalize_factor
                else:
                    # take all vertices
                    num_normal = total_normal
            else:
                # take num_per_subject vertices
                num_normal = self.num_per_subject
        else:
            if self.num_per_subject is not None:
                if self.equalize:
                    num_normal = self.num_per_subject * self.equalize_factor
                    num_lesional = self.num_per_subject
                else:
                    raise NotImplementedError
                    # would need to set num_normal and num_lesional to fraction of total_lesional and total_normal
                    # and ensure that together they sum to num_per_subject
                    # but we probably do not want to use this option anyways
            else:
                if self.equalize:
                    # select same number of vertices for normal and lesional
                    num_normal = total_lesional * self.equalize_factor
                    num_lesional = total_lesional
                else:
                    # select all vertices
                    num_normal = total_normal
                    num_lesional = total_lesional
        # account for active selection: can sample more normal vertices from disk that are used to choose training data points
        # e.g. train on badly classified normal vertices
        if self.active_selection:
            num_normal = self.active_selection_pool_factor * num_normal
        return num_lesional, num_normal

    def load_and_sample_data(self, seed=False):
        """
        Load features from hdf5 files and subsample.

        Uses get_num_vertices_to_sample to get the number of lesional and normal vertices to sample
        Optionally (self.boundary) exclude the boundary zone between lesional and normal vertices.

        Returns:
            data_dict: data dictionary containing features from each patient sorted by "lesion" and "normal"
        """
        if seed:
            np.random.seed(seed)
        # load subject_data
        data_dict = {}

        for subj_id in self.subject_ids:
            self.log.debug(f"loading data for {subj_id}")
            if self.normalise != False:
                self.log.debug(f"normalise data with parameters from {self.normalise}")
            self.log.debug(f"normalise data for {subj_id}")
            data_dict[subj_id] = {"lesion": [], "normal": []}
            subj = MeldSubject(subj_id, self.cohort)
            features, labels = load_combined_hemisphere_data(
                subj,
                self.features,
                features_to_ignore=self.features_to_replace_with_0,
                universal_features=self.universal_features,
                demographic_features=self.demographic_features,
                num_neighbours=self.num_neighbours,
                normalise=self.normalise,
            )

            # lesional vertices have label 1
            lesional_ids = np.where(labels == 1)[0]
            total_lesional = len(lesional_ids)
            # normal vertices have label 0 and depends on contra and boundary flags (exclude more vertices)
            mask_normal = labels == 0
            if self.boundary and subj.is_patient:
                # exclude boundary zone
                boundary_label = subj.load_boundary_zone()
                self.log.debug(
                    "{}, {}, {}".format(
                        subj_id,
                        sum(boundary_label[: int(len(boundary_label) / 2)]),
                        sum(boundary_label[int(len(boundary_label) / 2) :]),
                    )
                )
                self.log.debug(
                    "{}, {}, {}".format(
                        subj_id,
                        sum(labels[: int(len(boundary_label) / 2)]),
                        sum(labels[int(len(boundary_label) / 2) :]),
                    )
                )

                mask_normal = boundary_label == 0
            if self.contra:
                # exclude vertices on lesional hemishpere
                les_hemi = subj.get_lesion_hemisphere()
                if les_hemi == "lh":
                    mask_normal = np.arange(len(features)) >= len(features) // 2
                else:
                    mask_normal = np.arange(len(features)) < len(features) // 2
            normal_ids = np.where(mask_normal)[0]
            total_normal = len(normal_ids)
            # get number of vertices to sample from normal and lesional tissue
            num_lesional, num_normal = self.get_num_vertices_to_sample(subj_id, total_lesional, total_normal)
            self.log.debug(
                "selecting {} lesional (from {}) and {} normal vertices (from {}) for {}".format(
                    num_lesional, total_lesional, num_normal, total_normal, subj_id
                )
            )
            # select lesional data
            if num_lesional == 0:
                data_dict[subj_id]["lesion"] = np.zeros((0, len(self.features)))
            elif num_lesional == total_lesional:
                data_dict[subj_id]["lesion"] = features[lesional_ids]
            elif num_lesional < total_lesional:
                # randomly select lesional data
                selection = np.random.choice(lesional_ids, size=num_lesional, replace=False)
                data_dict[subj_id]["lesion"] = features[selection]
                assert (labels[selection] != 1).sum() == 0
            elif num_lesional > total_lesional:
                # oversample lesional data for this subject
                # randomly select features after sampling entire lesion
                selection = np.concatenate(
                    [lesional_ids, np.random.choice(lesional_ids, size=num_lesional - total_lesional, replace=True)]
                )
                data_dict[subj_id]["lesion"] = features[selection]
                assert (labels[selection] != 1).sum() == 0

            # select normal (non-lesional) data
            if num_normal == 0:
                data_dict[subj_id]["normal"] = np.zeros((0, len(self.features)))
            elif num_normal == total_normal:
                data_dict[subj_id]["normal"] = features[normal_ids]
            elif num_normal < total_normal:
                selection = np.random.choice(normal_ids, size=num_normal, replace=False)
                data_dict[subj_id]["normal"] = features[selection]
                assert labels[selection].sum() == 0
            elif num_normal > total_normal:
                # oversample normal data for this subject
                # randomly select features after sampling all normal vertices
                selection = np.concatenate(
                    [normal_ids, np.random.choice(normal_ids, size=num_normal - total_normal, replace=True)]
                )
                data_dict[subj_id]["normal"] = features[selection]
                assert labels[selection].sum() == 0
            # if active selection is turned on, create tf Dataset from normal vertices to use in active_selection_of_normal_vertices
            # this speeds up the active selection step at the end of each epoch
            if self.active_selection:
                data_dict[subj_id]["normal_tfDataset"] = tf.data.Dataset.from_tensor_slices(
                    [data_dict[subj_id]["normal"]]
                )
        return data_dict

    def active_selection_of_normal_vertices(self, data_dict, do_random_selection=False):
        """select normal vertices to train on and save selection in data_dict"""
        if not do_random_selection:
            assert (
                self.model is not None
            ), "need a tf.keras.Model in self.model in order to do active selection. Please set this before calling this function"
        for subj_id in data_dict.keys():
            num_vertices = int(len(self.data_dict[subj_id]["normal"]) / self.active_selection_pool_factor)
            num_active_selection = int(num_vertices * self.active_selection_frac)
            if do_random_selection:
                num_active_selection = 0
            # select the num_active_selection worst performing vertices
            if num_active_selection > 0:
                pred = self.model.predict(self.data_dict[subj_id]["normal_tfDataset"])
                selected_ids_active = np.argsort(pred[:, 0])[-num_active_selection:]
            else:
                selected_ids_active = []
            # select the remaining vertices randomly
            ids = np.arange(0, len(self.data_dict[subj_id]["normal"]))
            selected_ids_random = np.random.choice(
                ids[~np.in1d(ids, selected_ids_active)], num_vertices - num_active_selection, replace=False
            )
            selected_ids = np.concatenate([selected_ids_active, selected_ids_random], axis=0).astype(int)
            data_dict[subj_id]["normal_selected_ids"] = selected_ids
        return data_dict

    def prepare_data_list(self, data_dict, do_random_selection=False):
        """combine data in 1d lists for training"""
        if self.active_selection:
            data_dict = self.active_selection_of_normal_vertices(data_dict, do_random_selection)
        features = []
        labels = []
        for subj_id in data_dict.keys():
            # lesional data
            if MeldSubject(subj_id, self.cohort).is_patient:
                features.append(data_dict[subj_id]["lesion"])
                labels.append(np.repeat(1, len(features[-1])))
            # normal data
            # if using active_selection, select normal vertices using information from data_dict
            # (written with active_selection_of_normal_vertices)
            if self.active_selection:
                features.append(data_dict[subj_id]["normal"][data_dict[subj_id]["normal_selected_ids"]])
            else:
                features.append(data_dict[subj_id]["normal"])
            labels.append(np.repeat(0, len(features[-1])))
        features = np.concatenate(features, axis=0).astype(np.float32)
        labels = np.concatenate(labels, axis=0).astype(np.uint8)
        return (features, labels)

    def shuffle_data_list(self):
        features, labels = self.data_list
        ids = np.arange(len(features))
        np.random.shuffle(ids)
        features = features[ids]
        labels = labels[ids]
        self.data_list = (features, labels)

    def __getitem__(self, idx):
        batch_features = np.array(self.data_list[0][idx * self.batch_size : (idx + 1) * self.batch_size])
        batch_labels = np.array(self.data_list[1][idx * self.batch_size : (idx + 1) * self.batch_size])
        return batch_features, batch_labels

    def on_epoch_end(self):
        """after each epoch, reload and shuffle the data"""
        if self.resample_each_epoch:
            self.data_dict = self.load_and_sample_data()
        if self.resample_each_epoch or self.active_selection:
            self.data_list = self.prepare_data_list(self.data_dict)
        if self.shuffle_each_epoch:
            self.shuffle_data_list()

    def __len__(self):
        return math.ceil(len(self.data_list[0]) / self.batch_size)
