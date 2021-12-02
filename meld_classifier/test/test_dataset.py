#### tests for dataset.py (Dataset class) ####
# tested functions:
#   load_combined_hemisphere_data
#   Dataset - behaviour with different flags, active selection
# NOTE:
#   these tests require a test dataset, that is created with get_test_data()
#   executing this function may take a while the first time (while the test data is being created)
# MISSING TESTS:
#   Dataset - test asserting correct handling of boundary zones in Dataset

from meld_classifier.dataset import load_combined_hemisphere_data, Dataset
from meld_classifier.download_data import get_test_data
from meld_classifier.meld_cohort import MeldSubject, MeldCohort
import pytest
from meld_classifier.paths import NVERT, DEFAULT_HDF5_FILE_ROOT
from meld_classifier.network_tools import build_model
import numpy as np
from copy import deepcopy


@pytest.fixture(autouse=True)
def setup_teardown_tests():
    get_test_data()
    yield


@pytest.fixture(scope="session")
def data_parameters():
    data_parameters = {
        "site_codes": ["TEST"],
        "scanners": ["15T", "3T"],
        "group": "patient",
        "features_to_exclude": [],
        "subject_features_to_exclude": ["FLAIR"],
        "fold_n": 0,
        "number_of_folds": 10,
        "hdf5_file_root": DEFAULT_HDF5_FILE_ROOT,
        # --- params for Dataset class creation ---
        "batch_size": 1024,
        "contra": False,
        "boundary": False,
        "num_per_subject": None,
        "equalize": True,
        "equalize_factor": 1,
        "resample_each_epoch": False,
        "shuffle_each_epoch": True,
        "active_selection": True,
        "active_selection_pool_factor": 5,
        "active_selection_frac": 0.5,
        "num_neighbours": 0,
    }
    return data_parameters


def test_load_combined_hemisphere_data():
    c = MeldCohort(hdf5_file_root=DEFAULT_HDF5_FILE_ROOT)
    subj = MeldSubject("MELD_TEST_15T_FCD_0002", cohort=c)
    features, labels = load_combined_hemisphere_data(subj, [".inter_z.asym.intra_z.combat.on_lh.gm_FLAIR_0.25.sm10.mgh"])
    features = features.astype(int)

    # check that features are ordered list of indices (only the case for TEST dataset)
    assert ((features[1 : len(features) // 2] - features[: len(features) // 2 - 1]) > 0).all()
    # features should be cortex indices stacked (once for lh, once for rh)
    assert (features[: len(features) // 2, 0] == c.cortex_label).all()


# Dataset class tests
def test_dataset_flags(data_parameters):
    c = MeldCohort(hdf5_file_root=data_parameters["hdf5_file_root"])

    subject_ids = c.get_subject_ids(**data_parameters)
    features_list = c.get_features(features_to_exclude=data_parameters["features_to_exclude"])

    # test contra flag
    # make sure that selected non-lesional data comes from contra hemisphere
    cur_data_params = dict(data_parameters, contra=True)
    dataset = Dataset([subject_ids[0]], cohort=c, features=features_list, **cur_data_params, is_val_dataset=False)

    all_features, all_labels = load_combined_hemisphere_data(MeldSubject(subject_ids[0], cohort=c), features_list)
    # check that features are correctly labelled
    features, labels = dataset.data_list
    assert (np.in1d(features[labels == 1], all_features[all_labels == 1])).all()

    # that non-lesional labels are all coming from the same hemisphere (only have one subject here)
    assert not (features[labels == 0] > NVERT).any() or (features[labels == 0] > NVERT).all()

    # test equalize flag
    labels = dataset.data_list[1]
    assert sum(labels == 0) == sum(labels == 1)

    # test equalize_factor flag
    equalize_factor = 2
    cur_data_params = dict(data_parameters, equalize_factor=equalize_factor, equalize=True)
    dataset = Dataset([subject_ids[0]], cohort=c, features=features_list, **cur_data_params, is_val_dataset=False)
    labels = dataset.data_list[1]
    assert sum(labels == 0) == sum(labels == 1) * equalize_factor

    # test num_per_subject_flag
    cur_data_params = dict(data_parameters, num_per_subject=100, equalize=True)
    dataset = Dataset([subject_ids[0]], cohort=c, features=features_list, **cur_data_params, is_val_dataset=False)
    assert len(dataset.data_list[0]) == 200
    labels = dataset.data_list[1]
    assert sum(labels) == 100


def test_dataset_active_selection(data_parameters):
    c = MeldCohort(hdf5_file_root=data_parameters["hdf5_file_root"])

    subject_ids = c.get_subject_ids(**data_parameters)
    features_list = c.get_features(features_to_exclude=data_parameters["features_to_exclude"])

    # test active selection
    cur_data_params = dict(data_parameters, active_selection=True)
    dataset = Dataset([subject_ids[0]], cohort=c, features=features_list, **cur_data_params, is_val_dataset=False)

    # need to create model, because active selection
    model = build_model(
        n_features=len(features_list),
        learning_rate=0.01,
        layer_sizes=[20],
        dropout=0.2,
    )
    dataset.model = model

    features1, labels1 = deepcopy(dataset.data_list)
    dataset.on_epoch_end()
    features2, labels2 = dataset.data_list
    # make sure that data is resampled / changed
    assert len(np.setdiff1d(features1, features2)) < len(features1)


def test_dataset_neighbours(data_parameters):
    import meld_classifier.mesh_tools as mt

    c = MeldCohort(hdf5_file_root=data_parameters["hdf5_file_root"])

    subject_ids = c.get_subject_ids(**data_parameters)
    features_list = c.get_features(features_to_exclude=data_parameters["features_to_exclude"])

    # test creating dataset with multiple neighbours
    np.random.seed(0)
    ds0 = Dataset([subject_ids[0]], cohort=c, features=features_list, **data_parameters, is_val_dataset=False)
    cur_data_params = dict(data_parameters, num_neighbours=5)
    np.random.seed(0)
    ds5 = Dataset([subject_ids[0]], cohort=c, features=features_list, **cur_data_params, is_val_dataset=False)

    # check that datasets return correct shape
    num_features = ds0.data_list[0].shape[1]
    assert num_features * 6 == ds5.data_list[0].shape[1]
    # first features should be the same
    assert (ds0.data_list[0] == ds5.data_list[0][:, :num_features]).all()
    # labels should be the same
    assert (ds0.data_list[1] == ds5.data_list[1]).all()
    # are neighbours correct? (Uses the fact that the TEST data features are ascending numbers)
    vert = (ds5.data_list[0][:, 0] % NVERT).astype(int)
    neighs = (ds5.data_list[0][:, range(num_features, num_features * 6, num_features)] % NVERT).astype(int)
    # compare with neighbours from file
    neighbours = c.neighbours
    selected_neighbours = np.array([neigh[:5] for neigh in neighbours])

    assert (selected_neighbours[vert] == neighs).all()

    # check that reshaping (for use in nn model) works as expected: num_batches, num_verts, num_features
    reshaped = ds5.data_list[0].reshape(-1, 6, num_features)
    assert (reshaped[:, :, 0] == reshaped[:, :, 1]).all()
