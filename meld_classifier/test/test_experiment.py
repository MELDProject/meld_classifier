#### tests for experiment.py ####
# this is a long test, if want to exclude it for quick testing, run
# pytest -m "not slow"
# tested functions:
#   _train_val_test_split_folds - test reproducibility of splitting in folds
#   train_network - is running / creates expected outputs / can evaluate -> part of these tests are in test_script_run

from meld_classifier.experiment import Experiment
from meld_classifier.meld_cohort import MeldCohort
import os
import pytest
import datetime
from meld_classifier.paths import DEFAULT_HDF5_FILE_ROOT


@pytest.fixture(scope="session")
def experiment(tmpdir_factory):
    data_parameters = {
        "site_codes": ["TEST"],
        "scanners": ["15T", "3T"],
        "hdf5_file_root": DEFAULT_HDF5_FILE_ROOT,
        "dataset": "MELD_dataset_TEST.csv",
        "group": "both",
        "features_to_exclude": [],
        "subject_features_to_exclude": [""],
        "min_area_threshold": 50,
        "number_of_folds": 10,
        "fold_n": [0],
        "iteration": 0,
        "batch_size": 1024,
        "contra": False,
        "num_per_subject": None,
        "equalize": True,
        "equalize_factor": 1,
        "active_selection": True,
        "active_selection_pool_factor": 5,
        "active_selection_frac": 0.5,
        "resample_each_epoch": False,
        "shuffle_each_epoch": True,
        "universal_features": "",
        "num_neighbours": 0,
    }

    network_parameters = {
        ##### network architecture #####
        "layer_sizes": [20],
        "dropout": 0.2,
        ##### training hyper-params #####
        "learning_rate": 0.0001,
        "max_patience": 10,
        "num_epochs": 1,
        "loss": "binary_crossentropy",
        "weighting": None,
        "optimal_threshold": 0.4,
        "date": datetime.datetime.now().strftime("%y-%m-%d"),
    }

    experiment_path = str(tmpdir_factory.mktemp("experiment"))
    # experiment_path = f'test_results/iteration_{network_parameters["date"]}'
    experiment = Experiment.create_with_parameters(data_parameters, network_parameters, experiment_path, "iteration_0")
    experiment.init_logging()
    return experiment


def test_train_val_test_split_folds(experiment):
    c = MeldCohort(hdf5_file_root=DEFAULT_HDF5_FILE_ROOT)
    subject_ids = c.get_subject_ids(site_codes=['TEST'])

    # get all possible splits
    train_ids = [[] for _ in range(5)]
    val_ids = [[] for _ in range(5)]
    test_ids = [[] for _ in range(5)]
    for i in range(5):
        train_ids[i], val_ids[i], test_ids[i] = experiment._train_val_test_split_folds(
            subject_ids, iteration=i, number_of_folds=5
        )

    # test ids should be the same for each fold
    for i in range(5):
        assert (test_ids[i] == test_ids[0]).all()
    # val ids should always be different
    for i in range(5):
        for j in range(5):
            if i == j:
                continue
            assert val_ids[i].shape != val_ids[j].shape or (val_ids[i] != val_ids[j]).all()

    # get iteration 0 fold again
    train_ids0, val_ids0, test_ids0 = experiment._train_val_test_split_folds(
        subject_ids, iteration=0, number_of_folds=5
    )
    # train ids should be identical to train ids before
    assert (train_ids0 == train_ids[0]).all()


@pytest.mark.slow
def test_run_experiment(experiment):
    # train experiment
    experiment.train()

    # check that checkpoint files exists
    checkpoint_file = os.path.join(experiment.path, "models", f"{experiment.name}.index")
    assert os.path.isfile(checkpoint_file)
    # check that log file exists
    log_file = os.path.join(experiment.path, "logs", f"{experiment.name}.csv")
    assert os.path.isfile(log_file)

    # optimise threshold
    experiment.optimise_threshold()

    # check that created sens_sepc_curve
    sens_spec_curve_file = os.path.join(
        experiment.path, "results", "images", f"sensitivity_specificity_curve_{experiment.name}_2.png"
    )
    assert os.path.isfile(sens_spec_curve_file)

    # evaluate
    experiment.evaluate(make_images_flag=True, make_prediction_space_flag=True)

    # check that have created prediction space plot
    prediction_space_file = os.path.join(
        experiment.path, "results", "images", f"prediction_space_{experiment.name}.png"
    )
    assert os.path.isfile(prediction_space_file)
    # check that have saved test results
    assert os.path.isfile(os.path.join(experiment.path, "results", f"test_results_{experiment.name}.csv"))
    assert os.path.isfile(os.path.join(experiment.path, "results", f"per_subject_{experiment.name}_optimal.json"))
    assert os.path.isfile(os.path.join(experiment.path, "results", f"per_subject_{experiment.name}_0.5.json"))
    # check that have saved patient predictions
    _, val_ids, _ = experiment.get_train_val_test_ids()
    for val_id in val_ids:
        print(val_id)
        assert os.path.isfile(
            os.path.join(experiment.path, "results", "images", f"{experiment.name}_{val_id}.jpg")
        )
