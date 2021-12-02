# TODO this script was adapted to the new data classes, but is not yet tested.
# If you succesfully un it, remove this comment
from meld_classifier.experiment import Experiment
from meld_classifier.paths import EXPERIMENT_PATH
from meld_classifier.network_tools import ensemble_models
from meld_classifier.evaluation import Evaluator
import argparse
import numpy as np
import os
from glob import glob
import datetime
import logging
import shutil
import json
import tensorflow as tf


def _update_subj_ids(data_param_file, ensemble_experiments):
    train_ids = []
    for exp in ensemble_experiments:
        params = json.load(open(os.path.join(exp[0], f"data_parameters_{exp[1]}.json"), "r"))
        train_ids.extend(params["train_ids"])
    train_ids = list(np.unique(train_ids))

    params = json.load(open(data_param_file, "r"))
    params["train_ids"] = train_ids
    params["val_ids"] = []
    json.dump(params, open(data_param_file, "w"), indent=4)


def create_ensemble(experiment_name, experiment_path, ensemble_experiments, ensemble_folds=False):
    """
    Args:
        ensemble_folds
            if true, assume that ensemble is across all folds - this means that train ids need to be set to the union of
            all train ids, and val ids are empty

    """
    # create new experiment_path and copy over data_parameters and network_parameters
    if not os.path.exists(experiment_path):
        os.makedirs(experiment_path)

    # use parameters of first experiments for this
    data_param_file = os.path.join(
        ensemble_experiments[0][0], "data_parameters_{}.json".format(ensemble_experiments[0][1])
    )
    shutil.copyfile(data_param_file, os.path.join(experiment_path, "data_parameters_{}.json".format(experiment_name)))
    network_param_file = os.path.join(
        ensemble_experiments[0][0], "network_parameters_{}.json".format(ensemble_experiments[0][1])
    )
    shutil.copyfile(
        network_param_file, os.path.join(experiment_path, "network_parameters_{}.json".format(experiment_name))
    )
    if ensemble_folds:
        # merge all train ids
        _update_subj_ids(data_param_file, ensemble_experiments)

    # create results dir
    if not os.path.exists(os.path.join(experiment_path, "results", "images")):
        os.makedirs(os.path.join(experiment_path, "results", "images"))

    models = []
    for exp in ensemble_experiments:
        experiment = Experiment(experiment_path=exp[0], experiment_name=exp[1])
        eva = Evaluator(experiment, mode="val")
        models.append(eva.experiment.model)

    model_input = tf.keras.layers.Input(shape=models[0].input_shape[1:])
    ensemble = ensemble_models(models, model_input)
    ensemble.save(os.path.join(experiment_path, "models", experiment_name))


def evaluate_ensemble(experiment_name, experiment_path):
    # create experiment instance and evaluate
    exp = Experiment(experiment_path=experiment_path, experiment_name=experiment_name)
    # init logging
    exp.init_logging(console_level=logging.INFO)
    exp.optimise_threshold()
    # exp.evaluate() # TODO: could expose make_images_flag here


if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description="Ensemble trained experiments. Creates a new ensemble experiment that can be used with experiment_evaluation.py"
    )
    parser.add_argument(
        "--exp",
        nargs="+",
        action="append",
        metavar=("experiment_folder", "experiment_params"),
        help="Experiments in one folder to ensemble. Can be specified multiple times",
    )
    parser.add_argument(
        "--ensemble-experiment-path", default=None, help="experiment path of the resulting ensemble experiment"
    )
    parser.add_argument("--ensemble-exp-name", default="0", help="name of the resulting ensemble experiment")
    parser.add_argument("--folds", nargs="+", default=range(10))
    parser.add_argument(
        "--ensemble-folds",
        action="store_true",
        default=False,
        help="create one ensemble model over all folds (cannot be used to predict on val, because has no val set)",
    )
    args = parser.parse_args()

    # experiments to be compared
    experiments_dictionary = {exp[0]: exp[1:] for exp in args.exp}
    # add all experiments for experiment_folder if none are specified
    for exp_folder, exp_param in experiments_dictionary.items():
        if len(exp_param) == 0:
            exp_name = os.path.basename(exp_folder)[:-9]  # remove the date from foldername
            files = sorted(
                glob(
                    os.path.join(
                        EXPERIMENT_PATH,
                        exp_folder,
                        "fold_{}".format(args.folds[0]),
                        "data_parameters_{}_*.json".format(exp_name),
                    )
                )
            )
            for f in files:
                exp_param.append(os.path.splitext(os.path.basename(f))[0].split("_")[-1])
            experiments_dictionary[exp_folder] = exp_param

    output_dir = args.ensemble_experiment_path
    if output_dir is None:
        output_dir = os.path.join(list(experiments_dictionary.keys())[0])

    # create and evaluate ensemble
    ensemble_experiments = []
    experiment_name = "ensemble_{}".format(args.ensemble_exp_name)
    for fold in args.folds:
        for experiment_path, exp_params in experiments_dictionary.items():
            exp_name = os.path.basename(experiment_path)[:-9]
            ensemble_experiments.extend(
                [
                    [
                        os.path.join(EXPERIMENT_PATH, experiment_path, "fold_{}".format(fold)),
                        "{}_{}".format(exp_name, exp_param),
                    ]
                    for exp_param in exp_params
                ]
            )

        if not args.ensemble_folds:
            # create one experiment per fold
            experiment_dir = os.path.join(
                EXPERIMENT_PATH,
                output_dir,
                "ensemble_{}".format(datetime.datetime.now().strftime("%y-%m-%d")),
                "fold_{}".format(fold),
            )
            print("Evaluating ensemble of {}".format(ensemble_experiments))
            print("Saving to {} with name {}".format(experiment_dir, experiment_name))
            create_ensemble(experiment_name, experiment_dir, ensemble_experiments, ensemble_folds=args.ensemble_folds)
            # these ensembles can be evaluated on their val set
            evaluate_ensemble(experiment_name, experiment_dir)

            ensemble_experiments = []

    if args.ensemble_folds:
        # create one experiment overall
        experiment_dir = os.path.join(
            EXPERIMENT_PATH, output_dir, "ensemble_{}".format(datetime.datetime.now().strftime("%y-%m-%d")), "fold_all"
        )
        print("Evaluating ensemble of {}".format(ensemble_experiments))
        print("Saving to {} with name {}".format(experiment_dir, experiment_name))
        create_ensemble(experiment_name, experiment_dir, ensemble_experiments, ensemble_folds=args.ensemble_folds)
        evaluate_ensemble(experiment_name, experiment_dir)
