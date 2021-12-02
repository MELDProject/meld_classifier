# TODO this script was adapted to the new data classes, but is not yet tested.
# If you succesfully un it, remove this comment
from meld_classifier.evaluation import Evaluator
import argparse
import numpy as np
import meld_classifier.paths as paths
import json
from meld_classifier.experiment import get_subject_ids
import os

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Evaluate trained experiment on subject list")
    parser.add_argument("--subject-id", default="subs.csv", help="list of subjects on which to evaluate")
    parser.add_argument(
        "--experiment-folder", default="focal_loss_gamma_20-08-28", help="folder for experiment to evaluate"
    )
    parser.add_argument("--experiment-name", default="focal_loss_gamma_1", help="name of specific experiment model")
    parser.add_argument(
        "--output", default=paths.EXPERIMENT_PATH, help="location where the resulting images should be saved"
    )
    parser.add_argument("--folds", nargs="+", default=range(10), help="folds models to use. default is average of 10")
    parser.add_argument("--exp-path", default=paths.EXPERIMENT_PATH)
    parser.add_argument("--threshold", default="optimal")
    args = parser.parse_args()

    subject_ids = np.loadtxt(args.subject_id, dtype=str)
    # set up evaluation class from experiment and subjects
    # start with fold_0 for initialisation
    data_dictionary = {}
    fields_of_interest = ["result", "input_labels"]
    for f in args.folds:
        experiment_path = os.path.join(args.exp_path, args.experiment_folder, "fold_{}".format(f))
        data_parameters = json.load(
            open(os.path.join(experiment_path, "data_parameters_{}.json".format(args.experiment_name)))
        )
        network_parameters = json.load(
            open(os.path.join(experiment_path, "network_parameters_{}.json".format(args.experiment_name)))
        )

        experiment = Experiment(experiment_path=experiment_path, experiment_name=args.experiment_name)
        evaluator = Evaluator(
            experiment, mode="inference", make_images=False, make_prediction_space=False, subject_ids=subject_ids
        )
        evaluator.load_predict_data()
        # store predictions
        for subject in evaluator.data_dictionary.keys():
            if f == 0:
                data_dictionary[subject] = {}
                for field in fields_of_interest:
                    data_dictionary[subject][field] = evaluator.data_dictionary[subject][field] / len(args.folds)
            else:
                for field in fields_of_interest:
                    # averaging predictions
                    data_dictionary[subject][field] += evaluator.data_dictionary[subject][field] / len(args.folds)
    # replace final dictionary with mean and plot
    evaluator.data_dictionary = data_dictionary
    if args.threshold != "optimal":
        evaluator.threshold = float(args.threshold)
    evaluator.threshold_and_cluster()
    evaluator.plot_subjects_prediction(rootfile=os.path.join(args.output, "inference_{}.png"))
# predict and plot
# TODO is not efficient because we keep loading data and model over and over.
