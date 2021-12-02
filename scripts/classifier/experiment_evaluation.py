from meld_classifier.experiment_comparison import ExperimentComparison
import meld_classifier.paths as paths
import argparse
import numpy as np
import os
from glob import glob

if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description="Compare trained experiments by testing whether there are significant differences in performance.  Creates a comparative plot."
    )
    parser.add_argument(
        "--exp",
        nargs="+",
        action="append",
        metavar=("experiment_folder", "experiment_params"),
        help="Experiments in one folder to compare. Can be specified multiple times",
    )
    parser.add_argument(
        "--output",
        default="evaluation_image.png",
        help="location and name under which the resulting image should be saved",
    )
    parser.add_argument("--folds", nargs="+", default=range(10))
    parser.add_argument(
        "--restrict-subjects",
        default=None,
        help='restrict subjects to compare to subjects with "FLAIR" or "noFLAIR" features',
    )
    parser.add_argument("--threshold", default="optimal", choices=["optimal", "0.5"], help="threshold to evaluate at")
    args = parser.parse_args()

    # experiments to be compared
    experiments_dictionary = {exp[0]: exp[1:] for exp in args.exp}
    # add all experiments for experiment_folder if none are specified
    for exp_folder, exp_param in experiments_dictionary.items():
        if len(exp_param) == 0:
            exp_name = os.path.basename(exp_folder)[:-9]  # remove the date from foldername
            files = glob(
                os.path.join(
                    paths.EXPERIMENT_PATH,
                    exp_folder,
                    "fold_{}".format(args.folds[0]),
                    "data_parameters_{}_*.json".format(exp_name),
                )
            )
            for f in files:
                exp_param.append(os.path.splitext(os.path.basename(f))[0].split("_")[-1])
            experiments_dictionary[exp_folder] = exp_param

    print(experiments_dictionary)

    # stats we want plots of.
    stats_of_interest = {
        "fold": ["threshold"],  # ,'subject_sensitivity','subject_specificity'],
        "per_patient": ["dice_index", "n_clusters"],
        "per_control": ["n_clusters"],
    }
    params_for_experiment_name = None  # {'network_parameters': ['focal_loss_alpha', 'focal_loss_gamma']}

    comparator = ExperimentComparison(
        experiments_dictionary,
        folds=args.folds,
        params_for_experiment_name=params_for_experiment_name,
        restrict_subjects=args.restrict_subjects,
        threshold=args.threshold,
    )
    comparator.calculate_per_patient_ranks(
        stats_of_interest=stats_of_interest["per_patient"],
        subexperiments=np.unique(comparator.patients_df["subexperiment"])[:2],
    )

    comparator.plot_stats_of_interest(stats_of_interest=stats_of_interest, save=args.output)
