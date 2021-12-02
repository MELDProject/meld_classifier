from meld_classifier.evaluation import Evaluator
from meld_classifier.experiment import Experiment, load_config
from meld_classifier.meld_cohort import MeldCohort
import argparse
import numpy as np
import json
import os
import importlib
import sys
import h5py
from meld_classifier.paths import EXPERIMENT_PATH
import subprocess

# run with command:
# python ./test_and_stats.py --n-splits 100 --experiment-folder iteration_21-04-23/ensemble_21-04-26 --experiment-name ensemble_iteration --plot-images --fold all --saliency --save-per-split-files --run-on-slurm


def submit_test_and_stats_array(args, splits_to_run):
    """submit slurm sbatch job array"""
    # get script path
    dir_path = os.path.dirname(os.path.realpath(__file__))
    if args.run_on_cpu:
        script_path = os.path.abspath(os.path.join(dir_path, "../hpc/run_array_cpu_plotting.sbatch"))
    else:
        script_path = os.path.abspath(os.path.join(dir_path, "../hpc/run_array_gpu_plotting.sbatch"))
    # start array sbatch script
    params = ",".join(
        [
            f"n_splits={args.n_splits}",
            f"experiment_folder={args.experiment_folder}",
            f"experiment_name={args.experiment_name}",
            f"plot_images={'--plot-images' if args.plot_images else ''}",
            f"saliency={'--saliency' if args.saliency else ''}",
            f"test_mode={args.test_mode if args.test_mode else False}",
            f"fold={args.fold}",
            f"save_per_split_files={'--save-per-split-files' if args.save_per_split_files else ''}",
        ]
    )

    subprocess.run(
        [
            "sbatch",
            "--array",
            splits_to_run,
            f"--export={params}",
            script_path,
        ]
    )


def test_and_stats(args):
    # need to pass subject_ids and new hdf5 file root
    experiment_path = os.path.join(EXPERIMENT_PATH, args.experiment_folder, f"fold_{args.fold}")
    exp = Experiment(experiment_path=experiment_path, experiment_name=args.experiment_name)
    exp.init_logging()

    # Also need to add target file name for saving in test mode.
    # needs to be passed to load_predict_single_subject and downstream functions (~line 500 in evaluation.py)

    subject_ids = exp.data_parameters["test_ids"]
    save_dir = None

    # if mode test : load information to predict on new subjects
    if args.test_mode != "False":
        config = load_config(args.test_mode)
        new_data_parameters = config.data_parameters
        exp.cohort = MeldCohort(
            hdf5_file_root=new_data_parameters["hdf5_file_root"], dataset=new_data_parameters["dataset"]
        )
        subject_ids = exp.cohort.get_subject_ids(**new_data_parameters, lesional_only=False)
        save_dir = new_data_parameters["saved_hdf5_dir"]
        # create sub-folders if do not exist
        fold_name = "fold_" + str(args.fold)
        try:
            os.mkdir(os.path.join(save_dir, fold_name))
            os.mkdir(os.path.join(save_dir, fold_name, "results"))
            if args.plot_images:
                os.mkdir(os.path.join(save_dir, fold_name, "results", "images"))
        except OSError as error:
            print(error)
        save_dir = os.path.join(save_dir, fold_name)
    eva = Evaluator(exp, mode="inference", subject_ids=subject_ids, save_dir=save_dir)

    suffix = ""
    if args.save_per_split_files:
        suffix = f"_{args.split}"
    print("suffix at start", suffix)

    for subject in np.array_split(subject_ids, args.n_splits)[args.split]:
        print("suffix", suffix, "subject", subject)
        fname = os.path.join(experiment_path, "results", f"predictions_{args.experiment_name}_{args.split}.hdf5")
        # check if subject already exists
        # TODO might want to change this behaviour at some point
        if os.path.isfile(fname):
            with h5py.File(fname, mode="r") as f:
                if f"{subject}/lh/integrated_gradients_pred" in f:
                    print(f"subject {subject} already present, skipping this.")
                    continue
        fold = None
        if args.fold != "all":
            fold = args.fold
        eva.load_predict_single_subject(
            subject, fold=fold, plot=args.plot_images, saliency=args.saliency, suffix=suffix
        )


def get_splits_to_rerun(args):
    # get failed experiments
    experiment_path = os.path.join(EXPERIMENT_PATH, args.experiment_folder, f"fold_{args.fold}")
    exp = Experiment(experiment_path=experiment_path, experiment_name=args.experiment_name)

    subject_ids = exp.data_parameters["test_ids"]

    fname = os.path.join(experiment_path, "results", "predictions_" + args.experiment_name + "_{}.hdf5")

    splits_to_run = None
    for split in range(0, args.n_splits):
        rerun_split = False
        if not os.path.exists(fname.format(split)):
            rerun_split = True
        else:
            with h5py.File(fname.format(split), mode="r") as f:
                # check if all subjects present
                for subject in np.array_split(subject_ids, args.n_splits)[split]:
                    if not f"{subject}/lh/integrated_gradients_pred" in f:
                        rerun_split = True
        if rerun_split:
            if splits_to_run is None:
                splits_to_run = f"{split}"
            else:
                splits_to_run += f",{split}"

    return splits_to_run


if __name__ == "__main__":
    # Set up experiment
    parser = argparse.ArgumentParser(description="Predict test subjects, plot flat maps, and calculate saliency")
    parser.add_argument("--split", type=int, default=0, help="index of the split of ids to use")
    parser.add_argument(
        "--n-splits",
        type=int,
        default=1,
        help="number of splits in the list of ids to use. allowing parallel processing",
    )
    parser.add_argument(
        "--experiment-folder",
        help="Experiments folder",
    )
    parser.add_argument(
        "--experiment-name",
        help="subfolder to use, typically the ensemble model",
        default="ensemble_0",
    )
    parser.add_argument(
        "--plot-images",
        action="store_true",
        help="do the flat-map plotting",
    )
    parser.add_argument("--saliency", action="store_true", help="calculate integrated gradients")
    parser.add_argument("--fold", default="all", help="fold number to use (by default all)")
    parser.add_argument(
        "--save-per-split-files",
        action="store_true",
        help="whether to store results for each split in a separate file. Can be concatenated later with 'merge_prediction_files.py'",
    )
    parser.add_argument(
        "--run-on-slurm",
        dest="run_on_slurm",
        action="store_true",
        help="schedule a sbatch script for each split. only works on a slurm cluster. In this case all splits are run",
    )
    parser.add_argument("--run-on-cpu", dest="run_on_cpu", action="store_true")
    parser.add_argument(
        "--rerun",
        action="store_true",
        help="use this flag to only rerun non-existing splits. Otherwise all splits are overwritten. Only works with --save-per-split-files",
    )
    parser.add_argument(
        "--test-mode",
        help="test mode if prediction on new cohort. Needs to provide a file with the new subjects information",
        default="False",
    )
    args = parser.parse_args()

    if args.run_on_slurm:
        # submit sbatch jobs
        if args.rerun:
            # get failed experiments
            splits_to_run = get_splits_to_rerun(args)
            if len(splits_to_run) == 0:
                print("No splits to rerun. Terminating")
                sys.exit(0)
            else:
                print(f"Rerunning splits {splits_to_run}")
        else:
            splits_to_run = f"0-{args.n_splits-1}"
        submit_test_and_stats_array(args, splits_to_run=splits_to_run)
    else:
        # directly run specified split
        test_and_stats(args)
