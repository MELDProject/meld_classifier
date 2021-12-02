from meld_classifier.experiment import Experiment
import meld_classifier.paths as paths
import os
import sys
import logging
import csv
import argparse
import subprocess

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"


def make_boolean(s):
    return s == "True"


if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description="Run meld_classifier on specified task. Normally, you should not call this script, but use hpc/run_array.sbatch or the --run-on-slurm flag on run.py, which will call this script"
    )
    parser.add_argument("task_id", type=int)
    parser.add_argument("run_params_file")
    args = parser.parse_args()

    # get experiment path and dir etc. from run_params_file
    with open(args.run_params_file, "r") as f:
        reader = csv.DictReader(f, delimiter=",")
        i = 0
        for row in reader:
            if i == args.task_id:
                break
            i += 1
        experiment_path = row["experiment_path"]
        experiment_name = row["experiment_name"]
        mode = row["mode"]
        optimise_threshold = make_boolean(row["optimise_threshold"])
        make_images = make_boolean(row["make_images"])
        if row["optimal_threshold"] in [None, ""]:
            optimal_threshold = None
        else:
            optimal_threshold = float(row["optimal_threshold"])
    # init logging
    exp = Experiment(experiment_name=experiment_name, experiment_path=experiment_path)
    exp.init_logging(console_level=logging.INFO)
    log = logging.getLogger(__name__)
    log.info("task id is {} {}".format(args.task_id, i))
    log.info("Processing experiment: {} at path {}".format(experiment_name, experiment_path))
    # run the experiment
    exp.run_experiment(
        mode=mode,
        optimise_threshold_flag=optimise_threshold,
        make_images_flag=make_images,
        optimal_threshold=optimal_threshold,
    )
