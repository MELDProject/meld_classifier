import argparse
import numpy as np
import os
import meld_classifier.experiment as exp
import csv

if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description="Script to rerun failed experiments (only for slurm jobs). Checks for results folders and reruns in their absence"
    )
    parser.add_argument("--params-csv", help="path to params csv file where things failed", default="params.csv")
    parser.add_argument("--run-on-cpu", dest="run_on_cpu", action="store_true")
    args = parser.parse_args()

    # check which lines have no results already
    lines_to_rerun = []
    with open(args.params_csv, "r") as f:
        reader = csv.DictReader(f, delimiter=",")
        for i, row in enumerate(reader):

            experiment_path = row["experiment_path"]
            experiment_name = row["experiment_name"]
            if not os.path.isfile(
                os.path.join(experiment_path, "results", "per_subject_{}_optimal.json".format(experiment_name))
            ):
                lines_to_rerun.append(i)
    # reruns only these lines passing a comma separated list of them
    exp.resubmit_experiments_array(lines_to_rerun, args.params_csv, run_on_cpu=args.run_on_cpu)
