from meld_classifier.experiment import Experiment, save_config, load_config, submit_experiments_array
import meld_classifier.paths as paths
import numpy as np
import os
import sys
import logging
import csv
import argparse
from meld_classifier.set_basic import exclude_set

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"


def make_str(variable):
    # make string
    if isinstance(variable, list):
        variable_str = ",".join(map(str, variable))
    else:
        variable_str = str(variable)
    # exclude specific characters from file name
    list_exclude = ["{", "}", "_"]
    for l in list_exclude:
        if l in variable_str:
            variable_str = variable_str.replace(l, "")
    return variable_str


def get_fold_ns(data_parameters):
    fold_n = data_parameters["fold_n"]
    if isinstance(fold_n, list):
        return fold_n
    else:
        return [fold_n]


if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description="Entry point to meld_classifier. Trains and tests neural networks given a config file"
    )
    parser.add_argument(
        "mode",
        choices=["train", "eval", "all"],
        help="train: train model and optimize threshold (can be turned off with --no-optimise-threshold). eval: evaluate model and make images (if turned on with --make-images). all: both train and eval",
    )
    parser.add_argument(
        "--make-images",
        dest="make_images",
        action="store_true",
        help="turn plotting of images on (for mode eval/all). default: off",
    )
    parser.add_argument(
        "--no-make-images",
        dest="make_images",
        action="store_false",
        help="turn plotting of images off (also applies to mode eval)",
    )
    parser.add_argument(
        "--optimise-threshold",
        dest="optimise_threshold",
        action="store_true",
        help="turn optimisation of threshold on (also applies to mode eval). default: on ",
    )
    parser.add_argument(
        "--no-optimise-threshold",
        dest="optimise_threshold",
        action="store_false",
        help="turn optimisation of threshold off (for mode train/all)",
    )
    parser.set_defaults(make_images=False, optimise_threshold=True)
    parser.add_argument("--config-file", help="path to experiment_config.py file", default="experiment_config.py")
    parser.add_argument("--base-experiment-path", help="path to the experiments folder", default=paths.EXPERIMENT_PATH)
    parser.add_argument(
        "--run-on-slurm",
        dest="run_on_slurm",
        action="store_true",
        help="schedule an sbatch script for each experiment. Only works on a slurm cluster",
    )
    parser.add_argument("--run-on-cpu", dest="run_on_cpu", action="store_true")
    parser.add_argument(
        "--verbose",
        help="verbosity of console output. values: debug 10, info 20, warning 30, error 50. default: 20",
        type=int,
        default=logging.INFO,
    )
    parser.add_argument(
        "--optimal-threshold",
        default=None,
        type=float,
        help="manually set the optimal threshold. Use this flag together with --optimise-threshold to set the threshold (will not do the optimisation)",
    )
    parser.add_argument("--make-prediction-space", action="store_true", help="plot prediction space.")
    args = parser.parse_args()

    config = load_config(args.config_file)
    variable_network_parameters = config.variable_network_parameters
    network_parameters = config.network_parameters
    variable_data_parameters = config.variable_data_parameters
    data_parameters = config.data_parameters
    
    #if features_to_exclude is referring to a template set, replace
    if data_parameters['features_to_exclude'] in np.array(exclude_set.keys()):
        data_parameters['features_to_exclude'] = exclude_set[data_parameters['features_to_exclude']]
    # create list of experiments to run
    experiment_parameters = []
    # first, add experiments for variable_network_parameters
    for param in variable_network_parameters.keys():
        for variable in variable_network_parameters[param]:
            experiment_name = "{}_{}".format(param, make_str(variable))
            for fold_n in get_fold_ns(data_parameters):
                experiment_path = os.path.join(
                    args.base_experiment_path,
                    "{}_{}".format(param, network_parameters["date"]),
                    "fold_{}".format(fold_n),
                )
                cur_data_parameters = data_parameters.copy()
                cur_data_parameters["fold_n"] = fold_n
                cur_network_parameters = network_parameters.copy()
                cur_network_parameters[param] = variable
                experiment_parameters.append(
                    [experiment_path, experiment_name, cur_data_parameters, cur_network_parameters]
                )
    # then, add experiments for variable_data_parameters
    for param in variable_data_parameters.keys():
        variables = variable_data_parameters[param]
        for variable in variables:
            experiment_name = "{}_{}".format(param, make_str(variable))
            if isinstance(variables, dict):
                variable = variables[variable]  # get actual parm values
            for fold_n in get_fold_ns(data_parameters):
                experiment_path = os.path.join(
                    args.base_experiment_path,
                    "{}_{}".format(param, network_parameters["date"]),
                    "fold_{}".format(fold_n),
                )
                cur_data_parameters = data_parameters.copy()
                cur_data_parameters["fold_n"] = fold_n
                cur_data_parameters[param] = variable
                # hack to allow long list of features not get stored in folder names. Loads in from dictionary of options
                if param == "features_to_exclude":
                    from meld_classifier.set_basic import exclude_set

                    cur_data_parameters[param] = exclude_set[variable]
                experiment_parameters.append(
                    [experiment_path, experiment_name, cur_data_parameters, network_parameters]
                )

    # create experiment folders (one for each param that should be changed in variable_network_parameters and variable_data_parameters)
    for param, vals in variable_network_parameters.items():
        experiment_dir = os.path.join(args.base_experiment_path, "{}_{}".format(param, network_parameters["date"]))
        os.makedirs(os.path.join(experiment_dir), exist_ok=True)
        # save config
        save_config(
            variable_network_parameters={param: vals},
            variable_data_parameters={},
            data_parameters=data_parameters,
            network_parameters=network_parameters,
            save_path=os.path.join(experiment_dir, "experiment_config.py"),
        )
    for param, vals in variable_data_parameters.items():
        experiment_dir = os.path.join(args.base_experiment_path, "{}_{}".format(param, network_parameters["date"]))
        os.makedirs(os.path.join(experiment_dir), exist_ok=True)
        # save config
        save_config(
            variable_network_parameters={},
            variable_data_parameters={param: vals},
            data_parameters=data_parameters,
            network_parameters=network_parameters,
            save_path=os.path.join(experiment_dir, "experiment_config.py"),
        )

    if args.run_on_slurm:
        # start a job array with one job per experiment
        # save experiment_paths and experiment_names in list to be used for slurm
        run_params_fname = os.path.realpath(
            os.path.join(
                os.path.dirname(experiment_parameters[0][0]),
                "..",
                "run_parallel_params_{}.csv".format(np.random.randint(low=0, high=10000)),
            )
        )
        num_runs = len(experiment_parameters)
        with open(run_params_fname, "w") as f:
            writer = csv.writer(f)
            writer.writerow(
                ["experiment_path", "experiment_name", "mode", "optimise_threshold", "make_images", "optimal_threshold"]
            )
            for el in experiment_parameters:
                writer.writerow(
                    [el[0], el[1], args.mode, args.optimise_threshold, args.make_images, args.optimal_threshold]
                )

        # create experiment structure and save parameters
        if args.mode != "eval":
            for experiment_path, experiment_name, cur_data_parameters, cur_network_parameters in experiment_parameters:
                # init experiment and save parameters
                exp = Experiment.create_with_parameters(
                    cur_data_parameters, cur_network_parameters, experiment_path, experiment_name
                )
            # otherwise, experiments are already initialized

        print("Submitting sbatch job array for {} experiments, defined in {}".format(num_runs, run_params_fname))
        submit_experiments_array(num_runs, run_params_fname, run_on_cpu=args.run_on_cpu)

    else:
        # directly run the experiments
        for experiment_path, experiment_name, cur_data_parameters, cur_network_parameters in experiment_parameters:
            if args.mode == "eval":
                # create experiment object, without resaving parameters
                exp = Experiment(experiment_path, experiment_name)
            else:
                # init and save experiment parameters
                exp = Experiment.create_with_parameters(
                    cur_data_parameters, cur_network_parameters, experiment_path, experiment_name
                )

            # init logging
            exp.init_logging(console_level=args.verbose)
            log = logging.getLogger(__name__)
            log.info("Processing experiment: {} at path {}".format(experiment_name, experiment_path))
            try:
                exp.run_experiment(
                    mode=args.mode,
                    optimise_threshold_flag=args.optimise_threshold,
                    make_images_flag=args.make_images,
                    make_prediction_space_flag=args.make_prediction_space,
                    optimal_threshold=args.optimal_threshold,
                )
            except KeyboardInterrupt:
                # catch ctrl-c commands and exit
                # NOTE this could also be done with a signal.signal() handler
                log.info("ctrl-c (SIGINT) received, ending process now.")
                sys.exit(0)
            except:
                # catch and log any unexpected exceptions
                log.exception(
                    "Exception while training/evaluating experiment {} at path {}".format(
                        experiment_name, experiment_path
                    )
                )
                # log.info('continuing with training')
                # NOTE alternative is to terminate here as well with sys.exit(0)
                sys.exit(0)
