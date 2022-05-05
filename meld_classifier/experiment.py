# functions for creating and running experiments
import logging
import os
import json
import subprocess
import numpy as np
import tensorflow as tf

from meld_classifier.dataset import Dataset
from meld_classifier.network_tools import build_model
from meld_classifier.evaluation import Evaluator
from meld_classifier.training import Trainer
from meld_classifier.meld_cohort import MeldCohort, MeldSubject


def submit_experiments_array(num_runs, run_params_fname, run_on_cpu=False):
    """submit slurm sbatch job array"""
    # get script path
    dir_path = os.path.dirname(os.path.realpath(__file__))
    if run_on_cpu:
        script_path = os.path.abspath(os.path.join(dir_path, "../scripts/hpc/run_array_cpu.sbatch"))
    else:
        script_path = os.path.abspath(os.path.join(dir_path, "../scripts/hpc/run_array.sbatch"))
    # start array sbatch script
    # num_runs - 1
    subprocess.run(
        [
            "sbatch",
            "--array",
            "0-{}".format(num_runs - 1),
            '--export=run_params="{}"'.format(run_params_fname),
            script_path,
        ]
    )


def resubmit_experiments_array(num_list, run_params_fname, run_on_cpu=False):
    """resubmit slurm sbatch job array using list of failed exps"""
    # get script path
    dir_path = os.path.dirname(os.path.realpath(__file__))
    if run_on_cpu:
        script_path = os.path.abspath(os.path.join(dir_path, "../scripts/hpc/run_array_cpu.sbatch"))
    else:
        script_path = os.path.abspath(os.path.join(dir_path, "../scripts/hpc/run_array.sbatch"))
    # start array sbatch script
    subprocess.run(
        [
            "sbatch",
            "--array",
            ",".join(map(str, num_list)),
            '--export=run_params="{}"'.format(run_params_fname),
            script_path,
        ]
    )


def load_config(config_file):
    """load config.py file and return config object"""
    import importlib.machinery, importlib.util

    loader = importlib.machinery.SourceFileLoader("config", config_file)
    spec = importlib.util.spec_from_loader(loader.name, loader)
    config = importlib.util.module_from_spec(spec)
    loader.exec_module(config)
    return config


def save_config(variable_network_parameters, variable_data_parameters, data_parameters, network_parameters, save_path):
    """save config.py file to save_path. Resulting config.py file"""

    def str_repr(d, name):
        str_d = json.dumps(d, indent=4)
        str_d = str_d.replace(" true", " True")
        str_d = str_d.replace(" false", " False")
        str_d = str_d.replace(" null", " None")
        return "{} = {}".format(name, str_d)

    # only save network and data parameters
    with open(save_path, "w") as f:
        f.write(str_repr(variable_network_parameters, "variable_network_parameters"))
        f.write("\n")
        f.write(str_repr(variable_data_parameters, "variable_data_parameters"))
        f.write("\n")
        f.write(str_repr(data_parameters, "data_parameters"))
        f.write("\n")
        f.write(str_repr(network_parameters, "network_parameters"))
        f.write("\n")


class Experiment:
    def __init__(self, experiment_path, experiment_name):
        """
        Initialize an experiment from an existing path.
        Expects data_parameters.json and network_parameters.json in experiment_path.
        Args:
            experiment_path (str): path to the experiment folder (may contain several experiments)
            experiment_name (str): name of the current experiment (used to name files correctly)

        TODO give an error if have missing data_parameters or network_parameters
        """
        # init experiment
        self.path = experiment_path
        self.name = experiment_name
        self._log = None
        self.data_parameters = json.load(open(os.path.join(self.path, "data_parameters_{}.json".format(self.name))))
        self.network_parameters = json.load(
            open(os.path.join(self.path, "network_parameters_{}.json".format(self.name)))
        )
        self.cohort = MeldCohort(
            hdf5_file_root=self.data_parameters["hdf5_file_root"], dataset=self.data_parameters["dataset"]
        )
        # set by load_model
        self.model = None

    @classmethod
    def create_with_parameters(cls, data_parameters, network_parameters, experiment_path, experiment_name):
        """
        Set up an experiment with specified parameters.
        Create experiment folders and save parameters in experiment_path.
        For exemplary data_parameters and network_parameters, see the `scripts/experiment_config_template.py` file.
        """
        # create the experiment folders
        cls.create_experiment_folders(experiment_path)
        # save the parameters
        cls.save_experiment_parameters(data_parameters, network_parameters, experiment_path, experiment_name)
        # init Experiment object
        return cls(experiment_path, experiment_name)

    @staticmethod
    def create_experiment_folders(experiment_path):
        """create necessary directories for experiment"""
        os.makedirs(os.path.join(experiment_path), exist_ok=True)
        os.makedirs(os.path.join(experiment_path, "logs"), exist_ok=True)
        os.makedirs(os.path.join(experiment_path, "models"), exist_ok=True)
        os.makedirs(os.path.join(experiment_path, "results"), exist_ok=True)
        os.makedirs(os.path.join(experiment_path, "results", "images"), exist_ok=True)

    @staticmethod
    def exists_experiment(experiment_path, experiment_name):
        """check if experiment exists (i.e. if it has already been trained)
        check if data_parameters.json and network_parameters.json exist"""
        dparams = os.path.join(experiment_path, "data_parameters_{}.json".format(experiment_name))
        nparams = os.path.join(experiment_path, "network_parameters_{}.json".format(experiment_name))
        if os.path.exists(dparams) and os.path.exists(nparams):
            return True
        return False

    @staticmethod
    def save_experiment_parameters(data_parameters, network_parameters, experiment_path, experiment_name):
        """
        Save dictionaries to experiment_path using json
        """
        # data_parameters
        fname = os.path.join(experiment_path, "data_parameters_{}.json".format(experiment_name))
        json.dump(data_parameters, open(fname, "w"), indent=4)
        # network_parameters
        fname = os.path.join(experiment_path, "network_parameters_{}.json".format(experiment_name))
        json.dump(network_parameters, open(fname, "w"), indent=4)

    def save_parameters(self):
        return Experiment.save_experiment_parameters(
            self.data_parameters, self.network_parameters, self.path, self.name
        )

    def init_logging(self, console_level=logging.INFO):
        """
        Set up a logger for this experiment that logs to experiment_path and to stdout.
        Should only be called once per experiment (overwrites existing log files of the same name)
        """
        # remove all previous logging handlers associated with the root logger
        for handler in logging.root.handlers[:]:
            logging.root.removeHandler(handler)
        if not os.path.exists(os.path.join(self.path, "logs")):
            os.makedirs(os.path.join(self.path, "logs"))
        logging.basicConfig(
            level=logging.DEBUG,
            filename=os.path.join(
                self.path,
                "logs",
                f"{self.name}.log",
            ),
            filemode="w",
        )
        # also log warning messages to screen
        console = logging.StreamHandler()
        console.setLevel(console_level)
        logging.getLogger("").addHandler(console)
        # (mostly) silence tf logging
        tf_logger = logging.getLogger("tensorflow")
        tf_logger.setLevel(logging.ERROR)
        # (mostly) silence matplotlib logging
        mpl_logger = logging.getLogger("matplotlib")
        mpl_logger.setLevel(logging.WARNING)

    @property
    def log(self):
        if self._log is None:
            self._log = logging.getLogger(__name__)
        return self._log

    def n_features(self):
        """
        number of features that the model will be trained on.
        considers universal_features, self.features, and neighbours, demographic_features
        """
        features, _ = self.get_features()
        # features
        n_features = len(features)
        # universal features
        universal_features = self.data_parameters["universal_features"]
        if len(universal_features) > 0:
            if universal_features == ["coords"]:
                n_features += self.cohort.coords.shape[-1]
            else:
                raise NotImplementedError(universal_features)
        # demographic features
        n_features += len(self.data_parameters.get("demographic_features", []))
        # neighbours
        n_features = n_features * (self.data_parameters["num_neighbours"] + 1)
        return n_features

    def load_model(self, checkpoint_path=None):
        """
        build model and optionally load weights from checkpoint
        """
        if checkpoint_path is not None and os.path.isdir(checkpoint_path):
            # checkpoint contains both model architecture + weights
            self.log.info("Loading model architecture + weights from checkpoint")
            model = tf.keras.models.load_model(checkpoint_path)
        else:
            model = build_model(
                # build_model needs to know n_features without neighbours
                n_features=self.n_features() // (self.data_parameters["num_neighbours"] + 1),
                n_neighbours=self.data_parameters["num_neighbours"],
                **self.network_parameters,
            )
            if checkpoint_path is not None:
                model.load_weights(checkpoint_path)
        self.model = model

    # functions for training and evaluation
    def run_experiment(
        self,
        mode="train",
        optimise_threshold_flag=True,
        make_images_flag=False,
        make_prediction_space_flag=False,
        optimal_threshold=None,
    ):
        """run entire experiment"""
        if mode in ("train", "all"):
            # train model
            self.train()
            if optimise_threshold_flag:
                self.optimise_threshold(optimal_threshold=optimal_threshold)
        if mode in ("eval", "all"):
            # need to optimise threshold?
            if mode == "eval" and optimise_threshold_flag:
                self.optimise_threshold(optimal_threshold=optimal_threshold)
            # evaluate model
            self.evaluate(make_images_flag=make_images_flag, make_prediction_space_flag=make_prediction_space_flag)

    def train(self):
        """
        Train network.
        """
        trainer = Trainer(self)
        trainer.train()

    def evaluate(self, make_images_flag=False, make_prediction_space_flag=False):
        """Evaluate trained network.

        Results in a results csv file, and plots showing the predictions on brain surfaces.
        """
        if not Experiment.exists_experiment(self.path, self.name):
            self.log.error(
                f"experiment {self.path} with path {self.name} does not exist! Call create_with_parameters instead!"
            )
            return 0
        self.log.info("evaluating model...")
        # calculate evaluation metrics
        val_ev = Evaluator(
            experiment=self, mode="val", make_images=make_images_flag, make_prediction_space=make_prediction_space_flag
        )
        val_ev.evaluate()
        self.log.info("...evaluation complete")

    def optimise_threshold(self, optimal_threshold=None):
        """
        Optimize the lesion prediction threshold on the training set

        Writes the optimized threshold in the network_parameters.json
        """
        if not Experiment.exists_experiment(self.path, self.name):
            self.log.error(
                f"experiment {self.path} with path {self.name} does not exist! Call create_with_parameters instead!"
            )
            return 0
        if optimal_threshold is None:
            self.log.info("optimizing threshold...")
            # optimise threshold on subset of train_ids
            if isinstance(self.network_parameters["optimal_threshold"], float):
                self.log.warning(
                    f"optimal threshold {self.network_parameters['optimal_threshold']} defined in network_parameters will be overwritten!"
                )

            train_ev = Evaluator(experiment=self, mode="train")
            threshold, dice = train_ev.optimise_threshold(plot_curve=True)
            self.log.info(f"train dice index is {dice} optimal threshold is {threshold}")
        else:
            threshold = optimal_threshold
            self.log.info(
                f"given optimal threshold is {threshold}. Threshold defined in network_parameters will be overwritten!"
            )
        self.network_parameters["optimal_threshold"] = threshold
        # save network_parameters.json with calculated threshold
        self.save_parameters()
        self.log.info("...threshold optimization complete")

    def get_features(self):
        """
        get list of features that model should be trained on.
        Either read from data_parameters, or calculated and written to data_parameters
        """
        if "features" not in self.data_parameters:
            self.log.info("get features to train on")
            # get features
            #print(self.data_parameters["features_to_exclude"])
            features = self.cohort.get_features(features_to_exclude=self.data_parameters["features_to_exclude"])
            # get features that should be ignored
            _, features_to_ignore = self.cohort._filter_features(
                features_to_exclude=self.data_parameters.get("features_to_replace_with_0", []), return_excluded=True
            )
            self.log.debug(f"features {features}")
            self.log.debug(f"features_to_ignore {features_to_ignore}")

            # put train_ids, val_ids, test_ids, features in data_parameters
            self.data_parameters.update(
                {
                    "features": features,
                    "features_to_replace_with_0": features_to_ignore,
                }
            )
            # save updated data_parameters
            self.save_parameters()
        return self.data_parameters["features"], self.data_parameters["features_to_replace_with_0"]

    def get_train_val_test_ids(self):
        """
        return train val test ids.
        Either read from data_parameters (if exist), or created using _train_val_test_split_folds.

        returns train_ids, val_ids, test_ids
        """
        if "train_ids" not in self.data_parameters:
            self.log.info("getting train val test split")
            # get subject ids restricted to desired subjects
            subject_ids = self.cohort.get_subject_ids(**self.data_parameters)
            # get train val test split
            train_ids, val_ids, test_ids = self._train_val_test_split_folds(
                subject_ids,
                iteration=self.data_parameters["fold_n"],
                number_of_folds=self.data_parameters["number_of_folds"],
            )
            # put in data_parameters
            self.data_parameters.update(
                {
                    "train_ids": list(train_ids),
                    "test_ids": list(test_ids),
                    "val_ids": list(val_ids),
                }
            )
            # save updated data_parameters
            self.save_parameters()
        return self.data_parameters["train_ids"], self.data_parameters["val_ids"], self.data_parameters["test_ids"]

    def _train_val_test_split_folds(self, subject_ids, iteration=0, number_of_folds=10):
        """split subject_ids into train val and test.
        test_ids are defined in dataset_name.
        The remaining ids are split randomly (but with a fixed seed) in number_of_folds folds.

        Args:
            list_ids (list of str): subject ids to split
            number_of_folds (int): number of folds to split the train/val ids into
            iteration (int): number of validation fold, values 0,..,number_of_folds-1
        Returns:
            train_ids, val_ids, test_ids
        """
        np.random.seed(0)

        _, dataset_trainval_ids, dataset_test_ids = self.cohort.read_subject_ids_from_dataset()
        subject_ids = np.array(subject_ids)

        # get test_ids
        test_mask = np.in1d(subject_ids, dataset_test_ids)
        test_ids = subject_ids[test_mask]

        # get trainval_ids
        trainval_ids = subject_ids[~test_mask]
        trainval_ids = np.intersect1d(trainval_ids, dataset_trainval_ids)
        # split trainval_ids in folds
        np.random.shuffle(trainval_ids)
        folds = np.array_split(trainval_ids, number_of_folds)
        folds = np.roll(folds, shift=iteration, axis=0)
        train_ids = np.concatenate(folds[0:-1]).ravel()
        val_ids = folds[-1]
        return train_ids, val_ids, test_ids
