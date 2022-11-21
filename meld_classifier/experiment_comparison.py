import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from PIL import ImageFile

ImageFile.LOAD_TRUNCATED_IMAGES = True
import ptitprince as pt


from meld_classifier.paths import EXPERIMENT_PATH
from meld_classifier.meld_cohort import MeldCohort
import os
import glob
import pandas as pd
from statsmodels.formula.api import ols
import statsmodels.api as sm
import scipy.stats as stats
import json
from textwrap import wrap


class ExperimentComparison:
    def __init__(
        self,
        experiments_dictionary,
        experiment_path=EXPERIMENT_PATH,
        folds=range(10),
        threshold="optimal",
        params_for_experiment_name=None,
        restrict_subjects=None,
    ):
        """
        Class for comparing experiments. Calculates experiment statistics and generates plots summarising comparisons

        Args:
            threshold (string): "optimal" or "0.5"
            params_for_experiment_name: optional dict containing data and network parameters that should be included
                in the experiment name on the plots (useful when comparing experiments that vary two parameters)
            restrict_subjects (optional, string): "FLAIR" "noFLAIR", restrict subjects to compare to a subset of the original test subjects.
                Either use only subjects with FLAIR features, or only subjects without FLAIR features
        """
        self.threshold = threshold
        self.experiments_dictionary = experiments_dictionary
        self.experiment_path = experiment_path
        self.experiment_folders = list(experiments_dictionary.keys())
        self.params_for_experiment_name = params_for_experiment_name
        self.folds = folds
        self.restrict_subjects = restrict_subjects
        self.folds_df, self.fold_statistics = self.load_fold_results()
        self.patients_df, self.controls_df = self.load_subject_results()

    # --- data loading functions ---
    def load_subject_results(self):
        patient_stats = []
        control_stats = []
        patient_ids = []
        control_ids = []
        sub_column_p = []
        sub_column_c = []
        fold_column = []
        for folder in self.experiment_folders:
            for param in self.experiments_dictionary[folder]:
                for fold in self.folds:

                    # get the name by which the experiment should be represented
                    exp_name = self._get_experiment_name(
                        folder,
                        param,
                        fold=fold,
                        use_params=self.params_for_experiment_name is not None,
                        params=self.params_for_experiment_name,
                    )
                    experiment_variable = os.path.basename(folder)[:-9]
                    # load per-subject results
                    fold_dict = self._load_json(
                        os.path.join(
                            self.experiment_path,
                            folder,
                            "fold_{}".format(fold),
                            "results",
                            "per_subject_{}_{}_{}.json".format(experiment_variable, param, self.threshold),
                        )
                    )
                    # get data parameters (needed to know how to filter subjects)
                    data_parameters = json.load(
                        open(
                            os.path.join(
                                self.experiment_path,
                                folder,
                                "fold_{}".format(fold),
                                "data_parameters_{}_{}.json".format(experiment_variable, param),
                            )
                        )
                    )
                    subject_ids = self.filter_subjects(
                        list(fold_dict["patients"].keys()), hdf5_file_root=data_parameters["hdf5_file_root"]
                    )
                    for subject in sorted(subject_ids):
                        patient_stats.append(fold_dict["patients"][subject])
                        patient_ids.append(subject)
                        sub_column_p.append(exp_name)
                        fold_column.append(fold)
                    subject_ids = self.filter_subjects(
                        list(fold_dict["controls"].keys()), hdf5_file_root=data_parameters["hdf5_file_root"]
                    )
                    for subject in sorted(subject_ids):
                        control_stats.append(fold_dict["controls"][subject])
                        control_ids.append(subject)
                        sub_column_c.append(exp_name)
        patients_df = pd.DataFrame(patient_stats)
        patients_df["subexperiment"] = sub_column_p
        patients_df["subj_id"] = patient_ids
        patients_df = patients_df.rename(columns={0: "detected", 1: "n_clusters", 2: "dice_index"})
        patients_df["dice_index"] = np.log(patients_df["dice_index"] + 0.01)
        patients_df["specificity"] = patients_df["n_clusters"] == 0
        patients_df["n_clusters"] = np.log(patients_df["n_clusters"] + 0.5)
        patients_df["fold"] = fold_column

        controls_df = pd.DataFrame(control_stats)
        controls_df["subexperiment"] = sub_column_c
        controls_df["subj_id"] = control_ids
        controls_df = controls_df.rename(columns={0: "any_clusters", 1: "n_clusters"})
        controls_df["specificity"] = controls_df["n_clusters"] == 0
        controls_df["n_clusters"] = np.log(controls_df["n_clusters"] + 0.5)
        return patients_df, controls_df

    def filter_subjects(self, subject_ids, hdf5_file_root="{site_code}_{group}_featuremetrix.hdf5"):
        """filter subjects to FLAIR or no FLAIR, depending on self.restrict_subjects.
        Note: this is independent of the features that the model was actually trained on.
        It looks in the hdf5 and thus filters on general availability of FLAIR or not
        """
        if self.restrict_subjects is None:
            return subject_ids
        else:
            c = MeldCohort(hdf5_file_root=hdf5_file_root)
            # get all FLAIR subjects
            all_flair_subject_ids = cohort.get_subject_ids(subject_features_to_exclude=["FLAIR"])
            # restrict subjects to those that have flair features
            flair_subject_ids = [subj_id for subj_id in subject_ids if subj_id in all_flair_subject_ids]

            if self.restrict_subjects == "FLAIR":
                print("using {} of {} subjects".format(len(flair_subject_ids), len(subject_ids)))
                return flair_subject_ids
            elif self.restrict_subjects == "noFLAIR":
                # return difference between all subjects and flair subjects (resulting in those that dont have flair)
                noflair_subject_ids = list(np.setdiff1d(subject_ids, flair_subject_ids))
                print("using {} of {} subjects".format(len(noflair_subject_ids), len(subject_ids)))
                return noflair_subject_ids
            else:
                raise NotImplementedError(self.restrict_subjects)

    def load_fold_results(self):
        folds_column = []
        fold_stats = []
        sub_column = []

        for fold in self.folds:
            for folder in self.experiment_folders:
                for param in self.experiments_dictionary[folder]:
                    # extract variable name omitting the date
                    experiment_variable = os.path.basename(folder)[:-9]
                    exp_name = self._get_experiment_name(
                        folder,
                        param,
                        fold=fold,
                        use_params=self.params_for_experiment_name is not None,
                        params=self.params_for_experiment_name,
                    )
                    stats_df = pd.read_csv(
                        os.path.join(
                            self.experiment_path,
                            folder,
                            "fold_{}".format(fold),
                            "results",
                            "test_results_{}_{}.csv".format(experiment_variable, param),
                        )
                    )
                    folds_column.append(fold)

                    sub_column.append(exp_name)
                    if self.threshold == "optimal":
                        fold_stats.append(stats_df.loc[1])
                    elif self.threshold == "0.5":
                        fold_stats.append(stats_df.loc[0])
            # get names of statistics from one of the dataframes
            fold_statistics = list(stats_df.columns)
        # format into nice table
        folds_df = pd.DataFrame(fold_stats)
        folds_df["fold"] = folds_column
        folds_df["subexperiment"] = sub_column
        return folds_df, fold_statistics

    def _load_json(self, json_file):
        with open(json_file, "r") as f:
            results_dict = json.load(f)
        return results_dict

    def _get_experiment_name(self, experiment_folder, param_value, fold=0, use_params=False, params={}):
        exp_name = os.path.basename(experiment_folder)[:-9]
        if use_params is False:
            # take original experiment name consisting of parameter to vary + parameter value
            # remove date from experiment_folder (9 characters)
            name = "{}_{}".format(exp_name, param_value)
        else:
            # use format: parameter1_value1-parameter2_value2...
            exp_path = os.path.join(self.experiment_path, experiment_folder, "fold_{}".format(fold))
            data_params = self._load_json(
                os.path.join(exp_path, "data_parameters_{}_{}.json".format(exp_name, param_value))
            )
            network_params = self._load_json(
                os.path.join(exp_path, "network_parameters_{}_{}.json".format(exp_name, param_value))
            )
            name = []
            for p in params.get("data_parameters", []):
                name.append("{}_{}".format(p, data_params[p]))
            for p in params.get("network_parameters", []):
                name.append("{}_{}".format(p, network_params[p]))
            name = "-".join(name)
        return name

    # --- calculate comparison functions ---
    def calculate_per_patient_ranks(self, stats_of_interest, subexperiments):
        dataframe = self.patients_df
        print(stats_of_interest, subexperiments)
        df1 = dataframe[["subj_id", "fold", stats_of_interest[0], stats_of_interest[1]]][
            dataframe["subexperiment"] == subexperiments[0]
        ]
        df2 = dataframe[["subj_id", "fold", stats_of_interest[0], stats_of_interest[1]]][
            dataframe["subexperiment"] == subexperiments[1]
        ]
        df1 = df1.reset_index(drop=True)
        df2 = df2.reset_index(drop=True)
        diff_df = df1.copy()
        diff_df[stats_of_interest[0]] = df1[stats_of_interest[0]] - df2[stats_of_interest[0]]
        diff_df[stats_of_interest[1]] = df1[stats_of_interest[1]] - df2[stats_of_interest[1]]
        sorted = diff_df.sort_values(by=["dice_index"])
        sorted.to_csv(
            os.path.join(
                self.experiment_path,
                self.experiment_folders[0],
                "per_patient_differences_{}-{}.csv".format(subexperiments[0], subexperiments[1]),
            ),
            index=False,
        )
        return

    def anova(self, dataframe, statistic):
        """test independence of different experiments"""
        mod = ols('Q("{}") ~ Q("{}")'.format(statistic, "subexperiment"), data=dataframe).fit()
        try:
            aov_table = sm.stats.anova_lm(mod, typ=2)
        except ValueError:
            aov_table = sm.stats.anova_lm(mod, typ=1)
        stat_ = np.array(aov_table)[0, 2]
        p = np.array(aov_table)[0, 3]
        return stat_, p

    def anova_rm(self, dataframe, statistic):
        from statsmodels.stats.anova import AnovaRM

        if "subj_id" in dataframe.columns:
            aovrm = AnovaRM(dataframe, statistic, "subj_id", within=["subexperiment"])
        else:
            aovrm = AnovaRM(dataframe, statistic, "fold", within=["subexperiment"])
        aov_table = aovrm.fit()
        aov_table = np.array(aov_table.summary())
        stat_ = np.array(aov_table)[0, 0]
        p = np.array(aov_table)[0, 3]
        return stat_, p

    # TODO extract pairwise t-test and chi2 stat from plotting functions?

    # --- plotting functions ---
    def plot_pairwise_test(self, dataframe, statistic, ax=None):
        subexperiments = np.sort(np.unique(dataframe["subexperiment"]))
        # calculate pairwise t-test
        grid = np.zeros((len(subexperiments), len(subexperiments)))
        p_grid = np.zeros((len(subexperiments), len(subexperiments)))
        for k, exp1 in enumerate(subexperiments):
            for j, exp2 in enumerate(subexperiments):
                if k != j:
                    vals1 = dataframe[statistic][dataframe["subexperiment"] == exp1]
                    vals2 = dataframe[statistic][dataframe["subexperiment"] == exp2]
                    try:
                        t, p = stats.ttest_rel(vals1, vals2)
                    except ValueError:
                        t, p = stats.ttest_ind(vals1, vals2)
                    grid[k, j] = t
                    p_grid[k, j] = p
        # plot
        if ax is None:
            fig, ax = plt.subplots(1, 1)
        ax.matshow(grid, cmap="bwr", vmin=-2, vmax=2)
        for (i, j), z in np.ndenumerate(grid):
            ax.text(j, i, "{:0.2f}".format(z), ha="center", va="center")
        ax.xaxis.set_ticks_position("bottom")
        ax.set_xticks(np.arange(len(subexperiments)))
        ax.set_yticks(np.arange(len(subexperiments)))
        ax.set_xticklabels(["\n".join(wrap(exp, 20)) for exp in subexperiments])
        ax.set_yticklabels(["\n".join(wrap(exp, 20)) for exp in subexperiments])
        ax.set_title("{} in {}\npairwise t-tests".format(statistic, self.p_c))
        return ax

    def plot_chi2_test(self, statistic="sensitivity", ax=None):
        """test per subject sensitivity and specificity changes"""
        subexperiments = np.sort(np.unique(self.patients_df["subexperiment"]))
        # calculate ch2 statistic
        n_sub_experiments = len(subexperiments)
        grid = np.zeros((n_sub_experiments, n_sub_experiments))
        p_grid = np.zeros((n_sub_experiments, n_sub_experiments))
        names = ["sensitivity", "specificity"]
        score = []
        if statistic == "sensitivity":
            s = "detected"
            data_frame = self.patients_df
            title = "sensitivity in patients"

        else:
            s = statistic
            title = "specificity in controls"
            data_frame = self.controls_df

        for k, exp1 in enumerate(subexperiments):
            for j, exp2 in enumerate(subexperiments):
                if k != j:
                    detected = data_frame[s][data_frame["subexperiment"].isin([exp1, exp2])]
                    subexperiment = data_frame["subexperiment"][data_frame["subexperiment"].isin([exp1, exp2])]
                    table = pd.crosstab(detected, subexperiment)
                    table = sm.stats.Table(table)
                    rslt = table.test_ordinal_association()
                    if k > j:
                        zscore = rslt.zscore
                    else:
                        zscore = -rslt.zscore
                    grid[k, j] = zscore
                    p_grid[k, j] = rslt.pvalue
            score.append(np.mean(data_frame[s][data_frame["subexperiment"] == exp1]))

        # plot
        if ax is None:
            fig, ax = plt.subplots(1, 1)
        ax.matshow(grid, cmap="bwr", vmin=-2, vmax=2)
        for (i, j), z in np.ndenumerate(grid):
            ax.text(j, i, "{:0.2f}".format(z), ha="center", va="center")
        ax.xaxis.set_ticks_position("bottom")
        ax.set_xticks(np.arange(n_sub_experiments))
        ax.set_xticklabels(["{:.2f}".format(s) for s in score])
        ax.set_yticks(np.arange(n_sub_experiments))
        ax.set_yticklabels(["\n".join(wrap(exp, 20)) for exp in subexperiments])
        ax.set_title("{}\nchi2 test".format(title))
        return ax

    def plot_experiment_statistic(self, dataframe, statistic, ax=None):
        """
        experiments by statistic (subject specificity, threshold, etc) plot.

        The values for each experiment are shown with a raincloud plot.
        The title of the plot contains the results of an anova test which assesses
        whether differences between the experiments are significant.
        """
        # get summarizing stats comparing all experiments
        val, p = self.anova(dataframe, statistic)
        subexperiments = np.sort(np.unique(dataframe["subexperiment"]))

        # plot
        if ax is None:
            fig, ax = plt.subplots(1, 1)
        pt.RainCloud(
            x="subexperiment",
            y=statistic,
            data=dataframe,
            palette="Set2",
            bw=0.2,
            width_viol=0.6,
            ax=ax,
            orient="h",
            order=subexperiments,
        )

        ax.set_title("{} \nStatistic: {:.2f}, p value: {:.2f}".format(statistic, val, p))
        lbls = ax.get_yticklabels()
        ax.set_yticklabels(["\n".join(wrap(l.get_text(), 20)) for l in lbls])
        # reset xticks to be real numbers if clusters, rather than log
        if statistic == "n_clusters":
            xticks = np.array([0, 1, 2, 5, 10, 20])
            vals = np.log(xticks + 0.5)
            ax.set_xticks(vals)
            ax.set_xticklabels(xticks)
        return ax

    def plot_stats_of_interest(self, stats_of_interest, save=None):
        # TODO: how to call per-patient chi2 test?
        # currenty, is just always plotted
        if self.restrict_subjects is not None:
            # do not plot per fold statistics because are not correct for a reduced subject list
            if "fold" in stats_of_interest:
                if "threshold" in stats_of_interest["fold"]:
                    stats_of_interest["fold"] = ["threshold"]
                else:
                    del stats_of_interest["fold"]
        ncols = 2
        nrows = sum([len(vals) for vals in stats_of_interest.values()]) + 1
        fig, axes = plt.subplots(nrows, ncols, figsize=(12, 5 * nrows))
        if self.restrict_subjects is not None:
            fig.suptitle("Comparison restricted to {}".format(self.restrict_subjects))
        i = 0
        self.plot_chi2_test("sensitivity", ax=axes[i, 0])
        self.plot_chi2_test("specificity", ax=axes[i, 1])
        i += 1

        for df_name in stats_of_interest.keys():
            if df_name == "fold":
                dataframe = self.folds_df
                self.p_c = ""
            elif df_name == "per_patient":
                dataframe = self.patients_df
                self.p_c = "patients"
            elif df_name == "per_control":
                dataframe = self.controls_df
                self.p_c = "controls"
            for statistic in stats_of_interest[df_name]:
                self.plot_experiment_statistic(dataframe, statistic, ax=axes[i, 0])
                self.plot_pairwise_test(dataframe, statistic, ax=axes[i, 1])
                i += 1
        plt.tight_layout()
        if save is not None:
            plt.savefig(save)
