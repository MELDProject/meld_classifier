import datetime

# Usage: rename this file to experiment_config.py,
# adjust according to your needs,
# and run the experiments with `python run.py --config_file experiment_config.py`

##### variable parameters #####
# variables that define the experiments
# network or data parameters can be varied
# one experiment will be created for each varied paramater (network or data)
# NOTE at least one entry on variable_network_parameters or variable_data_parameters
# is required
variable_network_parameters = {
    # each entry here is a parameter and a list of values
    # e.g. 'learning_rate': [0.1,0.2,0.3]
}

variable_data_parameters = {
    # each entry here is a parameter and a list of values
    # e.g. 'iteration': [0,1,2,3,4,5]
    "iteration": [0, 1, 2, 3, 4, 5],
}

##### default input data parameters #####
data_parameters = {
    ##### selection of subject_ids and features to train/val/test on #####
    # `site_codes`: hospital sites to include,
    # valid inputs are: ['H1', 'H2', 'H3','H4','H5','H6','H7','H9','H10','H11',
    #    'H12','H14','H15','H16','H17','H18','H19', 'H21','H23','H24','H26','H27']
    "site_codes": [
        "H1",
        "H2",
        "H3",
        "H4",
        "H5",
        "H6",
        "H7",
        "H9",
        "H10",
        "H11",
        "H12",
        "H14",
        "H15",
        "H16",
        "H17",
        "H18",
        "H19",
        "H21",
        "H23",
        "H24",
        "H26",
        "H27",
    ],
    # `scanners`: include subjects with these scanners, valid inputs are ['15T','3T']
    "scanners": ["15T", "3T"],
    # `hdf5_file_root`: root hdf5 filename to be loaded.
    # This allows toggling between raw and combat normalized features.
    # Possible values: TODO update list
    #  - '{site_code}_{site_code}_featurematrix.hdf5' for non-combat normalized data
    #  - '{site_code}_{site_code}_featurematrix_combat.hdf5' for combat normalized data
    #  - '{site_code}_{site_code}_featurematrix_combat_2.hdf5' for combat normalized data
    #  - '{site_code}_{site_code}_featurematrix_combat_3.hdf5' for combat normalized data with H27
    "hdf5_file_root": "{site_code}_{group}_featurematrix_combat_6.hdf5",
    # `dataset`: name of the dataset file containing a list of subjects included in this version of the dataset
    # also contains a trainval/test split of the subjects that is to be respected by all models using this dataset
    # Possible values: TODO update list
    # - 'MELD_dataset_V3.csv' contains H27
    "dataset": "MELD_dataset_V6.csv",
    # `group`: initial section of subjects for training, can select patients only ('patient')
    # or controls+patients ('both')
    "group": "both",
    # `features_to_exclude`: features that should be excluded from training.
    # the entire feature name has to be specified, except for FLAIR features, which can be excluded using 'FLAIR'
    "features_to_exclude": [  # exclude unsmoothed curv and sulc features
        ".combat.on_lh.curv.mgh",
        ".combat.on_lh.sulc.mgh",
        ".inter_z.asym.intra_z.combat.on_lh.curv.mgh",
        ".inter_z.asym.intra_z.combat.on_lh.sulc.mgh",
        ".inter_z.intra_z.combat.on_lh.curv.mgh",
        ".inter_z.intra_z.combat.on_lh.sulc.mgh",
    ],
    # TODO comment
    # Possible values: ['coords'] Make sure it's in list otherwise loops over string
    "universal_features": [],
    "demographic_features": [],
    # `subject_features_to_exlude`: remove subjects that do not have these features.
    # This neccesary when comparing +/- FLAIR models so that you have the same subjects.
    # If you are including FLAIR as a feature and want to train on  everyone set to ''
    "subject_features_to_exclude": [""],
    # `features_to_replace_with_0`: features that should be included in the training, but have 0 as a value.
    # Useful for ablation study on including dummy feature variables
    "features_to_replace_with_0": [""],
    # `num_neighbours`: number of neighboring vertices whose features should be added to the current vertex
    # possible values are between 0 and 5
    "num_neighbours": 0,
    # `min_area_threshold`: minimum area threshold for a cluster to be considered lesional # TODO evaluation parameter!!!
    "min_area_threshold": 100,
    ##### train/val/test split #####
    # `number_of_folds`: split trainval data in number_of_folds folds
    # One will be used for validation, the rest for training.
    "number_of_folds": 10,
    # `fold_n`: nth fold from number_of_folds. Sets the number of the validation fold.
    # Can be a list (e.g. [0,1,2,3,4,5,6,7,8,9]), in this case all experiments
    "fold_n": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
    "iteration": 0,  # made up variable to iterate same set up multiple times to quantify training stability
    ##### parameters for Dataset() creation #####
    # `batch_size`: number of samples per batch
    "batch_size": 1024,
    ##### selection of lesion/normal vertices for training #####
    # `contra`: where to sample normal vertices from.
    # If True, only sample normal vertices from contralateral hemisphere
    # If False, sample normal vertices from both hemispheres
    "contra": False,
    # `boundary`: exclude normal vertices that are close to lesional vertices.
    # Creates an uncertainty boundary around lesions (due to low-quality annotations).
    # Only has an effect is contra is False.
    "boundary": True,
    # `num_per_subject`: number of vertices to randomly select per subject.
    # If None, all vertices are selected (respecting `equalize` flag).
    "num_per_subject": 2000,
    # `equalize`: sample the same amount of lesional/normal vertices for each subject.
    # If True, subsample normal vertices to have equal numbers of lesional and non-lesional examples.
    # If `num_per_subject` is defined, will select num_per_subject lesional and num_per_subject non-lesional
    # vertices (total 2*num_per_subject), oversampling lesional vertices if necessary.
    # `equalize_factor`: determines the factor between lesional and normal vertices. A factor of 1 results
    # a balanced dataset. A factor of 2 results in a dataset with twice as many normal vertices than lesional vertices.
    "equalize": True,
    "equalize_factor": 1,
    # `active_selection`: dynamic selection of normal vertices according to how well the model performs
    # (select more vertices that are harder to classify).
    # `active_selection_pool_factor`: how many more normal vertices should be loaded (to select training vertices from)?
    # `active_selection_frac`: fraction of vertices that should be selected according to the models performance. The remainder is selected randomly.
    # active_selection_frac=0 means random resampling of vertices from the pool of vertices.
    "active_selection": True,
    "active_selection_pool_factor": 5,
    "active_selection_frac": 0.0,
    ##### behavior during training #####
    # `resample_each_epoch`: regenerate training set each epoch (to randomly select other normal samples)
    "resample_each_epoch": False,
    # `shuffle_each_epoch`: shuffle the dataset each epoch
    "shuffle_each_epoch": True,
}

##### default network paramaters #####
network_parameters = {
    ##### network architecture #####
    # `network_type`: which network class should be used to create the model? Valid inputs are: 'dense' (for LesionClassifier) and 'neighbour' (for NeighbourLesionClassifier)
    "network_type": "dense",
    "layer_sizes": [40, 10],
    "dropout": 0.4,
    # mc dropout - montecarlo dropout.
    # setting>0 means dropout on predict.
    # If >0, automatically predict averages 50 predictions.
    "mc_dropout": 0,
    # specific parameters for network_type == 'neighbour'
    "shared_layers": [],
    "combine_vertices": "concat",
    ##### training hyper-params #####
    "learning_rate": 0.0001,
    "max_patience": 10,
    "num_epochs": 100,
    # `loss`: loss function used for training. Valid inputs are 'binary_crossentropy', 'soft_f1', and 'focal_loss'
    "loss": "focal_loss",
    # parameters for focal loss: `focal_loss_alpha` and `focal_loss_gamma`
    "focal_loss_alpha": 0.2,
    "focal_loss_gamma": 5,
    # `weighting`: class weighting, valid inputs are 'medial' or 'mean'
    "weighting": None,
    # `optimal_threshold`: determine best threshold based on cluster overlaps with true lesions after post-processing
    # if setting a pre-determined threshold here, run.py needs to be run with --no-optimise-threshold
    "optimal_threshold": 0.5,
    # `date`: current date, appended to experiment folder name to allow training of the same configuration several times
    # change this to test a network from a different date
    "date": datetime.datetime.now().strftime("%y-%m-%d"),
}
