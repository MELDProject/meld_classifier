# Training and evaluating models

With the MELD classifier pipeline, training and evaluating several models at the same time (e.g. for grid search) is easy. In the following, we describe the steps needed for creating the trained models released with this package.  Beware, that this will require advanced programming skills and knowledge in training deep neural networks.

## Prerequisites
- ensure that you have [installed MELD classifier](README.md#installation).
- create a `MELD_dataset_XX.csv` file which contains the train/test split. E.g., run `scripts/classifier/create_trainval_test_split.py`

## General usage
```
scripts/run.py
``` 
This is the main entry point for training and evaluating models. 
- Network and dataset configuration parameters are passed to the script using a config file. 
    - `scripts/experiment_config_template.py` is an exemplary config file with all possible parameters. 
    - The experiment configurations used to create the released models can be found in `scripts/experiment_config_train.py` and `scripts/experiment_config_test.py` (for training on the reverse cohort).
- For training + evaluation (with thresold optimization), call `python classifier/run.py all --config experiment_config.py`

### Training models
```
python run.py train --config experiment_config.py
```
- Train models based on parameters in experiment_config.py and optimize the threshold after training on the training set 
- optional flag `--no-optimise-threshold` turns optimizing the threshold off (allows using a pre-defined threshold)

### Evaluating models (on validation set)
```
python classifier/run.py eval --config experiment_config.py
```
- Evaluate models based on parameters in experiment_config.py, do not plot images
- optional flag `--make-images` turns on plotting results on brain surface

### Ensemble models
```
python classifier/ensemble.py --exp experiment_folder --ensemble-folds
```
- Create and evaluate an ensemble model from all individual folds for a more robust model

### Predict
```
python classifier/test_and_stats.py --experiment-folder experiment_folder --experiment-name experiment_name
```
- Predict final model on test set, plot predictions on flat maps and calculate saliency.
- To parallelise prediction, use the `--n-splits`, `--run-on-slurm` and `--save-per-split-files` arguments. After completion, merge results files with `scripts/classifier/merge_prediction_files.py`

### HPC
```
python classifier/run.py all --config experiment_config.py --run-on-slurm
```
- Set models up as individual slurm jobs using a batch array. To be used on a slurm cluster only.
    * adapt slurm scripts in [scripts/hpc](scripts/hpc) to your system's needs
- in case models failed to run, you can resubmit the failed jobs using `scripts/classifier/rerun.py`

## Compare models
```
python classifier/experiment_evaluation.py --exp experiment_folder
```
- Compare trained experiments by testing whether there are significant differences in performance.
