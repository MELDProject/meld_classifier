# Predict lesion on a new patient 

With the MELD classifier pipeline, if you are from an epilepsy centre who's data was used to train the classifier, predicting lesion locations on new patients is easy. In the following, we describe the steps needed for predict the trained model on a new patient. 
Note: No demographic information are required for this process.

If you would like to predict lesions on patients from new epilepsy centres or new MRI scanners or updated T1 / FLAIR sequences that were not used to train the classifier, you will need to used the [*new_site_pipeline*](#predict-lesion-on-a-patient-from-a-new-site) which is under development.
Note: Demographic information (e.g age and sex) will be required for this process.

Before running the below pipeline, ensure that you have [installed MELD classifier](README.md#installation).
The pipeline is split into 3 main scripts (detailed below) and can be found in `scripts/new_patient_pipeline`. 

## FreeSurfer reconstruction
```bash
python new_pt_pipeline_script1.py -id <sub_id>
```
- This script runs a FreeSurfer reconstruction on a participant
- REMINDER: you need to have set up your paths & organised your data before running Script 1 (see [Installation](README.md#-Installation))
- Within your MELD data folder should be an input folder that contains folders for each participant. 
- Within each participant folder should be a T1 folder that contains the T1 in nifti format ".nii" and where available a FLAIR folder that contains the FLAIR in nifti format ".nii"

## Feature Preprocessing
```bash
python new_pt_pipeline_script2.py -ids <text_file_with_subjects_ids> - site <site_code>
```
- The site code should start with H, e.g. H1. If you cannot remember your site code - contact the MELD team.
- This script:
    1. Extracts surface-based features needed for the classifier :
        * Samples the features
        * Creates the registration to the template surface fsaverage_sym
        * Moves the features to the template surface
        * Write feature in hdf5
    2. Preprocess features : 
        * Smooth features and write in hdf5
        * Combat harmonised and write in hdf5
        * Normalise the smoothed features (intra-subject & inter-subject (by controls)) and write in hdf5
        * Normalise the raw combat features (intra-subject, asymmetry and then inter-subject (by controls)) and write in hdf5

## Lesions prediction & MELD reports
```bash
python new_pt_pipeline_script3.py -ids <text_file_with_subjects_ids> - site <site_code>
```
- The site code should start with H, e.g. H1. If you cannot remember your site code - contact the MELD team.
- Features need to have been processed using script 2 and Freesurfer outputs need to be available for each subject
- This script : 
    1. Run the MELD classifier and predict lesion on new subject
    2. Register the prediction back into the native nifti MRI. Results are stored in `inputs/<sub_id>/predictions`.
    3. Create MELD reports with predicted lesion location on inflated brain, on native MRI and associated saliencies. Reports are stored in Results are stored in `inputs/<sub_id>/predictions/reports`.

# Predict lesion on a patient from a new site

COMING SOON

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