# Predict lesion on a new patient 

With the MELD classifier pipeline, if you are from an epilepsy centre who's data was used to train the classifier, predicting lesion locations on new patients is easy. In the following, we describe the steps needed for predict the trained model on a new patient. 
Note: No demographic information are required for this process.

If you would like to predict lesions on patients from new epilepsy centres or new MRI scanners or updated T1 / FLAIR sequences that were not used to train the classifier, you will need to used the Predict_on_patients_from_new_sites pipeline which is under development.
Note: Demographic information (e.g age and sex) will be required for this process.

## Installation
- Before being able to use the classifier on your data, some paths need to be set up and the pretrained model needs to be downloaded. 
- To set up the paths open the meld_config.ini.example and add the path to your meld data e.g. 

## Information about the pipeline
- The pipeline is split into 3 main scripts (detailed below). 

### First step 
Go into the meld_classifier folder 
```bash
  cd <path_to_meld_classifier_folder>
```
Each of the 3 following scripts needs to be run from the 'meld_classifier' folder

### Script 1 - FreeSurfer reconstruction
```bash
python scripts/new_patient_pipeline/new_pt_pipeline_script1.py -id <sub_id>
```
- This script runs a FreeSurfer reconstruction on a participant
- REMINDER: you need to have set up your paths & organised your data before running Script 1 (see Installation)
- Within your  MELD folder should be an input folder that contains folders for each participant. 
- Within each participant folder should be a T1 folder that contains the T1 in nifti format ".nii" and where available a FLAIR folder that contains the FLAIR in nifti format ".nii"

### Script 2 - Feature Preprocessing
```bash
python scripts/new_patient_pipeline/new_pt_pipeline_script2.py -ids <text_file_with_subjects_ids> - site <site_code>
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

### Script 3 - Lesions prediction & MELD reports
```bash
python scripts/new_patient_pipeline/new_pt_pipeline_script3.py -ids <text_file_with_subjects_ids> - site <site_code>
```
- The site code should start with H, e.g. H1. If you cannot remember your site code - contact the MELD team.
- Features need to have been processed using script 2 and Freesurfer outputs need to be available for each subject
- This script : 
1. Run the MELD classifier and predict lesion on new subject
2. Register the prediction back into the native nifti MRI. Results are stored in inputs/<sub_id>/predictions.
3. Create MELD reports with predicted lesion location on inflated brain, on native MRI and associated saliencies. Reports are stored in Results are stored in inputs/<sub_id>/predictions/reports.

