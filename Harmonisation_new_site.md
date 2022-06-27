# Compute the harmonisation parameters for a new site/scanner

This guidelines will explain how to compute the harmonisation parameters needed to predict lesions on patients from new epilepsy centres or new MRI scanners or updated T1 / FLAIR sequences that were not used to train the classifier. 

Note: 
- Demographic information (e.g age and sex) will be required for this process.
- If you are from an epilepsy centre who's data was used to train the classifier, you can directly refer to the guidelines from the [Predict_on_new_patient.md](Predict_on_new_patient.md)

In the following, we describe the steps needed to update the classifier for use on new site and to compute the harmonisation parameters. 
Once you have these parameters you can follow the general [guidelines](Predict_on_new_patient.md) to predict on your new subject.

## Disclaimer

The MELD surface-based FCD detection algorithm is intended for research purposes only and has not been reviewed or approved by the Medicines and Healthcare products Regulatory Agency (MHRA), European Medicine Agency (EMA) or by any other agency. Any clinical application of the software is at the sole risk of the party engaged in such application. There is no warranty of any kind that the software will produce useful results in any way. Use of the software is at the recipient's own risk.

## Information about the pipeline
Before running the below pipeline, ensure that you have [installed MELD classifier](README.md#installation).
Also you need to make sure that Freesurfer is activated in your terminal (you should have some printed FREESURFER paths when opening the terminal). Otherwise you will need to manually activate Freesurfer on each new terminal by running : 
```bash
export FREESURFER_HOME=<freesurfer_installation_directory>/freesurfer
source $FREESURFER_HOME/SetUpFreeSurfer.sh
```
with `<freesurfer_installation_directory>` being the path to where your Freesurfer has been installed.

## Guidelines to update the MELD classifier (for pilot sites only - REMOVE AFTER)
You will need to update your MELD code and files.

First, pull the new classifier codes from github on a new branch:  
```bash
cd meld_classifier 
git checkout --track origin/test_new_release 
```
 
Then, you can create a new environment to test the new classifier: 
```bash
# create the environment that will have the name meld_classifier_newrelease 
conda env create -f environment.yml 
# activate  meld_classifier_newrelease environment 
conda activate meld_classifier_newrelease 
# install meld_classifier with pip 
pip install -e . 
```

Finally, you will need to download new files: 
```bash
python scripts/update_meld_classifier.py 
```

## Compute the harmonisation paramaters 

The harmonisation parameters are computed using Combat Distributed [REF ??]
To get these parameters you will need a small cohort of subjects acquired from the same scanner and under the same protocol (sequence, parameters, ...).
Subjects can be controls and/or patients, but we advise to use ***at least 30 subjects*** to enable an accurate harmonisation. 
Make sure the data are quality (i.e no blurring, no artefacts, no cavities in the brain).

### First step - Organising your data!
You will need to organise the MRI data of the subjects

In the 'input' folder where your meld data has / is going to be stored, create a folder for each patient. 

The IDs should follow the structure MELD\_<site\_code>\_<scanner\_field>\_FCD\_000X

e.g.MELD\_H1\_3T\_FCD\_0001 

In each patient folder, create a T1 and FLAIR folder.

Place the T1 nifti file into the T1 folder. Please ensure 'T1' is in the file name.

Place the FLAIR nifti file into the FLAIR folder. Please ensure 'FLAIR' is in the file name.

![example](images/example_folder_structure.png)

You will also need to gather demographic information into a csv file as illustrated in the example below:

![example](images/example_demographic_csv.PNG)
- ID : MELD ID
- Age at preoperative: The age of the subject at the time of the preoperative T1 scan (in years)
- Sex: 1 if male, 0 if female

### Second step : Run 2 scripts to get the harmonisation parameters
You will need to make sure you are in folder containing the MELD classifier scripts
```bash
  cd <path_to_meld_classifier_folder>
```
Each of the 2 following scripts needs to be run from the 'meld_classifier' folder. 

#### Script 1 - FreeSurfer reconstruction and smoothing
```bash
python scripts/new_patient_pipeline/new_pt_pipeline_script1.py -id <sub_id> -site <site_code>
```
- This script runs a FreeSurfer reconstruction on a participant and smooth the data.
- Within your MELD folder should be an input folder that contains folders for each participant. 
- Within each participant folder should be a T1 folder that contains the T1 in nifti format ".nii" and where available a FLAIR folder that contains the FLAIR in nifti format ".nii"

#### Script 2a - Compute the harmonisation parameters for a new site
```bash
python scripts/new_patient_pipeline/new_site_harmonisation_script.py -ids <text_file_with_subjects_ids> -site <site_code> -demos <demographic_file>
```
- This script computes the combat parameters that are needed to harmonise your new site to the MELD cohort.
- The text_file_with_subjects_ids should contain the ids of the subject you want to use to compute the harmonisation parameters as detail in the above section.
- <demographic_file> should be the path to the csv file containing the demographic information as described in the above section.


## What's next ? 
Once you have successfully computed the harmonisation parameters, they should be saved on your meld_data folder.
You can now refer to the guidelines [Predict_on_new_patient.md](Predict_on_new_patient.md) to predict lesion in patients from that new scanner/site.
