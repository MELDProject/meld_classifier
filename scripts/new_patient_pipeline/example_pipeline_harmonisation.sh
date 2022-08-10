#!/bin/bash

### Example of pipeline to harmonise your data for a new site
## Note: This needs to be done only once, prior to do any prediction on new subjects. 
## Note: You will need to provide at least 30 subjects (can be patients or controls) for best perfomances. 
## Note: You will need to provide a demographic_file.csv containing demographic informations for these subjects

### This script will :
## 1. Segment the brain and extract surface-based features using freesurfer (optional: you can uncomment --fastsurfer to use fastsurfer instead) 
## 2. Compute the harmonisation parameters for your site 
## 3. (OPTIONAL) Predict using MELD classifier and create individual pdf report

##--------------------------------------------
### INITIALISATION

##Provide your site code
site_code="H4"

##Provide the subject_id of the subject you want to process, or the file containing a list of ids. 
##If providing 1 single id, list_ids should be left blank : "''"
##If providing a list of ids,  subject_id should be left blank : "''"
subject_id="''"
list_ids="list_subjects.txt"

##Provide the path to the demographic file. Needs to be a csv and contain the columns "ID","Sex" and "Age at preoperative" 
demographic_file="demographics_file.csv"

##Do not change here
DIR="$(dirname "$(realpath "$0")")"
cd $DIR

##------------------------------------------------------------------------
### COMMANDS

### Command to call script to segmentation brain and extract features
## Note: Uncomment "--fastsurfer" if you want to use instead of freesurfer
echo 'CALL SEGMENTATION PIPELINE'
python $DIR/run_script_segmentation.py \
                                -site $site_code \
                                -id $subject_id \
                                -ids $list_ids \
                                --parallelise \
                                #--fastsurfer 

### Command to call script to harmonise your data
## Note: If you wish to also predict on the subjects used for harmonisation, you can comment the "--harmo_only" and uncomment the next command for prediction
echo 'CALL PREPROCESSING/HARMONISATION PIPELINE'
python $DIR/run_script_preprocessing.py \
                            -site $site_code \
                            -id $subject_id \
                            -ids $list_ids \
                            -demos $demographic_file \
                            --harmo_only


### Command to call script to predict lesions 
## Note: Uncomment the whole command if you want to enable the prediction on the subject you are using for harmonisation
## Note: Uncomment "--no_prediction_nifti" if you don't need the prediction back into native volume, nor the report
## Note: Uncomment "--noreport" if you don't need the pdf report
# echo 'CALL PREDICTION PIPELINE'
# python $DIR/run_script_prediction.py \
#                             -site $site_code \
#                             -id $subject_id \
#                             -ids $list_ids \
#                             # --no_prediction_nifti \
#                             # --noreport

                                


