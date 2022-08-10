#!/bin/bash

### Example of pipeline to run prediction on 1 subject or multiple subjects. 
## Note: This pipeline considere that you already have harmonised your data for your site. If you did not, refer to example_pipeline_harmonisation.sh
## This script will :
## 1. Segment the brain and extract surface-based features using freesurfer (optional: you can uncomment --fastsurfer to use fastsurfer instead) 
## 2. Preprocess the surface-based features (combat harmonise and inter-intra normalisation)
## 3. Predict using MELD classifier and create individual pdf report

##--------------------------------------------
### INITIALISATION

## Provide your site code
site_code="H4"

## Provide the subject_id of the subject you want to process, or the file containing a list of ids. 
## If providing 1 single id, list_ids should be left blank : "''"
## If providing a list of ids,  subject_id should be left blank : "''"
subject_id="MELD_H4_3T_FCD_0001"
list_ids="''"   

## Do not change here
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

### Command to call script to preprocess data
## Note: If you did not harmonised your data before, you will need to run example_pipeline_harmonisation.sh
echo 'CALL PREPROCESSING/HARMONISATION PIPELINE'
python $DIR/run_script_preprocessing.py \
                            -site $site_code \
                            -id $subject_id \
                            -ids $list_ids \

### Command to call script to predict lesions
## Note: Uncomment "--no_prediction_nifti" if you don't need the prediction back into native volume, nor the report
## Note: Uncomment "--noreport" if you don't need the pdf report
echo 'CALL PREDICTION PIPELINE'
python $DIR/run_script_prediction.py \
                            -site $site_code \
                            -id $subject_id \
                            -ids $list_ids \
                            # --no_prediction_nifti \
                            # --noreport

                                


