########### master script for running the meld pipeline. this script assumes freesurfer outputs have already been created ########

## run "bash meld_pipeline.sh <fs_subjects_dir> <site_code> <text_file_with_subject_ids> <path_to_scripts> <path_to_save_outputs>

subjects_dir=$1
site_code=$2
subject_ids=$3
script_dir=$4
output_dir=$5

#register to symmetric fsaverage xhemi
echo "Creating registration to template surface"
bash "$script_dir"/create_xhemi.sh "$subjects_dir" "$subject_ids"

#create basic features
echo "Sampling features in native space"
bash "$script_dir"/sample_FLAIR_smooth_features.sh "$subjects_dir" "$subject_ids" "$script_dir"

#move features and lesions to template
echo "Moving features to template surface"
bash "$script_dir"/move_to_xhemi_flip.sh "$subjects_dir" "$subject_ids"
echo "Moving lesion masks to template surface"
bash "$script_dir"/lesion_labels.sh "$subjects_dir" "$subject_ids"

#create training_data matrix for all patients and controls.
echo "creating final training data matrix"
python "$script_dir"/create_training_data_hdf5.py "$subjects_dir" "$subject_ids" "$output_dir"
