SUBJECT_DIR=$1  # freesurfer output directory
subject_list=$2 # text file with ids
OUTPUT_DIR=$3   # folder to store final nifti files


cd "$SUBJECT_DIR"
export SUBJECTS_DIR="$SUBJECT_DIR"


## Import list of subjects
subjects=$(<"$subject_list")
echo $subjects

for sub in $subjects
    do
    m='prediction'
    # Moves left hemi from fsaverage to native space
    # --src is the source image i.e. the map you want to move back so change to the name of the cluster map in fsaverage_sym that you want to move back
    # --trg is the target image i.e. the name of the map you want to create in the subject's native space
    # the rest is the registration files
    mris_apply_reg --src "$sub"/xhemi/classifier/lh."$m".mgh --trg "$sub"/surf/lh."$m".mgh \
    --streg $SUBJECTS_DIR/fsaverage_sym/surf/lh.sphere.reg $SUBJECTS_DIR/"$sub"/surf/lh.sphere.reg
    # Moves the right hemi back from fsaverage to native. There are 2 steps
    #Step1: move left hemi fsaverage to right hemi of fsaverage
    # --src is the source image i.e. the map you want to move back so change to the name of the cluster map in fsaverage_sym that you want to move back. should be rh....
    # --trg is the target image i.e. the name of the map on the rh -  i called these rh.{name of file}_on_rh.mgh - these are still in template space (fsaverage)
    # Step 2: move from rh of fsaverage to native space
    # --src is the source image i.e. the name of the file you created in step1
    # --trg is the target image i.e. the name of the map you want to create in the subject's native space

    mris_apply_reg --src "$sub"/xhemi/classifier/rh."$m".mgh --trg "$sub"/xhemi/classifier/rh."$m"_on_rh.mgh  \
    --streg $SUBJECTS_DIR/fsaverage_sym/surf/lh.sphere.reg $SUBJECTS_DIR/fsaverage_sym/surf/rh.sphere.left_right

    mris_apply_reg --src "$sub"/xhemi/classifier/rh."$m"_on_rh.mgh --trg "$sub"/surf/rh."$m".mgh\
    --streg $SUBJECTS_DIR/fsaverage_sym/surf/rh.sphere.reg $SUBJECTS_DIR/"$sub"/surf/rh.sphere.reg
    

##  11. Convert from .mgh to .nii

    #Map from surface back to vol
    mri_surf2vol --identity "$sub" --template $SUBJECTS_DIR/"$sub"/mri/T1.mgz --o $SUBJECTS_DIR/"$sub"/mri/lh."$m".mgz \
    --hemi lh --surfval "$sub"/surf/lh."$m".mgh --fillribbon --float2int

    mri_surf2vol --identity "$sub" --template $SUBJECTS_DIR/"$sub"/mri/T1.mgz --o $SUBJECTS_DIR/"$sub"/mri/rh."$m".mgz \
    --hemi rh --surfval "$sub"/surf/rh."$m".mgh --fillribbon --float2int
    
    #Register back to original volume
    mri_vol2vol --mov $SUBJECTS_DIR/"$sub"/mri/lh."$m".mgz --targ $SUBJECTS_DIR/"$sub"/mri/orig/001.mgz  --regheader --o $SUBJECTS_DIR/"$sub"/mri/lh."$m".mgz --nearest
    
    mri_vol2vol --mov $SUBJECTS_DIR/"$sub"/mri/rh."$m".mgz --targ $SUBJECTS_DIR/"$sub"/mri/orig/001.mgz  --regheader --o $SUBJECTS_DIR/"$sub"/mri/rh."$m".mgz --nearest
    
    #convert to nifti
    mri_convert "$sub"/mri/lh."$m".mgz "$sub"/mri/lh."$m".nii -rt nearest
    mri_convert "$sub"/mri/rh."$m".mgz "$sub"/mri/rh."$m".nii -rt nearest
    
    #resample size
    

#     #combine vols from left and right hemis
#     fslmaths "$sub"/mri/lh."$m".nii \
#     -add "$sub"/mri/rh."$m".nii \
#     "$sub"/mri/"$m".nii
    
    #move files
    if [ ! -d "$OUTPUT_DIR"/"$sub"/predictions ]
    then
        mkdir "$OUTPUT_DIR"/"$sub"/predictions
    fi
    mv "$sub"/mri/lh."$m".nii "$OUTPUT_DIR"/"$sub"/predictions/lh."$m".nii
    mv "$sub"/mri/rh."$m".nii "$OUTPUT_DIR"/"$sub"/predictions/rh."$m".nii
    
    
    done