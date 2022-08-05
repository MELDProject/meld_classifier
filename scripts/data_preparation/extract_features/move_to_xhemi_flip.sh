#############################################
# This script Moves features to fsaverage_sym - a bilaterally symmetrical template
# It also moves the manual lesion label to fsaverage_sym
##Run on all patients and controls

## Change to your subjects directory ##
subject_dir=$1
subject_list=$2

cd "$subject_dir"
export SUBJECTS_DIR="$subject_dir"



## Import list of subjects
subjects=$(<"$subject_list")
# subjects=$subject_list

Measures="thickness.mgh  w-g.pct.mgh  curv.mgh sulc.mgh 
    gm_FLAIR_0.75.mgh  gm_FLAIR_0.5.mgh  gm_FLAIR_0.25.mgh
    gm_FLAIR_0.mgh  wm_FLAIR_0.5.mgh  wm_FLAIR_1.mgh 
    pial.K_filtered.sm20.mgh"

for s in $subjects
do
  #check if final file exists. if it does, skip this step
  if [ ! -e "$s"/xhemi/surf_meld/zeros.mgh ];  then
  #create one all zero overlay for inversion step
  cp fsaverage_sym/surf/lh.white.avg.area.mgh "$s"/xhemi/surf_meld/zeros.mgh
  mris_calc --output "$s"/xhemi/surf_meld/zeros.mgh "$s"/xhemi/surf_meld/zeros.mgh set 0
  fi 

  for m2 in $Measures
  do
    #check if final file exists. if it does, skip this step
    if [ ! -e "$s"/xhemi/surf_meld/lh.on_lh."$m2" ];  then
    # Move onto left hemisphere
    mris_apply_reg --src  "$s"/surf_meld/lh."$m2" --trg "$s"/xhemi/surf_meld/lh.on_lh."$m2"  --streg $SUBJECTS_DIR/"$s"/surf/lh.sphere.reg     $SUBJECTS_DIR/fsaverage_sym/surf/lh.sphere.reg
    fi
    #check if final file exists. if it does, skip this step
    if [ ! -e "$s"/xhemi/surf_meld/rh.on_lh."$m2" ];  then
    mris_apply_reg --src "$s"/surf_meld/rh."$m2" --trg "$s"/xhemi/surf_meld/rh.on_lh."$m2"    --streg $SUBJECTS_DIR/"$s"/xhemi/surf/lh.fsaverage_sym.sphere.reg     $SUBJECTS_DIR/fsaverage_sym/surf/lh.sphere.reg
    fi

  done

done
