##############################################################################

##This script needs to be run on all patients and all controls
# It registers the participant's data to the  bilaterally symmetrical template
SUBJECTS_DIR=$1
subject_list=$2

cd "$SUBJECTS_DIR"
export SUBJECTS_DIR="$SUBJECTS_DIR"

if [ ! -e fsaverage_sym ]
then
cp -r $FREESURFER_HOME/subjects/fsaverage_sym ./
fi

## Import list of subjects
subjects=$(<"$subject_list")
# subjects=$subject_list
# for each subject do the following
for s in $subjects
do
  if [ ! -e "$s"/xhemi ]
  then
  surfreg --s "$s" --t fsaverage_sym --lh
  surfreg --s "$s" --t fsaverage_sym --lh --xhemi
  mkdir "$s"/xhemi/surf_meld
  fi
  if [ ! -e "$s"/xhemi/surf_meld ]
  then
  mkdir "$s"/xhemi/surf_meld
  fi

done

