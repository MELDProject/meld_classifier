# ##############################################################################

# ##This script needs to be run on all patients and all controls
# # It registers the participant's data to the  bilaterally symmetrical template
# SUBJECTS_DIR=$1
# subject_list=$2

# cd "$SUBJECTS_DIR"
# export SUBJECTS_DIR="$SUBJECTS_DIR"

# if [ ! -e fsaverage_sym ]
# then
# cp -r $FREESURFER_HOME/subjects/fsaverage_sym ./
# fi

# ## Import list of subjects
# subjects=$(<"$subject_list")
# # subjects=$subject_list
# # for each subject do the following
# for s in $subjects
# do
#   if [ ! -e "$s"xhemi/surf/lh.fsaverage_sym.sphere.reg ]
#   then
#   surfreg --s "$s" --t fsaverage_sym --lh
#   surfreg --s "$s" --t fsaverage_sym --lh --xhemi
#   mkdir "$s"/xhemi/surf_meld
#   fi
#   if [ ! -e "$s"/xhemi/surf_meld ]
#   then
#   mkdir "$s"/xhemi/surf_meld
#   fi

# done

#python version of create xhemi
from functools import partial
import os
from os.path import join as opj
import shutil
import subprocess
import subprocess as sub
from argparse import ArgumentParser
import multiprocessing
def create_xhemi(subject_id, subjects_dir,template = 'fsaverage_sym'):
    #copy template
    if not os.path.isdir(opj(subjects_dir,template)):
        shutil.copy(opj(os.environ['FREESURFER_HOME'],'subjects',template), opj(subjects_dir, os.path.basename(template)))
    
        # if not os.path.isfile(opj(subjects_dir,subject,'xhemi','surf','lh.fsaverage_sym.sphere.reg')):

    if not os.path.isfile(opj(subjects_dir,subject_id,'surf','lh.fsaverage_sym.sphere.reg')):
        command = f'SUBJECTS_DIR={subjects_dir} surfreg --s {subject_id} --t {template} --lh'    
        try:
            proc = sub.check_call(command, shell=True, stdout = sub.DEVNULL)
        except sub.CalledProcessError as e:
            print(f'ERROR: create xhemi has failed for {subject_id}')
            print(e)
            return 
        proc.wait()

    if not os.path.isfile(opj(subjects_dir,subject_id,'xhemi','surf','lh.fsaverage_sym.sphere.reg')):
        command = f'SUBJECTS_DIR={subjects_dir} surfreg --s {subject_id} --t {template} --lh --xhemi'
        try:
            proc = sub.check_call(command, shell=True, stdout = sub.DEVNULL)
        except sub.CalledProcessError as e:
            print(f'ERROR: create xhemi has failed for {subject_id}')
            print(e)
            return 
        proc.wait()

            

def run_parallel(subject_ids, subjects_dir, num_procs = 10,template = 'fsaverage_sym' ):
    pass
    with multiprocessing.Pool(num_procs) as p:
        for _ in p.imap_unordered(partial(create_xhemi, subjects_dir=subjects_dir, template=template), subject_ids):
            pass

if __name__ == '__main__':
    parser = ArgumentParser(description="register the participant's data to the  bilaterally symmetrical template")
    #TODO think about how to best pass a list
    # parser.add_argument('-ids','--list_ids',
    #                     help='Subjects IDs in a text file',
    #                     required=True,)
    parser.add_argument('-id','--id',
                        help='Subjects ID',
                        required=True,)
    #TODO make this freesurfer default if not provided
    parser.add_argument('-sd','--subjects_dir',
                        help='Subjects directory...',
                        required=True,)
    parser.add_argument('-np','--num_procs',
                        help='Number of processes for parallel processing.',
                        default=1,
                        required=False,)
    args = parser.parse_args()
    if args.num_procs > 1:
        run_parallel([args.id], args.subjects_dir, num_procs=args.num_procs )
    else:
        create_xhemi([args.id], args.subjects_dir)