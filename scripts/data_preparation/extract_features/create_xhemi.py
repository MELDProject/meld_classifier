### Python functions to registers the participant's data to the  bilaterally symmetrical template

from functools import partial
import os
from os.path import join as opj
import shutil
import subprocess
import subprocess as sub
from argparse import ArgumentParser
import multiprocessing

def create_xhemi(subject_id, subjects_dir,template = 'fsaverage_sym'):
    
    #check FS folder subject exist
    if not os.path.isdir(opj(subjects_dir,subject_id)):
        print(f'ERROR: FS folder does not exist for {subject_id}')
        return

    #copy template
    if not os.path.isdir(opj(subjects_dir,template)):
        shutil.copy(opj(os.environ['FREESURFER_HOME'],'subjects',template), opj(subjects_dir, os.path.basename(template)))

    if not os.path.isfile(opj(subjects_dir,subject_id,'surf','lh.fsaverage_sym.sphere.reg')):
        command = f'SUBJECTS_DIR={subjects_dir} surfreg --s {subject_id} --t {template} --lh'    
        proc = sub.Popen(command, shell=True, stdout = sub.DEVNULL)
        proc.wait()

    if not os.path.isfile(opj(subjects_dir,subject_id,'xhemi','surf','lh.fsaverage_sym.sphere.reg')):
        command = f'SUBJECTS_DIR={subjects_dir} surfreg --s {subject_id} --t {template} --lh --xhemi'
        proc = sub.Popen(command, shell=True, stdout = sub.DEVNULL)
        proc.wait()

            

def run_parallel_xhemi(subject_ids, subjects_dir, num_procs = 10,template = 'fsaverage_sym' ):
    pass
    with multiprocessing.Pool(num_procs) as p:
        for _ in p.imap_unordered(partial(create_xhemi, subjects_dir=subjects_dir, template=template), subject_ids):
            pass

if __name__ == '__main__':
    parser = ArgumentParser(description="register the participant's data to the  bilaterally symmetrical template")
    #TODO think about how to best pass a list
    parser.add_argument('-ids','--list_ids',
                        help='Subjects IDs in a text file',
                        required=True,)
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
        run_parallel_xhemi([args.id], args.subjects_dir, num_procs=args.num_procs )
    else:
        create_xhemi([args.id], args.subjects_dir)