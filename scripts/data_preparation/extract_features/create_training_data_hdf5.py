##############################################################################

# This script writes out the per-vertex features and lesion classification for each patient and control to a .hdf5 file. 

#import relevant packages
import numpy as np
import nibabel as nb
import argparse
import io_meld as io
import h5py
import os

#parse commandline arguments pointing to subject_dir etc
parser = argparse.ArgumentParser(description='create feature matrix for all subjects')
parser.add_argument('subject_dir', type=str,
                    help='path to subject dir')
parser.add_argument('subject_ids',
                    type=str,
                    help='textfile containing list of subject ids')
parser.add_argument('output_dir',
                    type=str,
                    help='textfile containing list of subject ids')

args = parser.parse_args()


#save subjects dir and subject ids. import the text file containing subject ids
subject_dir=args.subject_dir
subject_ids=args.subject_ids
output_dir=args.output_dir
subject_ids=np.array(np.loadtxt(subject_ids, dtype='str', ndmin=1))
# subject_ids=np.array(subject_ids.split(' '))


#list features
features = np.array(['.on_lh.thickness.mgh', '.on_lh.w-g.pct.mgh', '.on_lh.curv.mgh','.on_lh.sulc.mgh',
    '.on_lh.gm_FLAIR_0.75.mgh', '.on_lh.gm_FLAIR_0.5.mgh', '.on_lh.gm_FLAIR_0.25.mgh',
    '.on_lh.gm_FLAIR_0.mgh', '.on_lh.wm_FLAIR_0.5.mgh', '.on_lh.wm_FLAIR_1.mgh',
    '.on_lh.pial.K_filtered.sm20.mgh'])
n_vert=163842
cortex_label=nb.freesurfer.io.read_label(os.path.join(subject_dir,'fsaverage_sym/label/lh.cortex.label'))
medial_wall = np.delete(np.arange(n_vert),cortex_label)



for subject in subject_ids:
    print("saving subject " + subject + "...")
    io.save_subject(subject,features,medial_wall, subject_dir, output_dir)



