##############################################################################

# This script calculated the absolute (modulus) intrinsic curvature 

#import relevant packages
import numpy as np
import nibabel as nb
import argparse
import io_meld as io
import os

#parse commandline arguments pointing to subject_dir etc
parser = argparse.ArgumentParser(description='filter intrinsic curvature')
parser.add_argument('subject_dir', type=str,
                    help='path to subject dir')
parser.add_argument('subject_ids',
                    type=str,
                    help='text file with ids')

args = parser.parse_args()


#save subjects dir and subject ids. import the text file containing subject ids
subject_dir=args.subject_dir
subject_ids=str(args.subject_ids)
print(subject_ids)
subject_ids=np.array(np.loadtxt(subject_ids, dtype='str',ndmin=1))
print(subject_ids)
# subject_ids=np.array(subject_ids.split(' '))

hemis=['lh','rh']

for h in hemis:
    for s in subject_ids:
        if not os.path.isfile(os.path.join(subject_dir, s,'xhemi/surf_meld', h+'.pial.K_filtered.mgh')):
            demo = nb.load(os.path.join(subject_dir, s, 'surf_meld', h + '.pial.K.mgh'))
            curvature=io.load_mgh(os.path.join(subject_dir, s, 'surf_meld', h + '.pial.K.mgh'))
            curvature=np.absolute(curvature)
            io.save_mgh(os.path.join(subject_dir, s, 'surf_meld', h+'.pial.K_filtered.mgh'),curvature,demo)


