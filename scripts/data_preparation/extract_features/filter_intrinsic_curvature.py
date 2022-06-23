##############################################################################

# This script calculated the absolute (modulus) intrinsic curvature and filter it

#import relevant packages
import numpy as np
import nibabel as nb
import argparse
import io_meld as io
import os

#parse commandline arguments pointing to subject_dir etc
parser = argparse.ArgumentParser(description='filter intrinsic curvature')
parser.add_argument('input', type=str,
                    help='path to the file')
parser.add_argument('output',
                    type=str,
                    help='path to the output file')

args = parser.parse_args()
file= str(args.input)

# if not os.path.isfile(file):
demo = nb.load(file)
curvature=io.load_mgh(file)
curvature=np.absolute(curvature)
curvature = np.clip(curvature, 0, 20)
io.save_mgh(args.output,curvature,demo)


