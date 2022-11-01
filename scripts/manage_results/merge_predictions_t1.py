import os
import numpy as np
import nibabel as nb
import shutil
import glob
import argparse
from os.path import join as opj
from meld_classifier.tools_commands_prints import get_m, run_command

def merge_predictions_t1(subject_ids, subjects_dir, output_dir, verbose=False):
    ''' fusion predictions and t1
    inputs:
        subject_ids :  subjects ID in an array
        subjects_dir :  freesurfer subjects directory 
        output_dir :  directory to save final prediction in native space
    '''

    for subject_id in subject_ids:
        
        # initialise
        t1_file=glob.glob(opj(subjects_dir, subject_id, 'T1', 'T1*.nii*'))[0]
        output_dir_subj = opj(output_dir, subject_id, 'predictions')
        prediction_file=opj(output_dir_subj, 'prediction.nii')

        if (os.path.isfile(t1_file)) and (os.path.isfile(prediction_file)):

            #find min max labels
            numlabels = [nb.load(prediction_file).get_fdata().min(), nb.load(prediction_file).get_fdata().max()]
            print(get_m(f"Number of clusters: {numlabels}", subject_id, "INFO"))

            #find intensity threshold T1
            t1_threshold = np.percentile(nb.load(t1_file).get_fdata(), 99)
            print(get_m(f"T1 thresholded at intensity {t1_threshold}", subject_id, "INFO"))

            #binarise predictions >0 to get mask of labels
            command = f'fslmaths {prediction_file} -bin {output_dir_subj}/labelmask.nii.gz'
            print(command)
            proc = run_command(command, verbose=verbose)

            # multiply prediction by -1 and add 1, then get inverse of label mask
            command = f'fslmaths {output_dir_subj}/labelmask.nii.gz -mul -1 -add 1 -bin {output_dir_subj}/labelmask_inv.nii.gz '
            proc = run_command(command, verbose=verbose)

            # Threshold T1 with the intensity and multiply by the labelmask inversed
            command = f'fslmaths {t1_file} -thr 0 -div {t1_threshold} -mul {output_dir_subj}/labelmask_inv.nii.gz {output_dir_subj}/t1_Rch.nii.gz'
            proc = run_command(command, verbose=verbose)

            # Threshold image at percentile 0.999
            command = f'fslmaths {output_dir_subj}/t1_Rch.nii.gz -uthr 0.999 {output_dir_subj}/tmp1.nii.gz'
            proc = run_command(command, verbose=verbose)

            # Threshold image at 1 and binarise
            command = f'fslmaths {output_dir_subj}/t1_Rch.nii.gz -thr 1 -bin {output_dir_subj}/tmp2.nii.gz'
            proc = run_command(command, verbose=verbose)

            # Add images together and create copy for RGB images
            command = f'fslmaths  {output_dir_subj}/tmp1.nii.gz -add  {output_dir_subj}/tmp2.nii.gz  {output_dir_subj}/t1_Rch.nii.gz'
            proc = run_command(command, verbose=verbose)

            shutil.copy(f'{output_dir_subj}/t1_Rch.nii.gz', f'{output_dir_subj}/t1_Gch.nii.gz')
            shutil.copy(f'{output_dir_subj}/t1_Rch.nii.gz', f'{output_dir_subj}/t1_Bch.nii.gz')

            #colors RGB per label
            labelsR=[1.0000000e+00,   1.0000000e+00,   0.0000000e+00,   0.0000000e+00,   5.8039216e-01,   1.0000000e+00,   1.0000000e+00,   0.0000000e+00,   0.0000000e+00,   4.1568627e-01,   9.4117647e-01,   7.2156863e-01,   3.9215686e-01,   4.0000000e-01,   2.9411765e-01,   9.8039216e-01,   9.4117647e-01,   6.9019608e-01,   5.0196078e-01,   8.6666667e-01,   1.0000000e+00,   1.0000000e+00,   9.4117647e-01,   5.9607843e-01,   1.0000000e+00]
            labelsG=[0.0000000e+00,   8.4313725e-01,   0.0000000e+00,  5.0196078e-01,   0.0000000e+00,   0.0000000e+00,   6.4705882e-01,   1.0000000e+00,   1.0000000e+00,   3.5294118e-01,   5.0196078e-01,   5.2549020e-01,   5.8431373e-01,   8.0392157e-01,   0.0000000e+00,   5.0196078e-01,   9.0196078e-01,   8.7843137e-01,   5.0196078e-01,   6.2745098e-01,   4.9803922e-01,   9.8039216e-01,   1.0000000e+00,   9.8431373e-01,   7.5294118e-01]
            labelsB=[0.0000000e+00,   0.0000000e+00,  1.0000000e+00,   0.0000000e+00,   8.2745098e-01,   1.0000000e+00,   0.0000000e+00,   1.0000000e+00,   0.0000000e+00,   8.0392157e-01,   5.0196078e-01,   4.3137250e-02,   9.2941176e-01,   6.6666667e-01,   5.0980392e-01,   4.4705882e-01,   5.4901961e-01,   9.0196078e-01,   0.0000000e+00,   8.6666667e-01,   3.1372549e-01,  8.0392157e-01,   1.0000000e+00,   5.9607843e-01,   7.9607843e-01]

            # for each label add the colored label mask on top of T1
            for label in range(1, int(numlabels[1])):
                print(get_m(f"Add cluster {label} to T1", subject_id, "INFO"))
                
                command = f'fslmaths {prediction_file} -thr {label} -uthr {label} -bin -mul {labelsR[label-1]} -add {output_dir_subj}/t1_Rch.nii.gz {output_dir_subj}/t1_Rch.nii.gz'
                proc = run_command(command, verbose=verbose)

                command = f'fslmaths {prediction_file} -thr {label} -uthr {label} -bin -mul {labelsG[label-1]} -add {output_dir_subj}/t1_Gch.nii.gz {output_dir_subj}/t1_Gch.nii.gz'
                proc = run_command(command, verbose=verbose)

                command = f'fslmaths {prediction_file} -thr {label} -uthr {label} -bin -mul {labelsB[label-1]} -add {output_dir_subj}/t1_Bch.nii.gz {output_dir_subj}/t1_Bch.nii.gz'
                proc = run_command(command, verbose=verbose)

            #merge RGB images in one 
            command = f'fslmerge -t {output_dir_subj}/predictions_merged_t1.nii.gz {output_dir_subj}/t1_Rch.nii.gz {output_dir_subj}/t1_Gch.nii.gz {output_dir_subj}/t1_Bch.nii.gz'
            proc = run_command(command, verbose=verbose) 

            #delete temporary files
            os.remove(f'{output_dir_subj}/t1_Rch.nii.gz')
            os.remove(f'{output_dir_subj}/t1_Gch.nii.gz')
            os.remove(f'{output_dir_subj}/t1_Bch.nii.gz')
            os.remove(f'{output_dir_subj}/tmp2.nii.gz')
            os.remove(f'{output_dir_subj}/tmp1.nii.gz')
            os.remove(f'{output_dir_subj}/labelmask_inv.nii.gz')
            os.remove(f'{output_dir_subj}/labelmask.nii.gz')
        
        else:
            print(get_m(f"Could not find T1 or predictions files. Skip merging T1 and predictions", subject_id, "WARNING"))

if __name__ == "__main__":
    #parse commandline arguments pointing to subject_dir etc
    parser = argparse.ArgumentParser(description='merge T1 and predictions')
    parser.add_argument('-subject_ids', type=str,
                        help='subject_ids')
    parser.add_argument('-subjects_dir', type=str,
                        help='directory with t1 and predictions input')
    parser.add_argument('-output_dir', type=str,
                        help='directory to save outputs')
    args = parser.parse_args()
    
    subjects_dir=args.subjects_dir
    output_dir=args.output_dir
   
    
    try:
        subject_ids = np.loadtxt(args.subject_ids, dtype="str", ndmin=1)
    except:
        subject_ids = [args.subject_ids]


    merge_predictions_t1(subject_ids, subjects_dir, output_dir, verbose=False)



