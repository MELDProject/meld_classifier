import os
from os.path import join as opj
from subprocess import Popen,  STDOUT, DEVNULL
import shutil
import argparse
import numpy as np
import nibabel as nb
from scripts.data_preparation.extract_features import io_meld
from scripts.data_preparation.extract_features.create_identity_reg import create_identity

def sample_flair_smooth_features(subject_id, subjects_dir):
    #TODO: rename function
    
    ds_features_to_generate = []
    dswm_features_to_generate = []

    os.makedirs(opj(subjects_dir, subject_id, "surf_meld"), exist_ok=True)

    create_identity(opj(subjects_dir, subject_id))

    hemispheres = ["rh", "lh"]

    if os.path.isfile(opj(subjects_dir, subject_id, "mri", "FLAIR.mgz")):
        for h in hemispheres:
            for d in [0.5, 0.25, 0.75, 0]:
                if not os.path.isfile(f"{subjects_dir}/{subject_id}/surf_meld/{h}.gm_FLAIR_{d}.mgh"):
                    ds_features_to_generate.append((h, d))
            for dwm in [0.5, 1]:
                if not os.path.isfile(f"{subjects_dir}/{subject_id}/surf_meld/{h}.wm_FLAIR_{dwm}.mgh"):
                    dswm_features_to_generate.append((h, dwm))

            print(f'INFO: sample FLAIR features : {ds_features_to_generate}')
            for dsf in ds_features_to_generate:
                # sampling volume to surface
                hemi = dsf[0]
                d = dsf[1]
                command = f"SUBJECTS_DIR='' mri_vol2surf --src {subjects_dir}/{subject_id}/mri/FLAIR.mgz --out {subjects_dir}/{subject_id}/surf_meld/{hemi}.gm_FLAIR_{d}.mgh --hemi {hemi} --projfrac {d} --srcreg {subjects_dir}/{subject_id}/mri/transforms/Identity.dat --trgsubject {subjects_dir}/{subject_id} --surf white"
                proc = Popen(command, shell=True, stdout = DEVNULL, stderr=STDOUT)
                proc.wait()

            print(f'INFO: sample FLAIR features : {dswm_features_to_generate}')
            # Sample FLAIR 0.5mm and 1mm subcortically & smooth using 10mm Gaussian kernel
            for dswmf in dswm_features_to_generate:
                hemi = dswmf[0]
                dwm = dswmf[1]
                command = f"SUBJECTS_DIR='' mri_vol2surf --src {subjects_dir}/{subject_id}/mri/FLAIR.mgz --out {subjects_dir}/{subject_id}/surf_meld/{hemi}.wm_FLAIR_{dwm}.mgh --hemi {hemi} --projdist -{dwm} --srcreg {subjects_dir}/{subject_id}/mri/transforms/Identity.dat --trgsubject {subjects_dir}/{subject_id} --surf white"
                proc = Popen(command, shell=True, stdout = DEVNULL, stderr=STDOUT)
                proc.wait()
    else:
        print(f'INFO: No FLAIR.mgh found for {subject_id}. Skip sampling FLAIR feature')

    
    for hemi in hemispheres:
        # Calculate curvature
        print(f'INFO: Calculate curvatures for {subject_id}')

        command = f"SUBJECTS_DIR={subjects_dir} mris_curvature_stats -f white -g --writeCurvatureFiles {subject_id} {hemi} curv"
        proc = Popen(command, shell=True, stdout = DEVNULL, stderr=STDOUT)
        proc.wait()

        command = f"SUBJECTS_DIR={subjects_dir} mris_curvature_stats -f pial -g --writeCurvatureFiles {subject_id} {hemi} curv"
        proc = Popen(command, shell=True, stdout = DEVNULL, stderr=STDOUT)
        proc.wait()

        # Convert mean curvature and sulcal depth to .mgh file type
        command = f"SUBJECTS_DIR='' mris_convert -c {subjects_dir}/{subject_id}/surf/{hemi}.curv {subjects_dir}/{subject_id}/surf/{hemi}.white {subjects_dir}/{subject_id}/surf_meld/{hemi}.curv.mgh"
        proc = Popen(command, shell=True, stdout = DEVNULL, stderr=STDOUT)
        proc.wait()

        command = f"SUBJECTS_DIR='' mris_convert -c {subjects_dir}/{subject_id}/surf/{hemi}.sulc {subjects_dir}/{subject_id}/surf/{hemi}.white {subjects_dir}/{subject_id}/surf_meld/{hemi}.sulc.mgh"
        proc = Popen(command, shell=True, stdout = DEVNULL, stderr=STDOUT)
        proc.wait()

        command = f"SUBJECTS_DIR='' mris_convert -c {subjects_dir}/{subject_id}/surf/{hemi}.pial.K.crv {subjects_dir}/{subject_id}/surf/{hemi}.white {subjects_dir}/{subject_id}/surf_meld/{hemi}.pial.K.mgh"
        proc = Popen(command, shell=True, stdout = DEVNULL, stderr=STDOUT)
        proc.wait()

        # get gaussian curvature
        print(f'INFO: Compute gaussian curvature for {subject_id}')
        input = opj(subjects_dir, subject_id, "surf_meld", f"{hemi}.pial.K.mgh")
        output = opj(subjects_dir, subject_id, "surf_meld", f"{hemi}.pial.K_filtered.mgh")
        demo = nb.load(input)
        curvature = io_meld.load_mgh(input)
        curvature = np.absolute(curvature)
        curvature = np.clip(curvature, 0, 20)
        io_meld.save_mgh(output, curvature, demo)
        command = f"SUBJECTS_DIR={subjects_dir} mris_fwhm --s {subject_id} --hemi {hemi} --cortex --smooth-only --fwhm 20 --i {subjects_dir}/{subject_id}/surf_meld/{hemi}.pial.K_filtered.mgh --o {subjects_dir}/{subject_id}/surf_meld/{hemi}.pial.K_filtered.sm20.mgh"
        proc = Popen(command, shell=True, stdout = DEVNULL, stderr=STDOUT)
        proc.wait()

        # get thickness
        print(f'INFO: Get thickness and white-greay matter contrast for {subject_id}')
        command = f"SUBJECTS_DIR={subjects_dir} mris_convert -c {subjects_dir}/{subject_id}/surf/{hemi}.thickness {subjects_dir}/{subject_id}/surf/{hemi}.white {subjects_dir}/{subject_id}/surf_meld/{hemi}.thickness.mgh"
        proc = Popen(command, shell=True, stdout = DEVNULL, stderr=STDOUT)
        proc.wait()


        shutil.copy(
            opj(subjects_dir, subject_id, "surf", f"{hemi}.w-g.pct.mgh"),
            opj(subjects_dir, subject_id, "surf_meld", f"{hemi}.w-g.pct.mgh"),
        )


if __name__ == "__main__":
    #parse commandline arguments pointing to subject_dir etc
    parser = argparse.ArgumentParser(description='sample FLAIR and create curvatures')
    parser.add_argument('subject_id', type=str,
                        help='subject_id')
    parser.add_argument('subjects_dir', type=str,
                        help='freesurfer subject directory ')
    args = parser.parse_args()
    #save subjects dir and subject ids. import the text file containing subject ids
    subject_id=args.subject_id
    subjects_dir=args.subject_id
    sample_flair_smooth_features(subject_id, subjects_dir)