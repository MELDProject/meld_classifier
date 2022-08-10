import os
import numpy as np
import nibabel as nb
import shutil
import scipy
from scipy import stats as st
from os.path import join as opj
from subprocess import check_call, DEVNULL, STDOUT
import meld_classifier.mesh_tools as mt

def get_adj_mat(surf):

        all_edges = np.vstack(
            [surf["faces"][:, :2], surf["faces"][:, 1:3], surf["faces"][:, [2, 0]]]
        )
        adj_mat = scipy.sparse.coo_matrix(
            (np.ones(len(all_edges), np.uint8), (all_edges[:, 0], all_edges[:, 1])),
            shape=(len(surf["coords"]), len(surf["coords"])),
        ).tocsr()
        return adj_mat
    
def cluster(mask,adj_mat):
        """cluster predictions and threshold based on min_area_threshold

        Args:
            mask: boolean mask of the per-vertex lesion predictions to cluster"""
        n_comp, labels = scipy.sparse.csgraph.connected_components(adj_mat[mask][:, mask])
        islands = np.zeros(len(mask))
        island_count=0
        # only include islands larger than minimum size.
        for island_index in np.arange(n_comp):
            include_vec = labels == island_index
            island_count += 1
            island_mask = mask.copy()
            island_mask[mask] = include_vec
            islands[island_mask] = island_count
        return islands
    
def reindex(clustered,prediction):
    """replace cluster numbers"""
    reindexed_clustered = np.zeros(len(clustered),dtype=int)
    for c in np.unique(clustered)[1:]:
        reindexed_clustered[clustered==c]=st.mode(prediction[clustered==c])[0][0]
    return reindexed_clustered

def correct_interpolation_error(input_mgh,input_surf,output_mgh):
    """correct errors due to interpolation from xhemi back to native"""
    prediction = np.array(mt.load_mgh(input_mgh))
    surf = mt.load_mesh_geometry(input_surf)
    adj = get_adj_mat(surf)
    clustered=cluster(prediction>0,adj)
    reindexed_clustered = reindex(clustered,prediction)
    mt.save_mgh(output_mgh,reindexed_clustered,nb.load(input_mgh))
    return

def register_subject_to_xhemi(subject_ids, subjects_dir, output_dir):
    ''' move the predictions from fsaverage to native space
    inputs:
        subject_ids :  subjects ID in an array
        subjects_dir :  freesurfer subjects directory 
        output_dir :  directory to save final prediction in native space
    '''
    for subject_id in subject_ids:
        # Moves left hemi from fsaverage to native space
        # --src is the source image i.e. the map you want to move back so change to the name of the cluster map in fsaverage_sym that you want to move back
        # --trg is the target image i.e. the name of the map you want to create in the subject's native space
        # the rest is the registration files
        command = f'SUBJECTS_DIR={subjects_dir} mris_apply_reg --src {subjects_dir}/{subject_id}/xhemi/classifier/lh.prediction.mgh --trg {subjects_dir}/{subject_id}/surf/lh.prediction.mgh --streg {subjects_dir}/fsaverage_sym/surf/lh.sphere.reg {subjects_dir}/{subject_id}/surf/lh.sphere.reg'
        check_call(command, shell=True, stdout = DEVNULL, stderr=STDOUT)

        # Moves the right hemi back from fsaverage to native. There are 2 steps
        #Step1: move left hemi fsaverage to right hemi of fsaverage
        # --src is the source image i.e. the map you want to move back so change to the name of the cluster map in fsaverage_sym that you want to move back. should be rh....
        # --trg is the target image i.e. the name of the map on the rh -  i called these rh.{name of file}_on_rh.mgh - these are still in template space (fsaverage)
        # Step 2: move from rh of fsaverage to native space
        # --src is the source image i.e. the name of the file you created in step1
        # --trg is the target image i.e. the name of the map you want to create in the subject's native space

        command = f'SUBJECTS_DIR={subjects_dir} mris_apply_reg --src {subjects_dir}/{subject_id}/xhemi/classifier/rh.prediction.mgh --trg {subjects_dir}/{subject_id}/xhemi/classifier/rh.prediction_on_rh.mgh --streg {subjects_dir}/fsaverage_sym/surf/lh.sphere.reg {subjects_dir}/fsaverage_sym/surf/rh.sphere.left_right'
        check_call(command,shell=True, stdout = DEVNULL, stderr=STDOUT)

        command = f'SUBJECTS_DIR={subjects_dir} mris_apply_reg --src {subjects_dir}/{subject_id}/xhemi/classifier/rh.prediction_on_rh.mgh --trg {subjects_dir}/{subject_id}/surf/rh.prediction.mgh --streg {subjects_dir}/fsaverage_sym/surf/rh.sphere.reg {subjects_dir}/{subject_id}/surf/rh.sphere.reg'
        check_call(command,shell=True, stdout = DEVNULL, stderr=STDOUT)

        #correct from interpolation error
        for hemi in ['lh','rh']:

            #correct prediction from float values in interpolation 
            input_file=opj(subjects_dir,subject_id,'surf',f'{hemi}.prediction.mgh')
            output_mgh=opj(subjects_dir,subject_id,'surf',f'{hemi}.prediction_corr.mgh')
            input_surf=opj(subjects_dir,subject_id,'surf',f'{hemi}.white')
            correct_interpolation_error(input_file,input_surf,output_mgh)

            #map from surface back to vol
            command = f'SUBJECTS_DIR={subjects_dir} mri_surf2vol --identity {subject_id} --template {subjects_dir}/{subject_id}/mri/T1.mgz --o {subjects_dir}/{subject_id}/mri/{hemi}.prediction.mgz --hemi {hemi} --surfval {subjects_dir}/{subject_id}/surf/{hemi}.prediction_corr.mgh --fillribbon'
            check_call(command,shell=True, stdout = DEVNULL, stderr=STDOUT)
        
            #register back to original volume
            command = f'SUBJECTS_DIR={subjects_dir} mri_vol2vol --mov {subjects_dir}/{subject_id}/mri/{hemi}.prediction.mgz --targ {subjects_dir}/{subject_id}/mri/orig/001.mgz  --regheader --o {subjects_dir}/{subject_id}/mri/{hemi}.prediction.mgz --nearest'
            check_call(command,shell=True, stdout = DEVNULL, stderr=STDOUT)

            #convert to nifti
            command = f'SUBJECTS_DIR={subjects_dir} mri_convert {subjects_dir}/{subject_id}/mri/{hemi}.prediction.mgz {subjects_dir}/{subject_id}/mri/{hemi}.prediction.nii -rt nearest'
            check_call(command,shell=True, stdout = DEVNULL, stderr=STDOUT)
            
        #move files
        save_dir=opj(output_dir,subject_id,'predictions')
        os.makedirs(save_dir, exist_ok=True)
            
        shutil.move(f'{subjects_dir}/{subject_id}/mri/lh.prediction.nii', f'{save_dir}/lh.prediction.nii')
        shutil.move(f'{subjects_dir}/{subject_id}/mri/rh.prediction.nii', f'{save_dir}/rh.prediction.nii')
            
        #combine vols from left and right hemis
        command=f'mri_concat --i {save_dir}/lh.prediction.nii --i {save_dir}/rh.prediction.nii --o {save_dir}/prediction.nii --combine'
        check_call(command,shell=True, stdout = DEVNULL, stderr=STDOUT)

if __name__ == "__main__":
    pass

#TODO: for sub in $subjects
#     do
#     
#     done




