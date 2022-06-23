import os
import numpy as np
import nibabel as nb
import scipy
import argparse
import meld_classifier.mesh_tools as mt
from scipy import stats as st

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
            size = np.sum(include_vec)

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

if __name__ == '__main__':

    #parse commandline arguments 
    parser = argparse.ArgumentParser(description='correct mgh prediction file from interpolation error')
    parser.add_argument('-dir','--dir',
                        help='Freesurfer subject directory',
                        required=True,)
    parser.add_argument('-input','--input',
                        help='input mgh file to correct',
                        required=True,)
    parser.add_argument('-output','--output',
                        help='input mgh file corrected',
                        required=True,)
    args = parser.parse_args()
    
    subdir = args.dir 
    input_mgh=args.input
    output_mgh=args.output

    input_surf=os.path.join(subdir,'surf','rh.white')
    correct_interpolation_error(input_mgh,input_surf,output_mgh)