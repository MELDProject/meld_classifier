import numpy as np
import nibabel as nb
import os
import h5py

def load_mgh(filename):
    """ import mgh file using nibabel. returns flattened data array"""
    mgh_file=nb.load(filename)
    mmap_data=mgh_file.get_data()
    array_data=np.ndarray.flatten(mmap_data)
    return array_data;


def import_mgh(filename):
    """ import mgh file using nibabel. returns flattened data array"""
    mgh_file=nb.load(filename)
    mmap_data=mgh_file.get_data()
    array_data=np.ndarray.flatten(mmap_data)
    return array_data;

def save_mgh(filename,array, demo):
    """ save mgh file using nibabel and imported demo mgh file"""
    mmap=np.memmap('/tmp/tmp', dtype='float32', mode='w+', shape=demo.get_data().shape)
    mmap[:,0,0]=array[:]
    output=nb.MGHImage(mmap, demo.affine, demo.header)
    nb.save(output, filename)


#function to load subject features
def load_subject_features(fs_id,features,subject_number,medial_wall,subjects_dir):
    n_vert=163842
    hemis=['lh','rh']
    h_index=-1
    feature_matrix = np.zeros((2*n_vert,len(features)+6))
    for h in hemis:
        h_index+=1
        #create empty matrix with columns for ids, P/C, FLAIR, Lesion Label, Vertex Number, Hemisphere and features
        hemisphere_feature_matrix = np.zeros((n_vert,len(features)+6))
        #subject_id
        hemisphere_feature_matrix[:,0] = np.ones(n_vert,dtype='float32')*subject_number
        #control or patient
        if "_C_" or "_c_" in fs_id:
            hemisphere_feature_matrix[:,1] = np.zeros(n_vert)
        else :
            hemisphere_feature_matrix[:,1] = np.ones(n_vert)
        #check if FLAIR available
        if os.path.isfile(os.path.join(subjects_dir, fs_id, 'xhemi/surf_meld/lh.inter_z.on_lh.intra_z.gm_FLAIR_0.75.sm10.mgh')):
            FLAIR_flag=1
        else:
            FLAIR_flag=0
        hemisphere_feature_matrix[:,2] = np.ones(n_vert)*FLAIR_flag
        lesion_label = os.path.join(subjects_dir, fs_id, 'xhemi/surf_meld', h +'.on_lh.lesion.mgh')
        if os.path.isfile(lesion_label):
            lesion = import_mgh(lesion_label)
            hemisphere_feature_matrix[:,3] = lesion
        #otherwise only zeros
        else :
            hemisphere_feature_matrix[:,3]= np.zeros(n_vert)
        #vertex
        hemisphere_feature_matrix[:,4] = np.arange(n_vert)
        #hemisphere
        hemisphere_feature_matrix[:,5] = np.ones(n_vert,dtype='float32')*h_index
        f_num=5
        for f in features:
            f_num+=1
            try :
                feature = import_mgh(os.path.join(subjects_dir, fs_id, 'xhemi/surf_meld', h+f))
                #set medial wall values to zero
                feature[medial_wall]=0
                hemisphere_feature_matrix[:,f_num] = feature
            except:
                hemisphere_feature_matrix[:,f_num] = np.ones(n_vert)*666
                if "FLAIR" not in f:
                    print('Feature '+f+' not found!')
        feature_matrix[h_index*n_vert : n_vert*(h_index+1),:]=hemisphere_feature_matrix
    return feature_matrix


def get_sitecode(fs_id):
    site_code=fs_id.split('_')[1]
    if site_code[0] != 'H':
        print('site code from subject id does not fit format "H<num>". please double check')
        site_code='false'
    return site_code

def get_cp(fs_id):
    cp=fs_id.split('_')[3]
    if cp in ("FCD" , "fcd"):
        c_p='patient'
    elif cp in ("C" , "c"):
        c_p='control'
    else:
        print('subject '+ fs_id + ' cannot be identified as either patient or control...')
        print('Please double check the IDs in the list of subjects')
        c_p='false'
    return c_p

def get_scanner(fs_id):
    sc=fs_id.split('_')[2]
    if sc in ("15T" , "1.5T" , "15t" , "1.5t" ):
        scanner="15T"
    elif sc in ("3T" , "3t" ):
        scanner="3T"
    else:
        print('scanner for subject '+ fs_id + ' cannot be identified as either 1.5T or 3T...')
        print('Please double check the IDs in the list of subjects')
        scanner='false'
    return scanner

def save_subject(fs_id,features,medial_wall,subject_dir, output_dir=None):
    failed=False
    n_vert=163842
    #get subject info from id
    c_p=get_cp(fs_id)
    scanner=get_scanner(fs_id)
    site_code=get_sitecode(fs_id)
    #skip subject if info not available
    if 'false' in (c_p, scanner, site_code):
        print("Skipping subject " + fs_id)
    hemis=['lh','rh']
    #save feature in hdf5 file
    if output_dir is None:
        output_dir = subject_dir
    hdf5_file = os.path.join(output_dir,site_code+"_"+c_p+"_featurematrix.hdf5")
    if hdf5_file is not None:
        if not os.path.isfile(hdf5_file):
            f = h5py.File(hdf5_file, "a")
        else:
            f = h5py.File(hdf5_file, "r+")
    for h in hemis:
        group=f.require_group(os.path.join(site_code,scanner,c_p,fs_id,h))
        for f_name in features:
            try :
                feature = import_mgh(os.path.join(subject_dir,fs_id,'xhemi/surf_meld',h+f_name))
                feature[medial_wall]=0
                dset=group.require_dataset(f_name,shape=(n_vert,), dtype='float32',compression="gzip", compression_opts=9)
                dset[:]=feature
            except:
                if "FLAIR" not in f_name:
                    print(f'ERROR: {fs_id} : Expected feature '+h+ ' '+ f_name + ' was not found. One step in the pipeline has failed')
                    failed=True
        lesion_name=os.path.join(subject_dir,fs_id,'xhemi/surf_meld',h+'.on_lh.lesion.mgh')
        if os.path.isfile(lesion_name):
            lesion = import_mgh(lesion_name)
            dset=group.require_dataset('.on_lh.lesion.mgh',shape=(n_vert,), dtype='float32',compression="gzip", compression_opts=9)
            dset[:]=lesion
    f.close()
    return failed
