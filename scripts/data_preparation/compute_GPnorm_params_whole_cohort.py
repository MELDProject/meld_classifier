from meld_classifier.paths import BASE_PATH, EXPERIMENT_PATH, MELD_DATA_PATH
from meld_classifier.meld_cohort import MeldCohort, MeldSubject
from meld_classifier.data_preprocessing import Preprocess, Feature
import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import nibabel as nb
import GPy

def fit_predict_gpy(y,x,max_age = 80):
    """fits gaussian kernel
    y is feature vector eg thickness for 1 vertex for all controls
    x is [age, sex] column
    max_age - maximum age in dataset for predicting
    returns mu - means for 0-max_age M & F
    std - standard devations for 0-max_age M & F
    
    """
    x_mean=np.mean(x,axis=0)
    x_std=np.std(x,axis=0)
    x_demean=(x-x_mean)/x_std
    #create and fit model
    k1 = GPy.kern.RBF(2, active_dims=(0, 1), lengthscale=2)
    k2 = GPy.kern.White(2, active_dims=(0, 1))
    k3 = GPy.kern.Linear(2)
    k_add = k1 + k2 + k3
    m = GPy.models.SparseGPRegression(x_demean, y.reshape(-1,1), kernel=k_add)
    m.optimize('bfgs', max_iters=100)
    #now predict z and std for each age & sex
    pred_vals=np.hstack([np.vstack([np.arange(max_age),np.zeros(max_age)]),
                     np.vstack([np.arange(max_age),np.ones(max_age)])]).T
    pred_age_demean=(pred_vals-x_mean)/x_std
    mu,std=m.predict(
        pred_age_demean)
    mu=mu.ravel()
    std=std.ravel()
    mu=mu.reshape((2,max_age)).T
    std = std.reshape((2,max_age)).T
    #to be stored in n_vertices x n_ages x n_sex matrix

    return mu,std


############################# MAIN ################################################################

#load cohort
site_codes=['H2', 'H3','H4','H5','H6','H7','H9','H10','H11','H12','H14','H15','H16','H17','H18','H19',
                  'H21','H23','H24','H26']
cohort= MeldCohort(hdf5_file_root='{site_code}_{group}_featurematrix_combat_6.hdf5', dataset=None,
             data_dir=MELD_DATA_PATH)

params_file='norm_GP_params.hdf5'


# create preprocessing object
norm = Preprocess(cohort, 
                  site_codes=site_codes, 
                  write_hdf5_file_root=None, 
                  data_dir=BASE_PATH)

# get controls ids
controls_ids = cohort.get_subject_ids(group="control")
# Give warning if list of controls empty
if len(controls_ids) == 0:
    print("WARNING: there is no controls in this cohort to do inter-normalisation")

features = [
         ".combat.on_lh.thickness.sm10.mgh",
#         ".combat.on_lh.w-g.pct.sm10.mgh",
#         '.combat.on_lh.curv.sm5.mgh',
#          '.combat.on_lh.gm_FLAIR_0.25.sm10.mgh',
#          '.combat.on_lh.gm_FLAIR_0.5.sm10.mgh',
#          '.combat.on_lh.gm_FLAIR_0.75.sm10.mgh',
#          '.combat.on_lh.gm_FLAIR_0.sm10.mgh',
#          '.combat.on_lh.pial.K_filtered.sm20.mgh',
#          '.combat.on_lh.sulc.sm5.mgh',
#          '.combat.on_lh.wm_FLAIR_0.5.sm10.mgh',
#          '.combat.on_lh.wm_FLAIR_1.sm10.mgh',
]
# for each feature, compute GP normalisation parameters and save them
for feature in features : 
    print(feature)
    vals_array = []
    included_subj = []
    for k,id_sub in enumerate(controls_ids):
        # create subject object
        subj = MeldSubject(id_sub, cohort=cohort)
        # append data to compute mean and std if feature exist
        if subj.has_features(feature):
            # load feature's value for this subject
            vals_lh = subj.load_feature_values(feature, hemi="lh")
            vals_rh = subj.load_feature_values(feature, hemi="rh")
            vals = np.array(np.hstack([vals_lh[cohort.cortex_mask], vals_rh[cohort.cortex_mask]]))
            vals_array.append(vals)
            included_subj.append(id_sub)
        else:
            pass
    included_subj= np.array(included_subj)
    print("Compute GP normalisation from {} controls".format(len(included_subj)))
    # compute normalisation if values exists
    if vals_array:
        vals_array = np.array(vals_array)
        # load in covariates - age, sex
        covars = norm.load_covars(included_subj).copy()
        # Remove controls with missing demographic info
        indices=[]
        for k,subject in enumerate(included_subj):
            if np.isnan(covars['ages'][k]) == True:
                print(subject)
                indices.append(k)
            elif np.isnan(covars['sex'][k]) == True:
                print(subject)
                indices.append(k)
            else:
                pass
        vals_array = np.delete(vals_array, obj=indices, axis=0)
        covars = covars.drop(indices)   
        # compute GP params for each vertices
        mu_mat = []
        std_mat = []
        for vertex_index in np.arange(2*sum(cohort.cortex_mask)):
            y=vals_array[:,vertex_index]
            x = np.array(covars[['ages','sex']])
            mu,std= fit_predict_gpy(y,x,max_age = 80)
            mu_mat.append(mu)
            std_mat.append(std)
    else:
        print('no data to compute GP normalisation parameters')
        pass
    # get mu and std parameters from controls
    params = {}
    params['mu_mat'] = np.array(mu_mat)
    params['std_mat'] = np.array(std_mat)
    #save parameters in hdf5
    if params_file!=None:
        params_file = os.path.join(BASE_PATH,params_file)
        print(f'save parameters in {params_file}')
        norm.save_norm_combat_parameters(feature, params, params_file)


