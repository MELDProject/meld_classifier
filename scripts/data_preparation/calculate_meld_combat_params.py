### script to calculate reference combat parameters to use in distributed harmonisation
from meld_classifier.paths import BASE_PATH, EXPERIMENT_PATH,MELD_DATA_PATH
from meld_classifier.meld_cohort import MeldCohort, MeldSubject
import os
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import pickle
import pandas as pd
import neuroCombat as nc
import meld_classifier.distributedCombat as dc
import numpy as np
import neuroCombat as nc
from meld_classifier.data_preprocessing import Preprocess, Feature

new_site_code = 'H99'
site_combat_path = os.path.join(BASE_PATH,'distributed_combat')
if not os.path.isdir(site_combat_path):
    os.makedirs(site_combat_path)

site_codes=['H2', 'H3','H4','H5','H6','H7','H9','H10','H11','H12','H14','H15','H16','H17','H18','H19',
                  'H21','H23','H24','H26',]
c_combat =  MeldCohort(hdf5_file_root='{site_code}_{group}_featurematrix_combat_6.hdf5', dataset=None,
                      data_dir=MELD_DATA_PATH)
listids = c_combat.get_subject_ids(site_codes=site_codes, lesional_only=True)

preprocessor=Preprocess(c_combat)
#load in precombat data
ref_subject_ids = c_combat.get_subject_ids(lesional_only=False)

#get feature names
features = {
    ".on_lh.thickness.mgh": 10,
    ".on_lh.w-g.pct.mgh": 10,
    ".on_lh.pial.K_filtered.sm20.mgh": None,
    ".on_lh.sulc.mgh": 5,
    ".on_lh.curv.mgh": 5,
    ".on_lh.gm_FLAIR_0.25.mgh": 10,
    ".on_lh.gm_FLAIR_0.5.mgh": 10,
    ".on_lh.gm_FLAIR_0.75.mgh": 10,
    ".on_lh.gm_FLAIR_0.mgh": 10,
    ".on_lh.wm_FLAIR_0.5.mgh": 10,
    ".on_lh.wm_FLAIR_1.mgh": 10,
}
feat = Feature()
features_smooth = [feat.smooth_feat(feature, features[feature]) for feature in features]
features_combat = [feat.combat_feat(feature) for feature in features_smooth]


for fi,feature in enumerate(features_smooth):
    print("harmonising :", feature)
    #load cohort
    precombat_features=[]
    combat_subject_include = np.zeros(len(ref_subject_ids), dtype=bool)
    new_site_codes=np.zeros(len(ref_subject_ids))
    print('loading')
    for k, subject in enumerate(ref_subject_ids):
        # get the reference index and cohort object for the site, 0 whole cohort, 1 new cohort
        site_code_index = new_site_codes[k]

        subj = MeldSubject(subject, cohort=c_combat)
        # exclude outliers and subject without feature
        if (subj.has_features(features_combat[fi])) :
            lh = subj.load_feature_values(features_combat[fi], hemi="lh")[c_combat.cortex_mask]
            rh = subj.load_feature_values(features_combat[fi], hemi="rh")[c_combat.cortex_mask]
            combined_hemis = np.hstack([lh, rh])
            precombat_features.append(combined_hemis)
            combat_subject_include[k] = True
        else:
            combat_subject_include[k] = False

    #load covars
    precombat_features = np.array(precombat_features).T
    covars = preprocessor.load_covars(ref_subject_ids)
    covars = covars[combat_subject_include].copy().reset_index()
    N=len(covars)
    bat = pd.Series(pd.Categorical(np.repeat('H0', N), categories=['H0', new_site_code]))
    covars['site_scanner']=bat
    covars = covars[['ages','sex','group','site_scanner']]

    print('calculating')

    #DO COMBAT steps
    #use var estimates from basic combat
    com_out = nc.neuroCombat(precombat_features, covars, 'site_scanner')
    with open(os.path.join(site_combat_path,f'MELD_{feature}_var.pickle'), "wb") as f:
        pickle.dump(com_out['estimates']['var.pooled'], f)
    #calculate reference estimates for distributed combat
    _ = dc.distributedCombat_site(precombat_features, bat, covars[['ages','sex','group']], 
                              file=os.path.join(site_combat_path,
                                                f'MELD_{feature}.pickle'), ref_batch = 'H0')

