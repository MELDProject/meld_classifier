#### tests for MeldSubject ####
# tested functions:
#   attributes: site_code, scanner, group
#   get_lesion_hemishpere
#   load_feature_lesion_data
#   other MeldSubject functions (just syntax)
# NOTE:
#   these tests require a test dataset, that is created with get_test_data()
#   executing this function may take a while the first time (while the test data is being created)
# MISSING TESTS:
# - more extensive tests for functions tested in test_meldsubject_api (just tests for syntax)

import pytest
from meld_classifier.meld_cohort import MeldCohort, MeldSubject
import numpy as np
from meld_classifier.download_data import get_test_data
from meld_classifier.paths import DEFAULT_HDF5_FILE_ROOT


def test_subject_parse():
    """
    test if MeldSubject.site_code, .group, .scanner work as expected
    """
    c = MeldCohort(hdf5_file_root=DEFAULT_HDF5_FILE_ROOT)
    subj = MeldSubject("MELD_H4_15T_FCD_0002", c)
    assert subj.site_code == "H4"
    assert subj.scanner == "15T"
    assert subj.group == "patient"
    subj = MeldSubject("MELD_H4_3T_C_0001", c)
    assert subj.group == "control"
    assert subj.scanner == "3T"


def test_get_lesion_hemisphere():
    c = MeldCohort(hdf5_file_root=DEFAULT_HDF5_FILE_ROOT)
    # ensure that patients have a lesional hemisphere
    patients = c.get_subject_ids(site_codes="H4", group="patient", lesional_only=True)
    for subj_id in patients:
        assert MeldSubject(subj_id, c).get_lesion_hemisphere() != None
    controls = c.get_subject_ids(site_codes="H4", group="control")
    for subj_id in controls:
        assert MeldSubject(subj_id, c).get_lesion_hemisphere() == None


testdata = [
    ["MELD_TEST_3T_FCD_0011", DEFAULT_HDF5_FILE_ROOT], 
    ["MELD_TEST_15T_FCD_0004", DEFAULT_HDF5_FILE_ROOT],
]


@pytest.mark.parametrize("subj_id,hdf5_file_root", testdata)
def test_meldsubject_api(subj_id, hdf5_file_root):
    """
    simple test that calls functions for one subject to test for implementation errors
    TODO could add more extensive tests for each one of these functions
    """
    get_test_data()

    c = MeldCohort(hdf5_file_root=hdf5_file_root)
    subj = MeldSubject(subj_id, cohort=c)

    subj.get_demographic_features("Age of onset")

    subj.load_feature_values(".inter_z.asym.on_lh.intra_z.gm_FLAIR_0.25.sm10.mgh")

    subj.get_lesion_area()

    subj.get_histology()

    len(subj.load_boundary_zone()) == 2 * len(c.cortex_label)


@pytest.mark.parametrize("subj_id,hdf5_file_root", testdata)
def test_load_feature_lesion_data(subj_id, hdf5_file_root):
    # TODO also test on TEST site where we know the expected feature values?
    get_test_data()

    c = MeldCohort(hdf5_file_root=hdf5_file_root)
    subj = MeldSubject(subj_id, cohort=c)
    lesion_hemi = subj.get_lesion_hemisphere()
    for hemi in ["lh", "rh"]:
        features, lesion = subj.load_feature_lesion_data(
            features=c.full_feature_list, hemi=hemi, features_to_ignore=c.full_feature_list[:2]
        )
        # check correct number of features
        assert features.shape[1] == len(c.full_feature_list)
        # check that features_to_ignore are 9
        assert np.all(features[:, :2] == 0)
        # check lesion correct
        if hemi == lesion_hemi:
            assert sum(lesion) > 0
        else:
            assert sum(lesion) == 0
