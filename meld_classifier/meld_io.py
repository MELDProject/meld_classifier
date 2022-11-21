import numpy as np


def get_sitecode(fs_id):
    site_code = fs_id.split("_")[1]
    if site_code[0] != "H":
        print('site code from subject id does not fit format "H<num>". please double check')
        site_code = "false"
    return site_code


def get_cp(fs_id):
    """use ID to determine patient or control"""
    cp = fs_id.split("_")[3]
    if cp in ("FCD", "fcd"):
        c_p = "patient"
    elif cp in ("C", "c"):
        c_p = "control"
    else:
        print("subject " + fs_id + " cannot be identified as either patient or control...")
        print("Please double check the IDs in the list of subjects")
        c_p = "false"
    return c_p


def get_scanner(fs_id):
    """ get scanner  3T or 1.5T"""
    sc = fs_id.split("_")[2]
    if sc in ("15T", "1.5T", "15t", "1.5t"):
        scanner = "15T"
    elif sc in ("3T", "3t"):
        scanner = "3T"
    else:
        print("scanner for subject " + fs_id + " cannot be identified as either 1.5T or 3T...")
        print("Please double check the IDs in the list of subjects")
        scanner = "false"
    return scanner


def resort_like(unsorted, sorted_list):
    """because set diff reorders weirdly.
    keep subject sorting like in original subject ids list"""
    subjects = []
    for sub in sorted_list:
        if sub in unsorted:
            subjects.append(sub)
    return np.array(subjects)
