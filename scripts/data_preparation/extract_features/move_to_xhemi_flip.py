import os
import argparse
# from subprocess import Popen,  STDOUT, DEVNULL
import shutil
from meld_classifier.tools_commands_prints import get_m, run_command


def move_to_xhemi_flip(subject_id, subjects_dir, verbose=False):
    measures = [
        "thickness.mgh",
        "w-g.pct.mgh",
        "curv.mgh",
        "sulc.mgh",
        "gm_FLAIR_0.75.mgh",
        "gm_FLAIR_0.5.mgh",
        "gm_FLAIR_0.25.mgh",
        "gm_FLAIR_0.mgh",
        "wm_FLAIR_0.5.mgh",
        "wm_FLAIR_1.mgh",
        "pial.K_filtered.sm20.mgh",
    ]

    os.makedirs(f"{subjects_dir}/{subject_id}/xhemi/surf_meld/", exist_ok=True)

    if not os.path.isfile(f"{subjects_dir}/{subject_id}/xhemi/surf_meld/zeros.mgh"):
        # create one all zero overlay for inversion step
        shutil.copy(
            f"{subjects_dir}/fsaverage_sym/surf/lh.white.avg.area.mgh",
            f"{subjects_dir}/{subject_id}/xhemi/surf_meld/zeros.mgh",
        )

        command = f"SUBJECTS_DIR={subjects_dir} mris_calc --output {subjects_dir}/{subject_id}/xhemi/surf_meld/zeros.mgh {subjects_dir}/{subject_id}/xhemi/surf_meld/zeros.mgh set 0"
        # proc = Popen(command, shell=True, stdout = DEVNULL, stderr=STDOUT)
        proc = run_command(command, verbose=verbose)
        proc.wait()

    print(get_m(f'Move feature to xhemi flip', subject_id, 'INFO'))
    for measure in measures:
        if not os.path.isfile(f"{subjects_dir}/{subject_id}/xhemi/surf_meld/lh.on_lh.{measure}"):
            command = f"SUBJECTS_DIR={subjects_dir} mris_apply_reg --src {subjects_dir}/{subject_id}/surf_meld/lh.{measure} --trg {subjects_dir}/{subject_id}/xhemi/surf_meld/lh.on_lh.{measure} --streg {subjects_dir}/{subject_id}/surf/lh.sphere.reg {subjects_dir}/fsaverage_sym/surf/lh.sphere.reg"
            # proc = Popen(command, shell=True, stdout = DEVNULL, stderr=STDOUT)
            proc = run_command(command, verbose=verbose)
            proc.wait()

        if not os.path.isfile(f"{subjects_dir}/{subject_id}/xhemi/surf_meld/rh.on_lh.{measure}"):
            command = f"SUBJECTS_DIR={subjects_dir} mris_apply_reg --src {subjects_dir}/{subject_id}/surf_meld/rh.{measure} --trg {subjects_dir}/{subject_id}/xhemi/surf_meld/rh.on_lh.{measure} --streg {subjects_dir}/{subject_id}/xhemi/surf/lh.fsaverage_sym.sphere.reg {subjects_dir}/fsaverage_sym/surf/lh.sphere.reg"
            # proc = Popen(command, shell=True, stdout = DEVNULL, stderr=STDOUT)
            proc = run_command(command, verbose=verbose)
            proc.wait()


if __name__ == "__main__":
    #parse commandline arguments pointing to subject_dir etc
    parser = argparse.ArgumentParser(description='move freesurfer volume to xhemi')
    parser.add_argument('subject_id', type=str,
                        help='subject_id')
    parser.add_argument('subjects_dir', type=str,
                        help='freesurfer subject directory ')
    args = parser.parse_args()
    #save subjects dir and subject ids. import the text file containing subject ids
    subject_id=args.subject_id
    subjects_dir=args.subject_id
    move_to_xhemi_flip(subject_id, subjects_dir)
