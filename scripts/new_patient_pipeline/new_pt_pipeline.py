import os
import argparse
import sys
import time
from scripts.new_patient_pipeline.run_script_segmentation import run_script_segmentation
from scripts.new_patient_pipeline.run_script_preprocessing import run_script_preprocessing
from scripts.new_patient_pipeline.run_script_prediction import run_script_prediction
from meld_classifier.tools_commands_prints import get_m

class Logger(object):
    def __init__(self, sys_type=sys.stdout, filename='MELD_output.log'):
        self.terminal = sys_type
        self.filename = filename
        self.log = open(self.filename, "a")

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)
    
    def flush(self):
        pass

if __name__ == "__main__":
    # parse commandline arguments
    parser = argparse.ArgumentParser(description="Main pipeline to predict on subject with MELD classifier")
    parser.add_argument("-id","--id",
                        help="Subject ID.",
                        default=None,
                        required=False,
                        )
    parser.add_argument("-ids","--list_ids",
                        default=None,
                        help="File containing list of ids. Can be txt or csv with 'ID' column",
                        required=False,
                        )
    parser.add_argument("-site",
                        "--site_code",
                        help="Site code",
                        required=True,
                        )
    parser.add_argument("--fastsurfer", 
                        help="use fastsurfer instead of freesurfer", 
                        required=False, 
                        default=False,
                        action="store_true",
                        )
    parser.add_argument("--parallelise", 
                        help="parallelise segmentation", 
                        required=False,
                        default=False,
                        action="store_true",
                        )
    parser.add_argument('-demos', '--demographic_file', 
                        type=str, 
                        help='provide the demographic files for the harmonisation',
                        required=False,
                        default=None,
                        )
    parser.add_argument('--harmo_only', 
                        action="store_true", 
                        help='only compute the harmonisation combat parameters, no further process',
                        required=False,
                        default=False,
                        )
    parser.add_argument('--skip_segmentation',
                        action="store_true",
                        help='Skip the segmentation and extraction of the MELD features',
                        )
    parser.add_argument('--no_nifti',
                        action="store_true",
                        default=False,
                        help='Only predict. Does not produce prediction on native T1, nor report',
                        )
    parser.add_argument('--no_report',
                        action="store_true",
                        default=False,
                        help='Predict and map back into native T1. Does not produce report',)
    parser.add_argument('--split',
                        action="store_true",
                        default=False,
                        help='Split subjects list in chunk to avoid data overload',
                        )
    parser.add_argument("--debug_mode", 
                        help="mode to debug error", 
                        required=False,
                        default=False,
                        action="store_true",
                        )
    

     
    #write terminal output in a log
    file_path=os.path.join(os.path.abspath(os.getcwd()), 'MELD_pipeline_'+time.strftime('%Y-%m-%d-%H-%M-%S') + '.log')
    sys.stdout = Logger(sys.stdout,file_path)
    sys.stderr = Logger(sys.stderr, file_path)
    
    args = parser.parse_args()
    print(args)
    
    #---------------------------------------------------------------------------------
    ### CHECKS
    if (args.harmo_only) & (args.demographic_file == None):
        print('ERROR: Please provide a demographic file using the flag "-demos" to harmonise your data')
        os.sys.exit(-1)

    #---------------------------------------------------------------------------------
    ### SEGMENTATION ###
    if not args.skip_segmentation:
        print(get_m(f'Call script segmentation', None, 'SCRIPT 1'))
        result = run_script_segmentation(
                            site_code = args.site_code,
                            list_ids=args.list_ids,
                            sub_id=args.id, 
                            use_parallel=args.parallelise, 
                            use_fastsurfer=args.fastsurfer,
                            verbose = args.debug_mode
                            )
        if result == False:
            print(get_m(f'Segmentation and feature extraction has failed at least for one subject. See log at {file_path}. Consider fixing errors or excluding these subjects before re-running the pipeline. Segmentation will be skipped for subjects already processed', None, 'SCRIPT 1'))    
            sys.exit()
    else:
        print(get_m(f'Skip script segmentation', None, 'SCRIPT 1'))

    #---------------------------------------------------------------------------------
    ### PREPROCESSING ###
    print(get_m(f'Call script preprocessing', None, 'SCRIPT 2'))
    run_script_preprocessing(
                    site_code=args.site_code,
                    list_ids=args.list_ids,
                    sub_id=args.id,
                    demographic_file=args.demographic_file,
                    harmonisation_only = args.harmo_only,
                    )

    #---------------------------------------------------------------------------------
    ### PREDICTION ###
    if not args.harmo_only:
        print(get_m(f'Call script prediction', None, 'SCRIPT 3'))
        result = run_script_prediction(
                            site_code = args.site_code,
                            list_ids=args.list_ids,
                            sub_id=args.id,
                            no_prediction_nifti = args.no_nifti,
                            no_report = args.no_report,
                            split = args.split,
                            verbose = args.debug_mode
                            )
        if result == False:
            print(get_m(f'Prediction and mapping back to native MRI has failed at least for one subject. See log at {file_path}. Consider fixing errors or excluding these subjects before re-running the pipeline. Segmentation will be skipped for subjects already processed', None, 'SCRIPT 3'))    
            sys.exit()
    else:
        print(get_m(f'Skip script predition', None, 'SCRIPT 3'))
                
    print(f'You can find a log of the pipeline at {file_path}')
