import os
import argparse
import sys
import time
from scripts.new_patient_pipeline.run_script_segmentation import run_script_segmentation
from scripts.new_patient_pipeline.run_script_preprocessing import run_script_preprocessing
from scripts.new_patient_pipeline.run_script_prediction import run_script_prediction

class Logger(object):
    def __init__(self, filename="Default"):
        self.terminal = sys.stdout
        self.filename = filename + time.strftime('%Y-%m-%d-%H-%M-%S') + '.log'
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
    

     
    #write terminal output in a log
    file_path=os.path.join(os.path.abspath(os.getcwd()), 'MELD_pipeline_')
    sys.stdout = Logger(file_path)

    args = parser.parse_args()
    print(args)
    
    #---------------------------------------------------------------------------------
    ### INITIALISE
    # fix some variables for friendly-use
    args.withoutflair = False  #To preprocess the data without FLAIR 
    args.split = True          #To split the subjects ids in chunk of 5 subjects (for large list of subjects)
    args.no_report = False  #To not create meld reports
   
    
    #---------------------------------------------------------------------------------
    ### CHECKS
    if (args.harmo_only) & (args.demographic_file == None):
        print('ERROR: Please provide a demographic file using the flag "-demos" to harmonise your data')
        os.sys.exit(-1)

    #---------------------------------------------------------------------------------
    ### SEGMENTATION ###  
    if not args.skip_segmentation:
        run_script_segmentation(
                            site_code = args.site_code,
                            list_ids=args.list_ids,
                            sub_id=args.id, 
                            use_parallel=args.parallelise, 
                            use_fastsurfer=args.fastsurfer,
                            )

    #---------------------------------------------------------------------------------
    ### PREPROCESSING ###

    run_script_preprocessing(
                    site_code=args.site_code,
                    list_ids=args.list_ids,
                    sub_id=args.id,
                    demographic_file=args.demographic_file,
                    harmonisation_only = args.harmo_only,
                    withoutflair=args.withoutflair,
                    )

    #---------------------------------------------------------------------------------
    ### PREDICTION ###
    if not args.harmo_only:

        run_script_prediction(
                            site_code = args.site_code,
                            list_ids=args.list_ids,
                            sub_id=args.id,
                            no_report = args.no_report,
                            split = args.split
                            )
                
    print(f'You can find a log of the pipeline at {file_path}')