# MELD classifier
Neural network lesion classifier for the MELD project.

The manuscript describing the classifier can be found here https://academic.oup.com/brain/advance-article/doi/10.1093/brain/awac224/6659752

*Code Authors : Hannah Spitzer, Mathilde Ripart, Sophie Adler, Konrad Wagstyl*

![overview](images/overview.png)

This package comes with a pretrained model that can be used to predict new subjects. It also contains code for training neural network lesion classifiers on new data.

## Disclaimer

The MELD surface-based FCD detection algorithm is intended for research purposes only and has not been reviewed or approved by the Medicines and Healthcare products Regulatory Agency (MHRA), European Medicine Agency (EMA) or by any other agency. Any clinical application of the software is at the sole risk of the party engaged in such application. There is no warranty of any kind that the software will produce useful results in any way. Use of the software is at the recipient's own risk.

## Installation

### Prerequisites
For preprocessing, MELD classifier requires Freesurfer. It is trained on data from versions 6 & v5.3. Please follow instructions on [Freesurfer](https://surfer.nmr.mgh.harvard.edu/fswiki/DownloadAndInstall) to install FreeSurfer v6. \
New update ! MELD pipeline is now also working with FastSurfer (quicker version of Fresurfer). If you wish to use FastSurfer instead please follow instructions for the [native install of Fastsurfer](https://github.com/Deep-MI/FastSurfer.git). Note that Fastsurfer requires to install Freesurfer V7.2 to works

### Conda installation
We use [anaconda](https://docs.anaconda.com/anaconda/install/mac-os/) to manage the environment and dependencies. Please follow instructions on [anaconda](https://docs.anaconda.com/anaconda/install/mac-os/) to install Anaconda.

Install MELD classifier and python dependencies:
```bash
# checkout and install the github repo 
git clone https://github.com/MELDProject/meld_classifier.git 

# enter the meld_classifier directory
cd meld_classifier
# create the meld classifier environment with all the dependencies 
# ! Note : If you have a new MAC1 OS system, you will need to install the special environments for new MAC1 users in the second command below.
conda env create -f environment.yml    # For Linux and old MAC os users
conda env create -f environment_MAC1.yml  # For new MAC1 users 
# activate the environment
conda activate meld_classifier
# install meld_classifier with pip (with `-e`, the development mode, to allow changes in the code to be immediately visible in the installation)
pip install -e .
```

### Set up paths and download model
Before being able to use the classifier on your data, some paths need to be set up and the pretrained model needs to be downloaded. For this, run:
```bash
python scripts/prepare_classifier.py
```

This script will ask you for the location of your **MELD data folder** and download the pretrained model and test data to a folder inside your MELD data folder. Please provide the path to where you would like to store MRI data to run the classifier on.


Note: You can also skip the downloading of the test data. For this, append the option `--skip-download-data` to the python call.

### FAQs
Please see our [FAQ](FAQs.md) for common installation problems.

### Verify installation
We provide a test script to allow you to verify that you have installed all packages, set up paths correctly, and downloaded all data. This script will run the pipeline to predict the lesion classifier on a new patient. It takes approximately 15minutes to run.

```bash
cd <path_to_meld_classifier>
conda activate meld_classifier
pytest
```
Note: If you run into errors at this stage and need help, you can re-run the command below to save the terminal outputs in a txt file, and send it to us. We can then work with you to solve any problems.
  ```bash
  pytest -s | tee pytest_errors.log
  ```
  You will find this pytest_errors.log file in <path_to_meld_classifier>. 

## Usage
With this package, you can use the provided classifier to predict subjects from existing and new sites. For new site, you will need to harmonise your data first. In addition, you can train your own classifier model.
For more details, check out the guides linked below:
- [Predict new subjects (existing site)](Predict_on_new_patient.md)
- [Harmonise your data (new site)](Harmonisation_new_site.md)
- [Train and evaluate models](Training_and_evaluating_models.md)

## Contribute
If you'd like to contribute to this code base, have a look at our [contribution guide](DEVELOP.md)

## Manuscript
Please check out our [manuscript](https://academic.oup.com/brain/advance-article/doi/10.1093/brain/awac224/6659752) to learn more.

An overview of the notebooks that we used to create the figures can be found [here](figure_notebooks.md).

A guide to using the MELD surface-based FCD detection algorithm on a new patient is found [here](https://docs.google.com/document/d/1vF5U1i-B45OkE_8wdde8yHHypp6W9xNN_1DBoEGmn0E/edit?usp=sharing).


## Acknowledgments

We would like to thank the [MELD consortium](https://meldproject.github.io//docs/collaborator_list.pdf) for providing the data to train this classifier and their expertise to build this pipeline.\
We would like to thank [Lennart Walger](https://github.com/1-w) and [Andrew Chen](https://github.com/andy1764), for their help testing and improving the MELD pipeline to v1.1.0
