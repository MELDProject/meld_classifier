# FAQs
This page answers some of the common problems encountered during installation of the 'meld_classifier' package and environment. 

## I have an error during the installation of the meld_classifier virtual environment for OS MAC1. 
```bash
conda env create -f environment_MAC1.yml
```

* **if the error is : *Solving environment: failed. ResolvePackageNotFound: tensorflow-deps***
  
  You will need to force conda to look at osx-arm64: 

    ```bash
    conda config --env --set subdir osx-arm64
    conda env create -f environment_MAC1.yml
    ```

## I have an error during the installation of the meld_classifier virtual environment for Linux and other MAC OS. 
```bash
conda env create -f environment.yml
```

* **if the error is : *CondaValueError: prefix already exists: <path_to_anaconda>/anaconda3/envs/meld_classifier***
  
  You need to remove the already existing environment and reinstall : 
  ```bash
  rm -r <path_to_anaconda>/anaconda3/envs/meld_classifier
  conda env create -f environment.yml
  ```
* **if the error is : *Your shell has not been properly configured to use ‘conda activate’***
 
  First, ensure your conda path is in the '\~/.bashrc' file and not in the '\~/.bash_profile'. 
  
  If in '\~/.bash_profile', copy all the lines between >>> conda initialize >>> and <<< conda initialize <<< into the '\~/.bashrc' file. 
  ```bash
  nano ~/.bash_profile
  nano ~/.bashrc
  ```
  And add the following line just after:
  ```
  source <path_to_anaconda3>/etc/profile.d/conda.sh
  ```
  Save and close the terminal and open a new one. 
   
* **if the error is : *Your shell has not been properly configured to use ‘conda activate’***
  
  You need to activate the bash shell:
  ```bash
  conda init bash
  ```
  Then close the terminal and open a new one. 

* **if the error is : *pip failed***
  
    Open the environment.yml file and comment out all the line below the 1st 'pip' 
    e.g:
    ```
    #- pip:
    #matplotlib_surface_plotting
    #- potpourri3d==0.0.4
    #- nilearn==0.8
    #- neuroCombat==0.2.12
    #- alibi==0.6
    #- patsy==0.5.2
    #- matplotlib==3.5.1
    #- torch==1.12
    #- torchvision==0.13
    #- SimpleITK==2.1.1
    #- git+https://github.com/Deep-MI/LaPy.git
    ```
    Re-install meld_classifier environment
    ```bash
    conda env create -f environment.yml
     ```
    And individually install the pip packages using :
    ```bash
    pip install <package_name>
     ```
  


   ## I have a package installation error when running the pytest.
  
  ```bash
  pytest
  ```

  * **if the error is: *Collected 0 items***
    
    It might mean you have launched the pytest from the wrong folder. Ensure you are in the meld_classifier folder to run :
    ```bash
      cd <path_to_meld_classifier>
      conda activate meld_classifier
      pytest
    ```
    You should get *Collected 45 items*


## I have an error when running the new patient pipeline scripts  .
  * **if the error happens after the predictions and is: *FileNotFoundError: No such file or no access: .../meld_data/output/fs_outputs/<patient_id>/xhemi/surf_meld/rh.on_lh.thickness.mgh*** 
    
    If this is the first time you are using the MELD pipeline and you have this error, you might have installed a version of Freesurfer that is too recent (V7.3 and above) and is incompatible with MELD. Check you Freesurfer version and if it is above V7.3 we would recommend deleting your current Freesurfer and/or installing Freesurfer V7.2 from https://surfer.nmr.mgh.harvard.edu/fswiki/rel7downloads and updating your paths to use this new version as shown in our guidelines and below: 
    ```bash
    export FREESURFER_HOME=<freesurfer_installation_directory>/freesurfer
    source $FREESURFER_HOME/SetUpFreeSurfer.sh
    ```
      Then you will need to run the pipeline again from scratch, but before that you will need to ensure that you have **deleted the folders and files below** so that they can be recreated with the right version of Freesurfer:
      - all the folders and files in ```output/fs_outputs``` 
      - the whole folder ```output/preprocessed_surf_data/MELD_<your_site_code>``` \
      WARNING: it will delete all the outputs created by the MELD pipeline (except the predictions and reports), ensure that you don't need these data for other reasons 



  * **if the error is: It seems that scikit-image has not been built correctly. Your install of scikit-image appears to be broken.**
    
    Reinstall scikit-image as follows:
    ```bash
      conda install -c conda-forge scikit-image=0.18
      ```
      And individually install llvm-openmp using :
      ```bash
      conda install -c conda-forge llvm-openmp
      ```
  * **if the error concerned matplotlib**
    
    Reinstall matplotlib as follows:
    ```bash
      pip uninstall matplotlib
      ```
      And individually install llvm-openmp using :
      ```bash
      pip install matplotlib
      ```
      
  * **if the error looks like:**
      ```
      dyld: lazy symbol binding failed: Symbol not found: ___emutls_get_address
      Referenced from: freesurfer/bin/../lib/gcc/lib/libgomp.1.dylib
      Expected in: /usr/lib/libSystem.B.dylib 
      dyld: Symbol not found: ___emutls_get_address
      Referenced from: freesurfer/bin/../lib/gcc/lib/libgomp.1.dylib
      Expected in: /usr/lib/libSystem.B.dylib
    ```
    It is an issue between Freesurfer V6 and the newer version of MacOs. This error can be solved by following the steps at the bottom of the [Freesurfer web page](https://surfer.nmr.mgh.harvard.edu/fswiki/MacOsInstall)

Contact the MELD team (meld.study@gmail.com) for more information or help