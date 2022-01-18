# FAQs
This page answers some of the common problems encountered during installation of the 'meld_classifier' package and environment. 

### I have an error during the installation of the meld_classifier virtual environment. 
```bash
conda env create -f environment.yml
```
* if the error is : *CondaValueError: prefix already exists: <path_to_anaconda>/anaconda3/envs/meld_classifier*
  
  You need to remove the already existing environment and reinstall : 
  ```bash
  rm -r <path_to_anaconda>/anaconda3/envs/meld_classifier
  conda env create -f environment.yml
  ```
* if the error is : *Your shell has not been properly configured to use ‘conda activate’*
 
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
   
* if the error is : *Your shell has not been properly configured to use ‘conda activate’*
  
  You need to activate the bash shell:
  ```bash
  conda init bash
  ```
  Then close the terminal and open a new one. 
    
  * if the error is : *pip failed*
  
    Open the environment.yml file and comment oout all the line below the 1st 'pip' 
    e.g:
    ```
    #- pip:
    #  - matplotlib_surface_plotting
    #  - potpourri3d
    #  - umap-learn
    #  - nilearn
    #  - git+https://github.com/kwagstyl/neuroCombat.git@estimates_shrink
    #  - alibi
    ```
    Re-install meld_classifier environment
    ```bash
    conda env create -f environment.yml
     ```
    And individually install the pip packages using :
    ```bash
    pip install <package_name>
     ```
 
   
