# How to use the cambridge HPC system (Wikles2-GPU)

The scripts in this folder are specific to the HPC system we used when developing the MELD classifier.
We leave them here as a template, but you will have to adapt them if you would like to run training or evaluation on a slurm-based system.

## General
Our project ids are
- CORE-WCHN-MELD-SL2-GPU (GPU time)
- CORE-WCHN-MELD-SL2-CPU (CPU time)

Use `mybalance` to check how much compute time we have left.

Official docs: https://docs.hpc.cam.ac.uk/hpc/user-guide/quickstart.html

## Setting up the environment
Below are the instructions on how to install a `meld` conda environment on Wilkes2

```
module load miniconda/3
# prepare conda environment
conda create --name meld python=3.6 tensorflow-gpu
source activate ~/.conda/envs/meld
conda install pandas matplotlib seaborn pytest pillow
conda install --channel conda-forge nibabel

# check out meld git repo at desired location
git clone git@github.com:kwagstyl/meld_classifier.git
cd meld_classifier
pip install -e .

```
For convenience, put the commands for activating conda in a script.
Contents of `activate_env.sh`:
```
#!/usr/bin/bash
module load miniconda/3
source activate ~/.conda/envs/meld
```
activate with `source activate_env.sh`

## Running jobs
- interactive jobs:
    - `sintr -A CORE-WCHN-MELD-SL2-GPU -p pascal -N1 -n1 --gres=gpu:1 -t 0:30:00 --qos=INTR` (30 mins, 1 GPU)
    - opens a shell on the GPU node.
    - there, need to activate conda (`source activate_env.sh`) 
    - now, can run python processes on GPU
- batch jobs:
    - run with `sbatch <job-script>.sbatch`
    - an example sbatch job script is in `meld_classifier/scripts/hpc`
    - when editing, note that both the source command to load the conda env and the python command to run the script must be put in the variable `application`, using FULL PATHS! E.g.:
	```
	CONDA_PATH="$HOME/.conda/envs/meld"
	MELD_PATH="$HOME/software/meld_classifier"
	application="source activate $CONDA_PATH; $CONDA_PATH/bin/python $MELD_PATH/scripts/run.py"
	```
	- use `run.sbatch` to call `run.py` and run experiments. You might have to adjust the max runtime for long experiments (find and edit the `#SBATCH --time=hh:mm:ss` line)
	- a slurm-jobid.out file will be automatically created in the current directory containing the job output
	- once a job has started, ssh-ing into the gpu node is possible (for e.g. looking at GPU usage)

## Checking jobs
squeue -u <user-name>

## Use Jupyter notebooks / Jupyter lab
- install jupyter / jupyterlab in the conda environment: `conda install jupyter jupyterlab`
- you can start a jupyter on the login node. This should not be used to do heavy computations, but only for exploration and looking at plots
	- start the notebook: `jupyter notebook --no-browser --ip=127.0.0.1 --port=8081` (`jupyter-lab --no-browser --ip=127.0.0.1 --port=8081`)
	- do port forwarding on the local machine: `ssh -L 8081:127.0.0.1:8081 -fN co-spit1@login-e-X.hpc.cam.ac.uk` (replace X with the actual number of the login node)
	- enjoy
- for computations, jupyter should be started on a compute node. For evaluations that dont require GPU, CPU nodes can be used
	- note the name of the login node 
	- start an interactive session on Skylake (CPU): `sintr -A CORE-WCHN-MELD-SL2-CPU -p skylake -N1 -n1 -t 0:30:00 --qos=INTR`
	- note the name of the compute node
	- start jupyter `jupyter notebook --no-browser --ip=* --port=8081`
	- do port forwarding on the local machine: `ssh -L 8081:cpu-e-Y:8081 -fN co-spit1@login-e-X.hpc.cam.ac.uk` (replace X with the number of the login node, and Y with the number of the compute node)
- If you encounter the “bind: Address already in use” issue, it’s because a port has already been opened (on your local machine). In which case you can stop the process that is associated with that port and try again: `lsof -ti:8081 | xargs kill -9`

