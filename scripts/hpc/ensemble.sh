CONDA_PATH="$HOME/.conda/envs/meld"
MELD_PATH="$HOME/software/meld_classifier"

source activate $CONDA_PATH
$CONDA_PATH/bin/python $MELD_PATH/scripts/classifier/ensemble.py --exp iteration_21-05-26 --ensemble-experiment-path iteration_21-05-26 --ensemble-exp-name iteration --ensemble-folds
