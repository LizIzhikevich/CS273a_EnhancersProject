# CS273a_EnhancersProject

## Installation instructions

* Install [Anaconda](https://docs.anaconda.com/anaconda/install/).
* Install the environment: `conda env create -f environment.yaml`.
* Activate the environment: `source activate config`.
* Start the jupyter server: `jupyter notebook`. This will automatically open up
a new window where you can run the notebooks.

When you're done working in this repository, deactivate the environment:
`source deactivate`.

## Data preprocessing
To parse the FASTA raw data file, run `python3 parse_data.py data.fa parsed_data`.

To preprocess and split the parsed data, run `python3 preprocess_data.py parsed_data preprocessed_data/` (assuming that the directory `preprocessed_data` already exists).

## Model fitting and evalaution
To fit and evaluate models that do not require hyperparameter tuning, run `python3 model_no_tune.py preprocessed_data/ model mode`, where `model` can be one of `nb`, `svm`, or `logit` and `mode` can be `debug` or anything else. To run all the available models, run `sh main_model_no_tune.sh`.

To tune the hyperparameter, fit, and evaluate models with hyperparameter, run `python3 model_tune.py preprocessed_data/ 5 model mode`, where `5` specifies 5-fold CV for hyperparameter tuning, `model` can be one of `l1_logit`, `l2_logit`, or `reg_svm`, and `mode` can be `debug`, `ballpark` (for ballpark estimate of good hyperparameter range), or anything else. To run all the available models, run `sh main_model_tune.sh`. 

The hyperparameters to search can be modified in `model_tune.py` (the variable `C`).

## Output
Precision recall graphs can be found in the prec_recall_curves folder and ROC curves
can be found in the roc_curves folder. Code used to generate these data is in
the plot_and_analyze.py file.

