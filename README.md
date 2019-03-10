# CS273a_EnhancersProject

## Installation instructions

* Install [Anaconda](https://docs.anaconda.com/anaconda/install/).
* Install the environment: `conda env create -f environment.yaml`.
* Activate the environment: `source activate config`.
* Start the jupyter server: `jupyter notebook`. This will automatically open up
a new window where you can run the notebooks.

When you're done working in this repository, deactivate the environment:
`source deactivate`.

## Parse the FASTA data file
Run `python3 parse_data.py data.fa parsed_data`.

## Output
Precision recall graphs can be found in the prec_recall_curves folder and ROC curves
can be found in the roc_curves folder. Code used to generate these data is in
the plot_and_analyze.py file.

