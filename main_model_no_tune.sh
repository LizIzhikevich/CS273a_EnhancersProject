#!/bin/bash

python3 model_no_tune.py preprocessed_data/ nb run
python3 model_no_tune.py preprocessed_data/ logit run
python3 model_no_tune.py preprocessed_data/ svm run