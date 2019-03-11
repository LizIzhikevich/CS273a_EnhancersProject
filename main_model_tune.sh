#!/bin/bash

python3 model_tune.py preprocessed_data/ 5 l1_logit run
python3 model_tune.py preprocessed_data/ 5 l2_logit run
python3 model_tune.py preprocessed_data/ 5 reg_svm run