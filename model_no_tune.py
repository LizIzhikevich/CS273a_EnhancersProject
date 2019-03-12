# Train and evaluate models that do not require hyperparameter tuning

import sys
import datetime
import os
import numpy as np
import pickle
from sklearn.svm import SVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
import evaluator

# Global variables
GEN_LABELS = [(0,1,2), 0, 1, 2, 3, 4, 5] # general enhancer function labels
GEN_LAB_STR = {(0,1,2): 'brain', 0: 'forebrain', 1: 'midbrain', 2: 'hindbrain', 3: 'heart', 4: 'limb', 5: 'others'}

FINE_LABELS = [0, 1, 2, 3, 4] # fine-grain enhancer function labels
FINE_LAB_STR = {0: 'forebrain', 1: 'midbrain', 2: 'hindbrain', 3: 'heart', 4: 'limb'}

# Error exit
def error_exit():
	print('model_no_tune.py <datadir> <model> <mode>')
	sys.exit(2)

def main(args):
	if len(args) < 3:
		error_exit()
	datadir = args[0]
	model = args[1]
	mode = args[2]

	# Load the data
	with open('%sX_train.pickle' % datadir, 'rb') as file:
		X_train = pickle.load(file)
	with open('%sY_train.pickle' % datadir, 'rb') as file:
		Y_train = pickle.load(file)
	with open('%sX_test.pickle' % datadir, 'rb') as file:
		X_test = pickle.load(file)
	with open('%sY_test.pickle' % datadir, 'rb') as file:
		Y_test = pickle.load(file)

	# Get smaller data sets for debugging
	if mode == 'debug':
		X_train = X_train[0:200]
		Y_train = Y_train[0:200]
		X_test = X_test[0:100]
		Y_test = Y_test[0:100]

	if model == 'nb':
		clf = MultinomialNB()
	elif model == 'svm':
		clf = SVC(decision_function_shape='ovr', class_weight='balanced', gamma='auto', 
			kernel='linear', probability=True)
	elif model == 'logit':
		clf = LogisticRegression(multi_class='ovr', class_weight='balanced', solver='liblinear')
	else:
		print('Error: %s not supported; model has to be one of nb, svm, or logit\n' % model)
		sys.exit(2)

	# Create directories for saving files
	master_resultdir = 'results'
	model_resultdir = '%s/%s_' % (master_resultdir, model)
	model_resultdir += datetime.datetime.now().strftime("%Y_%m_%d_%H_%M")

	if not os.path.exists(master_resultdir):
		os.makedirs(master_resultdir)
	if not os.path.exists(model_resultdir):
		os.makedirs(model_resultdir)

	# Create file for record keeping
	reportfile = '%s/report.txt' % model_resultdir
	with open(reportfile, 'w') as file:
		file.write('Model: %s\n' % model)
		file.write('Mode: %s\n' % mode)
		file.write('Modeling process started: %s\n' % datetime.datetime.now().strftime("%Y-%m-%d %H:%M"))

	# Fit the model
	print('Training model...')
	clf.fit(X_train, Y_train)
	with open('%s/trained_mdl.pickle' % model_resultdir, 'wb') as file:
		pickle.dump(clf, file)

	# Get predicted probability
	print('Evaluating model...')
	Y_train_pred = clf.predict_proba(X_train)
	Y_test_pred = clf.predict_proba(X_test)
	with open('%s/Y_train_pred.pickle' % model_resultdir, 'wb') as file:
		pickle.dump(Y_train_pred, file)
	with open('%s/Y_test_pred.pickle' % model_resultdir, 'wb') as file:
		pickle.dump(Y_test_pred, file)

	# Evaluate the model for training AUROC
	with open(reportfile, 'a') as file:
		file.write('Training results\n')
	evaluator.write_ovr_auroc(Y_train, Y_train_pred, GEN_LABELS, GEN_LAB_STR, reportfile)
	evaluator.write_ovo_auroc(Y_train, Y_train_pred, FINE_LABELS, FINE_LAB_STR, reportfile)

	# Evaluate the model for test AUROC
	with open(reportfile, 'a') as file:
		file.write('Testing results\n')
	evaluator.write_ovr_auroc(Y_test, Y_test_pred, GEN_LABELS, GEN_LAB_STR, reportfile)
	evaluator.write_ovo_auroc(Y_test, Y_test_pred, FINE_LABELS, FINE_LAB_STR, reportfile)

	with open(reportfile, 'a') as file:
		file.write('Modeling process ended: %s\n' % datetime.datetime.now().strftime("%Y-%m-%d %H:%M"))
	print('Done!')

if __name__ == "__main__":
	main(sys.argv[1:])
