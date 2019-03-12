# Tune the hyperparameter for, train, and evaluate models

import sys
from decimal import Decimal
import datetime
import os
import numpy as np
import pickle
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import LabelBinarizer
import evaluator

# Global variables
GEN_LABELS = [(0,1,2), 0, 1, 2, 3, 4, 5] # general enhancer function labels
GEN_LAB_STR = {(0,1,2): 'brain', 0: 'forebrain', 1: 'midbrain', 2: 'hindbrain', 3: 'heart', 4: 'limb', 5: 'others'}

FINE_LABELS = [0, 1, 2, 3, 4] # fine-grain enhancer function labels
FINE_LAB_STR = {0: 'forebrain', 1: 'midbrain', 2: 'hindbrain', 3: 'heart', 4: 'limb'}

# Error exit
def error_exit():
    print('model_tune.py <datadir> <numfold> <model> <mode>')
    sys.exit(2)

# Get the classifier given a model str and param
def get_clf(model_str, param):
    if model_str == 'reg_svm':
        clf = SVC(C=param, decision_function_shape='ovr', class_weight='balanced', gamma='auto', 
            kernel='linear', probability=True)
    elif model_str == 'l1_logit':
        clf = LogisticRegression(penalty='l1', C=param, multi_class='ovr', class_weight='balanced', 
            solver='liblinear')
    elif model_str == 'l2_logit':
        clf = LogisticRegression(penalty='l2', C=param, multi_class='ovr', class_weight='balanced', 
            solver='liblinear')
    else:
        print('Error: %s not supported; model has to be one of reg_svm, l1_logit, or l2_logit\n' % model_str)
        clf = None
        sys.exit(2)
    return clf

def main(args):
	if len(args) < 4:
		error_exit()
	datadir = args[0]
	numfold = int(args[1])
	model = args[2] # one of reg_svm, l1_logit, l2_logit
	mode = args[3] # debug, ballpark, or anything else

	# Set the sample sizes for different modes
	if mode == 'debug':
		train_sample_n = 200
		test_sample_n = 100
	elif mode == 'ballpark': # for ballpark hyparameter tuning
		train_sample_n = 500
		test_sample_n = 250

	# Load the overall training and test data
	with open('%sX_train.pickle' % datadir, 'rb') as file:
		X_train = pickle.load(file)
	with open('%sY_train.pickle' % datadir, 'rb') as file:
		Y_train = pickle.load(file)
	with open('%sX_test.pickle' % datadir, 'rb') as file:
		X_test = pickle.load(file)
	with open('%sY_test.pickle' % datadir, 'rb') as file:
		Y_test = pickle.load(file)

	if mode == 'debug' or mode == 'ballpark':
		X_train = X_train[0:train_sample_n]
		Y_train = Y_train[0:train_sample_n]
		X_test = X_test[0:test_sample_n]
		Y_test = Y_test[0:test_sample_n]

	# Load the k-fold training data
	X_ktrains = []
	Y_ktrains = []
	X_ktests = []
	Y_ktests = []
	for k in range(numfold):
		with open('%sX_ktrain_%d.pickle' % (datadir, k), 'rb') as file:
			X_ktrain = pickle.load(file)
		with open('%sY_ktrain_%d.pickle' % (datadir, k), 'rb') as file:
			Y_ktrain = pickle.load(file)
		with open('%sX_ktest_%d.pickle' % (datadir, k), 'rb') as file:
			X_ktest = pickle.load(file)
		with open('%sY_ktest_%d.pickle' % (datadir, k), 'rb') as file:
			Y_ktest = pickle.load(file)

		assert X_ktrain.shape[0] == Y_ktrain.shape[0]
		assert X_ktest.shape[0] == Y_ktest.shape[0]
    
		if mode == 'debug' or mode == 'ballpark':
			X_ktrain = X_ktrain[0:train_sample_n]
			Y_ktrain = Y_ktrain[0:train_sample_n]
			X_ktest = X_ktest[0:test_sample_n]
			Y_ktest = Y_ktest[0:test_sample_n]

		X_ktrains.append(X_ktrain)
		Y_ktrains.append(Y_ktrain)
		X_ktests.append(X_ktest)
		Y_ktests.append(Y_ktest)

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

	# Set up model parameters
	# The paramters can be changed for different models
	if model == 'reg_svm':
		C = [0.0001, 0.00005, 0.00001, 0.000005, 0.000001]
	elif model == 'l1_logit':
		C = [0.5, 0.1, 0.05, 0.01, 0.005, 0.001, 0.0005, 0.0001]
	elif model == 'l2_logit':
		C = [0.0001, 0.00001, 0.000001, 0.0000001, 0.00000001]
	else:
		print('Error: %s not supported; model has to be one of reg_svm, l1_logit, or l2_logit\n' % model)
		sys.exit(2)

	# Hyperparameter tuning
	print('Tuning hyperparameter...')
	kfold_results = np.zeros((numfold, len(C)))
	onehot_encoder = LabelBinarizer()
	onehot_encoder.fit(range(len(np.unique(Y_train))))
	for i in range(len(C)):
		for fold in range(numfold):
			fold_clf = get_clf(model, C[i])
			fold_clf.fit(X_ktrains[fold], Y_ktrains[fold])
			Y_ktest_pred = fold_clf.predict_proba(X_ktests[fold])
			auroc = roc_auc_score(onehot_encoder.transform(Y_ktests[fold]), Y_ktest_pred, average='macro')
			kfold_results[fold, i] = auroc
			print('Param %d/%d: %d fold completed' % (i + 1, len(C), fold + 1))
	kfold_result_avg = np.mean(kfold_results, axis=0)
	best_param = C[np.argmax(kfold_result_avg)]

	with open(reportfile, 'a') as file:
		file.write('Hyperparameter tuning resutls\n')
		for i in range(len(C)):
			file.write('C = %.3E: mean macro-AUROC = %.3f\n' % (C[i], kfold_result_avg[i]))
		file.write('Best C = %.3E\n\n' % best_param)

	# Train the final model
	print('Training final model...')
	clf = get_clf(model, best_param)
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
