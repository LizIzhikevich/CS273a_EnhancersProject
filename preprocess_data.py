import sys
import datetime
import itertools
import numpy as np
import pickle
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedKFold
from tqdm import tqdm
import random

# Global variables
K = 6 # 6mer
LABELS = ['forebrain', 'midbrain', 'hindbrain', 'heart', 'limb'] # tissue of interest
SEED=123 # random state

# Error exit
def error_exit():
    print('preprocess_data.py <parsed_datafile> <out_dir> <randomize>')
    sys.exit(2)

# Encode string labels as integers
def encode_label(labels):
    encoding = {}
    for i in range(len(labels)):
        encoding[labels[i]] = i
    return encoding

# Get the label code for a sample based on the associated tissues
def get_y(tissues, label_encoding):
    others_encoding = len(label_encoding)
    codes = [label_encoding[tissue] for tissue in tissues if tissue in label_encoding.keys()]
    if len(codes) == 0: # not associated with any tissue of interest
        return others_encoding
    elif len(codes) == 1:
        return codes[0]
    else: # associated with more than one tissue of interest; to be discarded
        return None

# Find all the possible DNA sequence kmers based on combinatorics
def find_possible_kmers(k):
    bases = ['G', 'C', 'T', 'A']
    possible_kmers = sorted([
        ''.join(mer)
        for mer in list(itertools.product(bases, repeat = 6))
    ])
    return possible_kmers

# Find the kmers for a sample DNA sequence
def find_seq_kmers(sequence, k):
    return [
        sequence[i:i+k]
        for i in range(len(sequence) - k + 1)
    ]

# Find the kmer encoding for a sample DNA sequence
def encode_sequence(sequence, possible_kmers):
    sequence = sequence.upper()
    kmer_dict = {}
    for kmer in find_seq_kmers(sequence, len(possible_kmers[0])):
        if kmer in kmer_dict:
            kmer_dict[kmer] += 1
        else:
            kmer_dict[kmer] = 1

    return [
        kmer_dict[kmer] if kmer in kmer_dict else 0
        for kmer in possible_kmers
    ]

# Wrtie the label distribution to a textfile for record
def write_Y_dist(Y, str_labels, textfile, preamble):
	Y_codes, Y_counts = np.unique(Y, return_counts=True)
	total_count = np.sum(Y_counts)
	with open(textfile, 'a') as file:
		file.write('%s\n' % preamble)
		for i in range(Y_codes.shape[0]):
			Y_code = Y_codes[i]
			Y_count = Y_counts[i]
			file.write('%s (%d): %d (%.2f%%)\n' % (str_labels[Y_code], Y_code, Y_count, Y_count/total_count*100))
		file.write('Total count: %d\n' % total_count)
		file.write('\n')

def main(args):
	if len(args) < 3:
		error_exit()
	parsed_datafile = args[0]
	out_dir = args[1]
	randomize = args[2] # randomize the negative sequences

	# Initiate the record for label distributions
	label_dist_file = '%slabel_dist.txt' % out_dir
	with open(label_dist_file, 'w') as file:
		file.write('Label distributions for different data sets\n')
		file.write(datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S\n\n"))

	# Load data
	with open(parsed_datafile, 'rb') as f:
		parsed_data = pickle.load(f)

	# Generate encoding maps
	possible_6mers = find_possible_kmers(K)
	label_encoding = encode_label(LABELS)
	with open('%spossible_6mers.pickle' % out_dir, 'wb') as file:
		pickle.dump(possible_6mers, file)
	with open('%slabel_encoding.pickle' % out_dir, 'wb') as file:
		pickle.dump(label_encoding, file)

	# Generate design matrix X and label vector Y as numpy arrays
	X = []
	Y = []
	for d in tqdm(parsed_data):
		sequence = d[0]
		tissues = d[2]
		y = get_y(tissues, label_encoding)
		if y is not None:
			if randomize == 'randomize' and y == len(label_encoding):
				print(y)
				print(randomize)
				sequence = ''.join(random.sample(sequence, len(sequence)))
			X.append(encode_sequence(sequence, possible_6mers))
			Y.append(y)
	X = np.array(X)
	Y = np.array(Y)

	# Stratified splitting for traing and test sets
	X_train, X_test, Y_train, Y_test = train_test_split(X, Y, stratify=Y, shuffle=True, 
		test_size=0.3, random_state=SEED)

	# Save files
	with open('%sX_train.pickle' % out_dir, 'wb') as file:
		pickle.dump(X_train, file)
	with open('%sX_test.pickle' % out_dir, 'wb') as file:
		pickle.dump(X_test, file)

	with open('%sY_train.pickle' % out_dir, 'wb') as file:
		pickle.dump(Y_train, file)
	with open('%sY_test.pickle' % out_dir, 'wb') as file:
		pickle.dump(Y_test, file)

	# Write label distributions
	str_labels = LABELS.copy()
	str_labels += ['others']
	write_Y_dist(Y_train, str_labels, label_dist_file, 'Whole training set')
	write_Y_dist(Y_test, str_labels, label_dist_file, 'Whole testing set')

	# Stratified splitting for 5-fold CV
	kfold_splitter = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
	kfold_splitter.get_n_splits(X_train, Y_train)
	fold = 0
	for ktrain_idx, ktest_idx in kfold_splitter.split(X_train, Y_train):
		X_ktrain, X_ktest = X_train[ktrain_idx], X_train[ktest_idx]
		Y_ktrain, Y_ktest = Y_train[ktrain_idx], Y_train[ktest_idx]

		# Save files
		with open('%sX_ktrain_%d.pickle' % (out_dir, fold), 'wb') as file:
			pickle.dump(X_ktrain, file)
		with open('%sX_ktest_%d.pickle' % (out_dir, fold), 'wb') as file:
			pickle.dump(X_ktest, file)

		with open('%sY_ktrain_%d.pickle' % (out_dir, fold), 'wb') as file:
			pickle.dump(Y_ktrain, file)
		with open('%sY_ktest_%d.pickle' % (out_dir, fold), 'wb') as file:
			pickle.dump(Y_ktest, file)

		# Write label distributions
		write_Y_dist(Y_ktrain, str_labels, label_dist_file, '%d Training fold' % fold)
		write_Y_dist(Y_ktest, str_labels, label_dist_file, '%d Test fold' % fold)
		fold += 1

if __name__ == "__main__":
	main(sys.argv[1:])
