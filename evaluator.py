# Toolbox for evaluating model performance

import numpy as np
import itertools
from sklearn.metrics import roc_auc_score

# Make binary true label and predicted probability
def binarize_Y(Y_actual, Y_predicted, pos_labels, neg_labels=None):
    bin_Y_predicted = np.sum(Y_predicted[:, pos_labels], axis=1)

    pos_idx = [label in pos_labels for label in Y_actual]
    if neg_labels == None:
        neg_idx = [label not in pos_labels for label in Y_actual]
    else:
        neg_idx = [label in neg_labels for label in Y_actual]
    
    bin_Y_actual = Y_actual.copy()
    bin_Y_actual[pos_idx] = 1
    bin_Y_actual[neg_idx] = 0
    selected_idx = [pos or neg for pos,neg in zip(pos_idx,neg_idx)]
    return bin_Y_actual[selected_idx], bin_Y_predicted[selected_idx]

# Record the AUROC for one vs. rest classification
def write_ovr_auroc(Y_actual, Y_predicted, pos_labels_lst, label_to_str, textfile):
    with open(textfile, 'a') as file:
        for pos_labels in pos_labels_lst:
            labelstr = label_to_str[pos_labels]
            if type(pos_labels) == tuple:
                pos_labels = list(pos_labels)
            else:
                pos_labels = [pos_labels]
            bin_Y_actual, bin_Y_pred = binarize_Y(Y_actual, Y_predicted, pos_labels)
            auroc = roc_auc_score(bin_Y_actual, bin_Y_pred)
            file.write('%s vs. rest: AUROC = %.3f\n' % (labelstr, auroc))
        file.write('\n')

# Record the AUROC for one vs. one classification
def write_ovo_auroc(Y_actual, Y_predicted, labels, label_to_str, textfile):
    pairs = list(itertools.combinations(labels, 2))
    with open(textfile, 'a') as file:
        for pair in pairs:
            pos_label = pair[0]
            neg_label = pair[1]

            pos_str = label_to_str[pos_label]
            neg_str = label_to_str[neg_label]

            bin_Y_actual, bin_Y_pred = binarize_Y(Y_actual, Y_predicted, [pos_label], [neg_label])
            auroc = roc_auc_score(bin_Y_actual, bin_Y_pred)
            file.write('%s vs. %s: AUROC = %.3f\n' % (pos_str, neg_str, auroc))
        file.write('\n')
