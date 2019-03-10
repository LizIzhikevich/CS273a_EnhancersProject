import pickle
from sklearn.metrics import roc_curve
import matplotlib.pyplot as plt

svm_clf = pickle.load(open('example_results/svm/svm_clf.pickle', 'rb'))

svm_predicted = pickle.load(open('example_results/svm/svm_test_predicted.pickle', 'rb'))
actual_y = pickle.load(open('example_Y_test.pickle', 'rb'))
nb_predicted = pickle.load(open('example_results/nb/nb_test_predicted.pickle', 'rb'))

def binarize_labels(actual_y, predicted_y, pos_label):
    pred_y_format = predicted_y[:, pos_label]
    actual_format = actual_y == pos_label
    return actual_format.astype(int), pred_y_format

def main():
	# Labels for the respective tissues
	labels = ["Forebrain", "Midbrain", "Hindbrain", "Heart", "Limb", "Other"]
	# Plot ROC to compare best model and baseline Naive Bayes model
	for i in range(0, 5):
		actual_svm, predicted_svm = binarize_labels(actual_y, svm_predicted, i)
		actual_nb, predicted_nb = binarize_labels(actual_y, nb_predicted, i)

		x1, y1, threshold1 = roc_curve(actual_svm, predicted_svm)
		x2, y2, threshold2 = roc_curve(actual_nb, predicted_nb)
		plt.clf()
		plt.title('ROC Curve for Enhancer Regions in %s Tissue \n with Various ML Models' %labels[i])
		plt.plot(x1, y1, label='SVM Algorithm')
		plt.plot([0,1], [0,1], color='black')
		plt.plot(x2, y2, label='Naive Bayes Algorithm')
		plt.ylabel('1 - Sensitivity')
		plt.xlabel('Specificity')
		plt.legend(loc='upper left')
		plt.savefig('roc_curves/%s' %labels[i])

# TODO: Plot the precision-recall curves

if __name__ == '__main__':
	main()

# TODO: For the best model, get the 6mer sequences with the top 10 most positive coefficients for each enhancer type
# in the best model
# TODO: Print them in a table for comparison.
# TODO: Compare their overlap in terms of set.
# TODO: Analyze and compare their ATGC content.