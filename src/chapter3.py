import numpy as np
import matplotlib.pyplot as plt
from imblearn.over_sampling import RandomOverSampler
from sklearn.model_selection import StratifiedKFold
from sklearn.base import clone


def sort_by_train(mnist):
    """Reorder the MNIST data as in the first edition of the book.
    Note: the function modifies the dataset *in-place*."""
    reorder_train = np.array(sorted([(target, i) for i, target in enumerate(
        mnist.target[:60000])]))[:, 1]
    reorder_test = np.array(sorted([(target, i) for i, target in enumerate(
        mnist.target[60000:])]))[:, 1]
    mnist.data[:60000] = mnist.data[reorder_train]
    mnist.target[:60000] = mnist.target[reorder_train]
    mnist.data[60000:] = mnist.data[reorder_test]
    mnist.target[60000:] = mnist.target[reorder_test]


def plot_digits(instances, images_per_row=10, **options):
    """Arrange the MNIST images in a grid and visualize them."""
    size = 28
    images_per_row = min(len(instances), images_per_row)
    images = [instance.reshape(size, size) for instance in instances]
    n_rows = (len(instances) - 1) // images_per_row + 1
    row_images = []
    n_empty = n_rows * images_per_row - len(instances)
    images.append(np.zeros((size, size * n_empy)))
    for row in range(n_rows):
        rimages = images[row * images_per_row : (row + 1) * images_per_row]
        row_images.append(np.concatenate(rimages, axis=1))
    image = np.concatenate(row_images, axis=0)
    plt.imshow(image, cmap=mpl.cm.binary, **options)
    plt.axis('off')


def plot_precision_recall_vs_threshold(precisions, recalls, thresholds):
    plt.plot(thresholds, precisions[:-1], 'b--', label='Precision',
             linewidth=2)
    plt.plot(thresholds, recalls[:-1], 'g-', label='Recall',
             linewidth=2)
    plt.xlabel('Threshold', fontsize=16)
    plt.legend(loc='upper left', fontsize=16)
    plt.ylim([0, 1])


def plot_precision_vs_recall(precisions, recalls):
    plt.plot(recalls, precisions, 'b-', linewidth=2)
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.axis([0, 1, 0, 1])


def plot_f1_scores_vs_thresholds(precisions, recalls, thresholds):
    f1_scores = 2 * precisions * recalls / (precisions + recalls)
    plt.plot(thresholds, f1_scores[1:], 'b-', linewidth=2)
    plt.xlabel('Threshold')
    plt.ylabel('F1 score')
    plt.ylim([0, 1])


def find_threshold_by_precision(precisions, thresholds, p_cutoff):
    """This function finds the lowest index that guarantess a precision
    of at leat p_cutoff."""
    return thresholds[np.argmax(precisions >= p_cutoff)]


def plot_roc_curve(fpr, tpr, label=None):
    """Plot a ROC curve given the output of the `roc_curve` function."""
    plt.plot(fpr, tpr, linewidth=2, label=label)
    plt.plot([0, 1], [0, 1], 'k--')
    plt.axis([0, 1, 0, 1])
    plt.xlabel('False Positive Rate', fontsize=16)
    plt.ylabel('True Positive Rate', fontsize=16)


def imbalanced_cross_val_predict(clf, x, y, cv):
    """**NOTE** this function could probably be implemented in a much more
    elegant way by using an estimator that performs the over-sampling.
    """
    skf = StratifiedKFold(n_splits=cv, random_state=42)
    ros = RandomOverSampler()
    y_pred = []
    y_true = []
    for train_idx, test_idx in skf.split(x, y):
        clone_clf = clone(clf)
        # Create the training and validation folds
        x_train_folds, y_train_folds = x[train_idx], y[train_idx]
        x_test_fold, y_test_fold = x[test_idx], y[test_idx]

        # Over-sample the training folds
        x_train_folds_os, y_train_folds_os = ros.fit_resample(
            x_train_folds, y_train_folds)

        # Fit the classifier to the over-sampled set
        clone_clf.fit(x_train_folds_os, y_train_folds_os)

        # Save the true validation labels
        y_true = np.concatenate([y_true, y_test_fold])
        
        # Predict the class label in the left-out fold
        preds = clone_clf.predict(x_test_fold)
        y_pred = np.concatenate([y_pred, preds])
    return y_true, y_pred
