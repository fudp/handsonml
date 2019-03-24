import numpy as np
import matplotlib.pyplot as plt


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

