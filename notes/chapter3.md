# Chapter 3 - Classification

## Measuring Accuracy Using Cross-Validation

**Important**: don't confuse `cross_val_score` with `cross_val_predict`.

Interesting example of building a cross-validation score *by hand*. Note the use of `sklearn.base.clone`. Note that `skfolds.split` returns the indices of the training and the test folds.

```py
skfolds = StratifiedKFold(n_splits=3, random_state=42)
for train_ix, test_ix in skfolds.split(X_train, y_train):
    clone_clf = clone(sgd_clf) # Clone the classifier at each iteration
    ...
```

More simply, one can use `cross_val_score` from `sklearn.model_selection`. In the book we initially test it using accuracy as a metric.

```py
cross_val_score(sgd_clf, X_train, y_train, cv=3, scoring='accuracy')
```

### Baseline Classifier by Hand

The `Never5Classifier` is a nice example of how one can create a simple estimator by hand. It contains only the `fit` and the `predict` methods. The former contains just a `pass` statement. The latter returns a vector of zeros of size `(n, 1)` where `n` is the number of input rows.

## `cross_val_predict`

While `cross_val_score` computes a score, for example accuracy, `cross_val_predict` produces a prediction for each observation, using k-fold CV.

```py
cv_preds = cross_val_predict(sgd_clf, X_train, y_train, cv=3)
```

## Confusion Matrix

The correct way to use the confusion matrix is the following:

```py
from sklearn.metrics import confusion_matrix
confusion_matrix(y_true, cv_preds)
```

True classes are in the row of the matrix, while the predicted ones are in the columns.

## Precision, Recall, F1, AUROC

Precision = TP / (TP + FP)
Recall = TP / (TP + FN)

TP is the number of actual positives that we call positive or, equivalently, the number of positive tests that are actually positive.
Precision answers the question: what is the fraction of actual positives in what I call positive?
Recall answers the question: what is the fraction of what I call positive out of the actual positives?

Precision is high when FP is close to zero, no matter how large or small is TP. Precision rewards the lack of false positives. Recall rewards the lack of false negatives.

## Decision Function

Given a classifier, e.g. `sgd_clf`, we can get the decision score with `sgd_clf.decision_function()`. We can compute the scores for all observations and compare them with a manually selected threshold.

```py
y_scores = sgd_clf.decision_function(X_train)
preds = y_scores > threshold
```

## TODO

Try manually augmenting the MNIST dataset with the `scipy.ndimage.interpolation` module. 
