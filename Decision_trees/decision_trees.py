#!/usr/bin/env python3

"""
Homework 4: Logistic Regression and Decision Trees
Part 2: Decision Trees
"""

from __future__ import division
import numpy as np

import sklearn.datasets
from collections import Counter
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier as DTC


def split_data(X, y, attribute_index, theta):
    """
    Divides the data into two parts
    based on attribute at the value of theta
    """
    idx1 = X[:, attribute_index] < theta
    idx2 = np.logical_not(idx1)

    X1 = X[idx1]
    y1 = y[idx1]

    X2 = X[idx2]
    y2 = y[idx2]

    return [X1, X2], [y1, y2]


def compute_information_content(y):
    count = Counter(y).values()
    n = len(y)
    prob = [x / n for x in count]
    info = 0
    for i in range(len(prob)):
        info += -1 * prob[i] * np.log2(prob[i])

    return info


def compute_information_a(X, y, atttribute_index, theta):

    n = len(y)
    X_split, y_split = split_data(X, y, atttribute_index, theta)
    info_a = 0
    for i in range(len(y_split)):
        info_a += len(y_split[i]) / n * compute_information_content(y_split[i])

    return info_a


def compute_information_gain(X, y, attribute_index, theta):

    info_a = compute_information_a(X, y, attribute_index, theta)
    info = compute_information_content(y)
    gain = info - info_a

    return gain


def cross_validation(n_splits, shuffle, feat_map, X, y, c):
    """
    Splits the data for cross validation. For each split we
    find 'c' most important features and accuracy score for
    each split.
    Returns most imp feature out of all splits beased on frequency
    and the average score
    """
    cv = KFold(n_splits=n_splits, shuffle=shuffle)
    clf = DTC()

    score = list()
    feat_imp = list()

    for train_index, test_index in cv.split(X, y):
        X_train, y_train = X[train_index], y[train_index]
        X_test, y_test = X[test_index], y[test_index]

        clf.fit(X_train, y_train)
        feat_sort = np.argsort(-clf.feature_importances_)

        for i in range(c):
            feat_imp.append(feat_map[str(feat_sort[i])])

        y_pred = clf.predict(X_test)
        score.append(accuracy_score(y_test, y_pred))

    freq = Counter(feat_imp)
    feat_imp_name = sorted(freq, key=freq.__getitem__)

    return feat_imp_name, np.mean(score)


if __name__ == "__main__":

    iris = sklearn.datasets.load_iris()
    X = iris.data
    y = iris.target

    feature_names = iris.feature_names
    num_features = len(set(feature_names))

    print("Exercise 2.b")
    print("------------")
    print(
        "Split ( speal length (cm) < 5.5 ): \
        information gain = {:.2f}".format(
            compute_information_gain(X, y, 0, 5.5)
        )
    )
    print(
        "Split ( speal width (cm)  < 3.0 ): \
        information gain = {:.2f}".format(
            compute_information_gain(X, y, 1, 3.0)
        )
    )
    print(
        "Split ( petal length (cm) < 2.0 ): \
        information gain = {:.2f}".format(
            compute_information_gain(X, y, 2, 2.0)
        )
    )
    print(
        "Split ( petal width (cm)  < 1.0 ): \
        information gain = {:.2f}".format(
            compute_information_gain(X, y, 3, 1.0)
        )
    )

    print("")

    print("Exercise 2.c")
    print("------------")

    print(
        "I would select petal length (cm) < 2.0 or \
        petal width (cm)  < 1.0 to be the first because \
        they have the maximum information gain"
    )

    print("")

    ####################################################################
    # Exercise 2.d
    ####################################################################

    # Do _not_ remove this line because you will get different splits
    # which make your results different from the expected ones...
    np.random.seed(42)

    print("Exercise 2.d")
    print("------------")

    feat_map = {
        "0": "sepal length",
        "1": "sepal width",
        "2": "petal length",
        "3": "petal width",
    }

    n_splits = 5
    shuffle = True
    feat_imp_name, avg_score = cross_validation(n_splits, shuffle, feat_map, X, y, c=2)

    print(
        "Mean accuracy score using cross-validation is \
        {:.2f}".format(
            avg_score * 100
        )
    )
    print("-------------------------------------\n")

    print("For the original data, the two most important features are:")
    print(feat_imp_name)

    print("-------------------------------------------\n")

    # Reduced Data: Removing label 2
    X = X[y != 2]
    y = y[y != 2]

    feat_imp_name, avg_score = cross_validation(n_splits, shuffle, feat_map, X, y, c=1)

    print("")
    print("For the reduced data, the most important feature is:")
    print(feat_imp_name)
