"""
Homework 4: Logistic Regression and Decision Trees
Part 1: Logistic Regression
"""

#!/usr/bin/env python3
from __future__ import division
import pandas as pd
import numpy as np
import sys

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import StandardScaler


def compute_metrics(y_true, y_pred):
    """
    Computes several quality metrics of the predicted labels and prints
    them to `stdout`.

    :param y_true: true class labels
    :param y_pred: predicted class labels
    """

    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()

    print("Exercise 1.a")
    print("------------")
    print("TP: {0:d}".format(tp))
    print("FP: {0:d}".format(fp))
    print("TN: {0:d}".format(tn))
    print("FN: {0:d}".format(fn))
    print("Accuracy: {0:.3f}".format(accuracy_score(y_true, y_pred)))


def data_read_transform(train_file, test_file):

    """
    Reads the csv file, mean fit the data
    and transform using standard functions
    Returns the test and train data
    """
    try:
        df = pd.read_csv(train_file)
        X_train = df.iloc[:, 0:7].values
        Y_train = df.iloc[:, 7].values

        df = pd.read_csv(test_file)
        X_test = df.iloc[:, 0:7].values
        Y_test = df.iloc[:, 7].values

    except IOError:
        print("Either train or test file cannot be opened")
        sys.exit(1)

    std = np.std(X_train, axis=0)
    # transforming input variables
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.fit_transform(X_test)

    return X_train, Y_train, X_test, Y_test, std


def LDA(X, y):
    """
    Calculates the parameters of Linear Discriminant Analysis
    Finds intra class mean and sigma matrix for the whole dataset
    """
    mu_0 = np.mean(X[y == 0, :], axis=0)
    mu_1 = np.mean(X[y == 1, :], axis=0)

    sigma_inv = np.linalg.inv(np.cov(X.T))

    w = np.dot((mu_1 - mu_0).T, sigma_inv)
    b = 0.5 * (
        np.dot(mu_0.T, np.dot(sigma_inv, mu_0))
        - np.dot(mu_1.T, np.dot(sigma_inv, mu_1))
    )

    return w, b


if __name__ == "__main__":

    ###################################################################

    train_file = "./data/diabetes_train.csv"
    test_file = "./data/diabetes_test.csv"

    X_train, Y_train, X_test, Y_test, std = data_read_transform(train_file, test_file)

    ###################################################################

    print("\nExercise 1.a")
    print("------------")

    clf = LogisticRegression().fit(X_train, Y_train)
    Y_pred = clf.predict(X_test)
    compute_metrics(Y_test, Y_pred)

    print("\nExercise 1.b")
    print("------------")

    w, b = LDA(X_train, Y_train)
    Y_LDA = (np.dot(X_test, w.T) + b) > 0
    # compute_metrics(Y_test, Y_LDA)

    print(
        "For the diabetes dataset I would choose Logistic Regression,\
         because it has an accuracy score of {:.3f}".format(
            accuracy_score(Y_test, Y_pred)
        ),
        end=" ",
    )
    print("while LDA has an accuracy of {:.3f}".format(accuracy_score(Y_test, Y_LDA)))

    print("\nExercise 1.c")
    print("------------")

    print(
        "I would choose LDA for applications where number of false \
        positives needs to be smaller or there is a small number of \
        point in the relevant class.",
        end=" ",
    )
    print(
        "While I would use Logistic Regression if I am more concerned \
        about the accuracy and the classes are roughly balanced.",
        end=" "
    )

    print("\nExercise 1.d")
    print("------------")

    coeff = clf.coef_[0]

    print("The coefficient for npreg is {:.2f}.".format(coeff[0]), end=" ")
    print(
        "Calculating the exponential function results in {:.2f}, \
        which amounts to an increase in diabetes risk of {:.1f} \
        percent per additional pregnancy.".format(
            np.exp(coeff[0]), (np.exp(coeff[0]) - 1) * (1 / std[0]) * 100
        )
    )
