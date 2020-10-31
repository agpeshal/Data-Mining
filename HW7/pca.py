import scipy as sp
import scipy.linalg as linalg
import matplotlib.pyplot as pl
import numpy as np

from utils import plot_color

"""############################"""
"""Principle Component Analyses"""
"""############################"""

"""
Compute Covariance Matrix
Input: Matrix of size #samples x #features
Output: Covariance Matrix of size #features x #features
Note: Do not use scipy or numpy cov. Implement the function yourself.
      You can of course add an assert to check your covariance function
      with those implemented in scipy/numpy.
"""


def computeCov(X=None):

    X = np.asarray(X)
    n, d = X.shape
    return 1.0 / n * np.dot(X.T, X)


"""
Compute PCA
Input: Covariance Matrix
Output: [eigen_values,eigen_vectors] sorted in such a why that eigen_vectors[:,0] is the first principle component
        eigen_vectors[:,1] the second principle component etc...
Note: Do not use an already implemented PCA algorithm. However, you are allowed to use an implemented solver 
      to solve the eigenvalue problem!
"""


def computePCA(matrix=None):

    eigen_vals, eigen_vec = linalg.eigh(matrix)

    sort_idx = np.argsort(-1 * eigen_vals)

    eigen_vec_sort = np.zeros_like(eigen_vec)
    for i, j in enumerate(sort_idx):
        eigen_vec_sort[:, i] = eigen_vec[:, j]

    return sorted(eigen_vals, reverse=True), eigen_vec_sort


"""
Transform Input Data Onto New Subspace
Input: pcs: matrix containing the first x principle components
       data: input data which should be transformed onto the new subspace
Output: transformed input data. Should now have the dimensions #samples x #components_used_for_transformation
"""


def transformData(pcs=None, data=None):

    return np.matmul(data, pcs)


"""
Compute Variance Explaiend
Input: eigen_values
Output: return vector with varianced explained values.
"""


def computeVarianceExplained(evals=None):

    return np.abs(evals) / (sum(np.abs(evals)))


"""############################"""
"""Different Plotting Functions"""
"""############################"""

"""
Plot Transformed Data
Input: transformed: data matrix (#sampels x 2)
       labels: target labels, class labels for the samples in data matrix
       filename: filename to store the plot
"""


def plotTransformedData(transformed=None, labels=None, normalised=False):
    pl.figure()
    # PLOT FIGURE HERE
    labels_uniq = np.unique(labels)
    for lb in labels_uniq:
        idx = np.where(labels == lb)

        pl.scatter(transformed[idx][:, 0], transformed[idx][:, 1], s=5, label=str(lb))

    pl.legend()
    # You can use plot_color[] to obtain different colors for your plots
    # Save File
    if normalised == True:
        pl.savefig("scatter_normalised")
    else:
        pl.savefig("scatter_unnormalised")


"""
Plot Cumulative Explained Variance
Input: var: variance explained vector
       filename: filename to store the file
"""


def plotCumSumVariance(var=None, normalised=False):
    pl.figure()
    pl.plot(sp.arange(var.shape[0]), sp.cumsum(var) * 100)
    pl.xlabel("Principle Component")
    pl.ylabel("Cumulative Variance Explained in %")
    pl.grid(True)
    # Save file
    if normalised == True:
        pl.savefig("cumsum_normalised")
    else:
        pl.savefig("cumsum_unnormalised")


"""############################"""
"""Data Preprocessing Functions"""
"""############################"""

"""
Exercise 2 Part 2:
Data Normalisation (Zero Mean, Unit Variance)
"""


def dataNormalisation(X=None):
    mean = np.mean(X, axis=0)
    std = np.std(X, axis=0)

    return (X - mean[:, None].T) / std[:, None].T
