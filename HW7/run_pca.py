"""
Homework: Principal Component Analysis
Course  : Data Mining II (636-0019-00L)
"""

# import all necessary functions
from utils import *
from pca import *

"""
Main Function
"""
if __name__ in "__main__":
    # Initialise plotting defaults
    initPlotLib()

    ##################
    # Exercise 2:

    # Simulate Data
    data = simulateData()
    # print(data)
    # Perform a PCA
    # 1. Compute covariance matrix

    cov = computeCov(data["data"])

    # 2. Compute PCA by computing eigen values and eigen vectors
    evals, pcs = computePCA(cov)
    # Getting first two principal components
    pcs_2 = pcs[:, :2]

    # 3. Transform your input data onto a 2-dimensional subspace using the first two PCs
    transformed = transformData(pcs_2, data["data"])
    # print(transformed.shape)
    # 4. Plot your transformed data and highlight the three different sample classes
    plotTransformedData(transformed, data["target"])

    # 5. How much variance can be explained with each principle component?
    var = computeVarianceExplained(evals)

    # var = sp.array([]) #Compute Variance Explained
    sp.set_printoptions(precision=2)
    print("Variance Explained Exercise 2.1: ")
    for i in range(15):
        print("PC %d: %.2f" % (i + 1, var[i]))
    # 6. Plot cumulative variance explained per PC
    plotCumSumVariance(var)

    ##################
    # Exercise 2 Part 2:

    # 1. normalise data
    data["data"] = dataNormalisation(data["data"])
    cov = computeCov(data["data"])

    print(cov.shape)
    # 2. Compute PCA by computing eigen values and eigen vectors
    evals, pcs = computePCA(cov)
    # Getting first two principal components
    pcs_2 = pcs[:, :2]

    # 3. Transform your input data onto a 2-dimensional subspace using the first two PCs
    transformed = transformData(pcs_2, data["data"])
    # print(transformed.shape)
    # 4. Plot your transformed data and highlight the three different sample classes
    plotTransformedData(transformed, data["target"], normalised=True)

    # 5. How much variance can be explained with each principle component?
    var = computeVarianceExplained(evals)

    # var = sp.array([]) #Compute Variance Explained
    sp.set_printoptions(precision=2)
    print("Variance Explained Exercise 2.1: ")
    for i in range(15):
        print("PC %d: %.2f" % (i + 1, sum(var[i])))
    # 6. Plot cumulative variance explained per PC
    plotCumSumVariance(var, normalised=True)
