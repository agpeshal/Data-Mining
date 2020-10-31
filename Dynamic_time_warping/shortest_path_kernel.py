from __future__ import division, print_function

import numpy as np
from math import inf


def floyd_warshall(A):
    n = A.shape[0]
    D = np.asarray(A, dtype=np.float64)

    for i, j in np.argwhere(A == 0):
        if i != j:
            D[i][j] = inf
    for k in range(n):
        for i in range(n):
            for j in range(n):
                if D[i][j] > D[i][k] + D[k][j]:
                    D[i][j] = D[i][k] + D[k][j]

    return D


def spkernel(S1, S2):
    S1 = (np.triu(S1)).flatten()
    S2 = (np.triu(S2)).flatten()

    m1 = list(S1[np.nonzero(S1)])
    m2 = list(S2[np.nonzero(S2)])

    min_range = min(min(m1), min(m2))
    max_range = max(max(m1), max(m2))

    hist1 = np.histogram(m1, range=(min_range, max_range))[0]
    hist2 = np.histogram(m2, range=(min_range, max_range))[0]

    return np.dot(hist1, hist2)
