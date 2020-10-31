"""
Homework 1: Distance functions on vectors
Course    : Data Mining (636-0018-00L)

Auxiliary functions.

"""

from __future__ import division
import numpy as np
import math


def manhattan_dist(v1, v2):
    return sum(abs(v1 - v2))


def hamming_dist(v1, v2):
    v1 = (v1 > 0).astype(float)
    v2 = (v2 > 0).astype(float)
    return sum(abs(v1 - v2))


def euclidean_dist(v1, v2):
    return math.sqrt(sum(np.power(v1 - v2, 2)))


def chebyshev_dist(v1, v2):
    return max(abs(v1 - v2))


def minkowski_dist(v1, v2, d):
    return np.power(sum(abs(np.power(v1 - v2, d))), 1.0 / d)
