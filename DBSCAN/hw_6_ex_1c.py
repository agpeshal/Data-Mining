#!/usr/bin/env python3

"""
Homework 6: Random Walk Kernel and DBSCAN
"""

import numpy as np
import networkx as nx
from hw_6_ex_1a import *
from hw_6_ex_1b import *


def degree_matrix(A):
    """
    Creates degree matrix from given adjacency matrix A

    Args:
        Adjacency matrix

    Returns:
        np.ndarray: Degree matrix of graph represented by adjacency matrix A
    """

    deg = np.asarray(np.sum(A, axis=0))[0]
    degree_matrix = np.diag(deg)

    return degree_matrix


def starting_prob_vector(A):
    """
    Creates an 1 x n dimensional vector whose entries
    are the degree of the nth node normalized by the sum of all degrees

    Args:
        Adjacency matrix

    Returns:
        np.ndarray: 1 x n dimensional vector with starting probabilities
    """

    deg = np.asarray(np.sum(A, axis=0))[0]

    return deg / sum(deg)


def stopping_prob_vector(A):
    """
    Creates an 1 x n dimensional vector whose entries
    are the degree of the nth node normalized by the sum of all degrees

    Args:
        Adjacency matrix

    Returns:
        np.ndarray: 1 x n dimensional vector with stopping probabilities
    """

    deg = np.asarray(np.sum(A, axis=0))[0]

    return deg / sum(deg)


def rw_kernel(G_1, G_2, lam=0.01):
    """
    Calculates the random walk kernel as defined in equation 2 of the homework sheet
    The starting and stopping probabilities are supposed to be calculated in this function

    Args:
        G_1 (nx.Graph): First graph for direct product graph construction
        G_2 (nx.Graph): Second graph for direct product graph construction
        lam (float): lambda parameter

    Returns:
        The value of the random walk graph kernel applied to the graphs
        of figure 1 of the homework sheet
    """

    G_x = direct_product_graph(G_1, G_2)
    A_x = nx.to_numpy_matrix(G_x)

    G_prod = direct_product_graph_adj_matrix(G_1, G_2)
    W_x = nx.to_numpy_matrix(G_prod)

    q_x = starting_prob_vector(A_x)
    p_x = stopping_prob_vector(A_x)

    I = np.identity(W_x.shape[0])

    kernel = np.dot(np.dot(q_x.T, np.linalg.inv(I - lam * W_x)), p_x)

    return kernel


if __name__ == "__main__":
    # This is the code that gets executed when you run 'python hw_6_ex_1c.py'

    # Generate Graph 1 with edges from:
    #    node 1 to node 2
    #    node 1 to node 3
    #    node 2 to node 3

    G_1 = nx.Graph()
    G_1.add_edges_from([("1", "2"), ("1", "3"), ("2", "3")])

    # Generate Graph 2
    #    node 1' to node 2'
    #    node 1' to node 4'
    #    node 2' to node 3'
    #    node 3' to node 4'

    G_2 = nx.Graph()
    G_2.add_edges_from([("1'", "2'"), ("1'", "4'"), ("2'", "3'"), ("3'", "4'")])

    kernel_result = rw_kernel(G_1, G_2)
    print(kernel_result[0, 0])
