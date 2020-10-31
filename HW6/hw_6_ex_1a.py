#!/usr/bin/env python3

"""
Homework 6: Random Walk Kernel and DBSCAN
"""

import networkx as nx


def generate_graph(G_1, G_2):
    """
    Generates a graph with the vertices of the direct product graph of G_1 and G_2.

    Args:
        G_1 (nx.Graph): First graph for direct product graph construction
        G_2 (nx.Graph): Second graph for direct product graph construction

    Returns:
        nx.Graph: A networkx graph object that contains the nodes of
                  the direct product graph without any edges
    """
    edgeless_graph = nx.Graph()

    nodes = []
    for v1 in G_1.nodes():
        for v2 in G_2.nodes():
            nodes.append(str(v1) + str(v2))

    edgeless_graph.add_nodes_from(nodes)

    return edgeless_graph


def add_edges_to_product_graph(product_G, G_1, G_2):
    """
    Adds edges to an edgeless direct product graph.

    Args:
        product_G: The edgeless product graph from function generate_graph
        G_1 (nx.Graph): First graph for direct product graph construction
        G_2 (nx.Graph): Second graph for direct product graph construction
    """

    for e1 in G_1.edges():
        for e2 in G_2.edges():
            product_G.add_edge(e1[0] + e2[0], e1[1] + e2[1])
            product_G.add_edge(e1[0] + e2[1], e1[1] + e2[0])

    return product_G


def direct_product_graph(G_1, G_2):
    """
    Generates the direct product graph of G_1 and G_2 in two steps:
        1. Generate an edgeless direct product graph
        2. Add respective edges to the direct product graph

    Returns:
        nx.Graph: The direct product graph of G_1 and G_2
    """

    edgeless_graph = generate_graph(G_1, G_2)
    product_G = add_edges_to_product_graph(edgeless_graph, G_1, G_2)

    return product_G


if __name__ == "__main__":
    # This is the code that gets executed when you run 'python hw_6_ex_1a.py'

    # Generate Graph 1 with edges from:
    # node 1 to node 2
    # node 1 to node 3
    # node 2 to node 3

    G_1 = nx.Graph()
    G_1.add_edges_from([("1", "2"), ("1", "3"), ("2", "3")])

    # Generate Graph 2
    # node 1' to node 2'
    # node 1' to node 4'
    # node 2' to node 3'
    # node 3' to node 4'

    G_2 = nx.Graph()
    G_2.add_edges_from([("1'", "2'"), ("1'", "4'"), ("2'", "3'"), ("3'", "4'")])

    G_x = direct_product_graph(G_1, G_2)
    print("The edges of the resulting direct product graph are:")
    for edge in G_x.edges():
        print(edge)

    print("Its adjacency matrix looks like {}".format(nx.to_numpy_array(G_x)))
