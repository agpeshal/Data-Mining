import networkx as nx

my_graph = nx.Graph()  # Generates a new empty graph object

# Adding nodes
my_graph.add_node("node_1")  # Adds node "node_1" to the graph
my_graph.add_nodes_from([1, 2, 3])  # Adds nodes 1, 2, and 3 to the graph

# Adding edges
my_graph.add_edge(1, 2)  # Adds an edge between node 1 and 2
my_graph.add_edges_from([(1, 3), (2, 3)])  # Adds 3 edges

for node in my_graph.nodes():  # Let's you iterate over all nodes
    print(node)

for edge in my_graph.edges():  # Let's you iterate over all edges
    print(edge)

A = nx.to_numpy_matrix(my_graph)  # Returns the graph's adjacency matrix
print(A)
my_graph = nx.from_numpy_matrix(A)  # Returns graph from adjacency matrix
