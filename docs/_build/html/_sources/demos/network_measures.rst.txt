Network Measures Demo
=====================

This demo computes structural metrics for a semantic network generated using the **Conceptual Network** method from fluency data. These measures are useful for comparing networks across individuals, groups, or experimental conditions.

---

**Workflow Summary:**

1. Load and flatten fluency data in the **"animals"** category.
2. Generate a semantic network using the Conceptual Network method.
3. Convert the resulting adjacency matrix to a NetworkX graph.
4. Compute common network metrics using built-in NetworkX functions.
5. Save the results to a `.pkl` file for further use.

---

**Metrics Calculated:**

- `clustering_coefficient`: How often a node’s neighbors are connected.
- `density`: Ratio of edges to all possible edges in the graph.
- `number_of_edges`: Total connections in the network.
- `number_of_nodes`: Vocabulary size represented in the graph.
- `average_node_degree`: Average degree of nodes, based on neighbors' degree.
- `average_shortest_path_length`: Average number of steps in shortest paths (largest component only).
- `diameter`: Longest shortest path (largest component only).

These measures are calculated using both the full network and the **largest connected component**, which avoids failures in disconnected graphs.

---

**Functions and Tools Used:**

- `snafu.conceptualNetwork`
- `networkx.Graph`
- `nx.average_clustering`, `nx.density`, `nx.diameter`, etc.
- `pickle.dump()` for result storage

**Output:**

- `cn_metrics_expected.pkl`: A serialized Python dictionary of the computed metrics.

----

.. automodule:: demos.network_measures
   :members:
   :undoc-members:
   :show-inheritance: