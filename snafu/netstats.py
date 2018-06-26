import numpy as np
import networkx as nx

def degree_dist(g):
    if isinstance(g,np.ndarray):
        g=nx.to_networkx_graph(g)    # if matrix is passed, convert to networkx
    d=dict(g.degree()).values()
    vals=list(set(d))
    counts=[d.count(i) for i in vals]
    return list(zip(vals, counts))
