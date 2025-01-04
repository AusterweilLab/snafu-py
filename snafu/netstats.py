from . import *

def degree_dist(g):
    if isinstance(g,np.ndarray):
        g=nx.to_networkx_graph(g)    # if matrix is passed, convert to networkx
    d=dict(g.degree()).values()
    vals=list(set(d))
    counts=[d.count(i) for i in vals]
    return list(zip(vals, counts))

# return small world statistic of a graph
# returns metric of largest component if disconnected
def smallworld(a):
    g_sm=nx.from_numpy_matrix(a)
    g_sm=g_sm.subgraph(max(nx.connected_components(g_sm),key=len))   # largest component
    numnodes=g_sm.number_of_nodes()
    numedges=g_sm.number_of_edges()
    nodedegree=(numedges*2.0)/numnodes
    
    c_sm=nx.average_clustering(g_sm)        # c^ws in H&G (2006)
    #c_sm=sum(nx.triangles(usfg).values())/(# of paths of length 2) # c^tri
    l_sm=nx.average_shortest_path_length(g_sm)
    
    # c_rand same as edge density for a random graph? not sure if "-1" belongs in denominator, double check
    #c_rand= (numedges*2.0)/(numnodes*(numnodes-1))   # c^ws_rand?  
    c_rand= float(nodedegree)/numnodes                  # c^tri_rand?
    l_rand= np.log(numnodes)/np.log(nodedegree)    # approximation, see humphries & gurney (2008) eq 11
    #l_rand= (np.log(numnodes)-0.5772)/(np.log(nodedegree)) + .5 # alternative ASPL from fronczak, fronczak & holyst (2004)
    s=(c_sm/c_rand)/(l_sm/l_rand)
    return s
