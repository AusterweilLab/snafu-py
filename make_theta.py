import rw
import numpy as np
import networkx as nx
import pickle

graphs=[]
matsize=30      # number of nodes in group graph
numsubs=50

thetamat = np.random.rand(matsize,matsize)

for sub in range(numsubs):
    graph = np.zeros((matsize,matsize))
    for i in range(matsize):
        for j in range(matsize):
            if i>j:
                graph[i][j] = np.random.binomial(1, thetamat[i][j])
                graph[j][i] = graph[i][j]                               # symmetric and no self-loops
    graph=nx.from_numpy_matrix(graph)
    graph=max(nx.connected_component_subgraphs(graph), key=len) # take largest component just in case
    if nx.number_of_nodes(graph) != matsize:
        raise ValueError('Generated unconnected graph')
    graph=nx.to_numpy_matrix(graph)
    graphs.append(graph)

for gnum, graph in enumerate(graphs):
    fh=open("theta_graphs/theta_"+str(gnum)+".pickle","w")
    pickle.dump(graphs[gnum],fh)
    fh.close()

# check 
#for i in range(matsize):
#    for j in range(matsize):
#        if i>j:
#            tmp=[]
#            for sub in range(numsubs):
#                tmp.append(graphs[sub][i,j])
#            print np.mean(tmp), thetamat[i][j]
