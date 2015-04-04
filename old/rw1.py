# V1
# - Generate random Erdos-Renyi graph
# - Take a random walk
# - Reconstruct graph simply by drawing edges between nodes in the random walk


import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import random

# random walk given an adjacency matrix and start node and number of steps; returns a list of tuples
def random_walk(a,steps,s=0):
    walk=[]
    if sum(a)[0,s] == 0:
        print 'Warning: Starting node in random walk not connected to any other node!'
        return walk
    for i in range(steps):
        p=s
        s=random.choice(np.where(np.array(a[s])[0])[0]) # there must be a better way?
        walk.append((p,s))
    return walk
   
# flat list from walk
def path_from_walk(walk):
    if walk == []:
        print 'Warning: Walk is empty!'
        return []
    path=list(zip(*walk)[0]) # first element from each tuple
    path.append(walk[-1][1]) # second element from last tuple
    return path

# adjacency matrix from walk
def a_from_walk(walk,numnodes=0):
    if walk == []:
        print 'Warning: Walk is empty!'
        return []
    if numnodes==0:
        numnodes=max(path_from_walk(walk))+1
    a=np.zeros((numnodes,numnodes))
    for i in walk:
        a[i[0],i[1]]=1
    return np.matrix(a.astype(int))



# Generate Erdos-Renyi graph using G(n,p) [nodes, probability of transition]
n=10
p=.2
g=nx.binomial_graph(n,p)            # networkx graph

a=nx.adjacency_matrix(g).todense()  # adjacency matrix
t=a/sum(a.astype(float))            # transition matrix
#i=nx.incidence_matrix(g).todense() # incidence matrix

if 0 in sum(a):
    print 'Warning: Some nodes in initial graph have no edges!'

walk=random_walk(a,10,0) # random walk 10 steps on a, starting from node 0
new_a=a_from_walk(walk)  # recover adjacency matrix from walk

if len(set(path_from_walk(walk))) < n:
    print 'Warning: Some nodes in initial graph missing in reconstructed graph!'

# Draw graph
#nx.draw(g)
#plt.show()
