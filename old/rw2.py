# V2
# - Generate random Watts-Strogatz graph (G)
# - Generate random participant data (X)
# - Generate possible Zs for X
# - Find P(Z|G)

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import random
import operator
import math

# random walk given an adjacency matrix that hits every node; returns a list of tuples
def random_walk(a,s=None):
    if s is None:
        s=random.choice(range(len(a)))
    walk=[]
    unused_nodes=set(range(len(a)))
    unused_nodes.remove(s)
    while len(unused_nodes) > 0:
        p=s
        s=random.choice(np.nonzero(a[s])[0]) # follow random edge
        walk.append((p,s))
        if s in unused_nodes:
            unused_nodes.remove(s)
    return walk

# flat list from walk
def path_from_walk(walk):
    path=list(zip(*walk)[0]) # first element from each tuple
    path.append(walk[-1][1]) # second element from last tuple
    return path

# Unique nodes in random walk preserving order
# (aka fake participant data)
# http://www.peterbe.com/plog/uniqifiers-benchmark
def observed_walk(walk):
   seen = {}
   result = []
   for item in path_from_walk(walk):
       if item in seen: continue
       seen[item] = 1
       result.append(item)
   return result

def genX(a):
    return observed_walk(random_walk(a))

# constrained random walk
# generate random walk that results in observed x
def genZfromX(x,a):
    x=x[:] # make a local copy
    # restrict walk to only the next node in x OR previously visited nodes
    possibles=np.zeros(len(a),dtype=int)
    possibles[[x[0],x[1],x[2]]] = 1

    # add first two Xs to random walk, delete from X
    walk=[(x[0], x[1])]
    x.pop(0)
    x.pop(0)

    while len(x) > 0:
        p=walk[-1][1]
        pruned_links=np.intersect1d(np.nonzero(a[p])[0],np.nonzero(possibles)[0])
        s=random.choice(pruned_links)
        walk.append((p,s))
        if s in x:
            if len(x) > 1:
                possibles[x[1]] = 1
            x.pop(0) # technically remove "s", but i know it's position 0

    return walk

# probability of random walk Z on link matrix A
def logprobZ(walk,a):
    t=a/sum(a.astype(float))                        # transition matrix
    logProbList=[]
    for i,j in walk:
        logProbList.append(math.log(t[i,j]))
    logProb=reduce(operator.add, logProbList)
    logProb=logProb + math.log(1/float(len(a)))     # base rate of first node when selected uniformly
    return logProb

# Generate a connected Watts-Strogatz small-world graph
# (n,k,p) = (number of nodes, each node connected to k-nearest neighbors, probability of rewiring)
# k has to be even, tries is number of attempts to make connected graph
def genG(n,k,p,tries=1000):
    g=nx.connected_watts_strogatz_graph(n,k,p,tries) # networkx graph
    a=np.array(nx.adjacency_matrix(g).todense())     # adjacency matrix
    #i=nx.incidence_matrix(g).todense()              # incidence matrix
    return g, a
       

if __name__ == "__main__":
    numnodes=20     # number of nodes in graphd
    numlinks=4      # initial number of edges per node (must be even)
    probRewire=.75  # probability of re-wiring an edge

    # Generate small-world graph
    g,a=genG(numnodes,numlinks,probRewire) 

    # Generate fake participant data
    x=genX(a)

    # Draw graph
    #pos=nx.spring_layout(g)
    #nx.draw(g,pos)
    #nx.draw_networkx_labels(g,pos,labels,font_size=12)
    #plt.show()
