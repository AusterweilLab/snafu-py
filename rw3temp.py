# V3
# - slightly faster genZfromX() code
# - Gibbs sampling
# - todo(rho function [P(X|Z)] )

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

# flat list from tuple walk
def path_from_walk(walk):
    path=list(zip(*walk)[0]) # first element from each tuple
    path.append(walk[-1][1]) # second element from last tuple
    return path

# tuple walk from flat list
def walk_from_path(path):
    walk=[]
    for i in range(len(path)-1):
        walk.append((path[i],path[i+1])) 
    return walk

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

# given a random walk Z, generate z new Z using Gibbs sampling
def gibbZ(x,a,sweeps=10):
    seed=path_from_walk(genZfromX(x,a)) # markov chain starting at random node
    
    # restrict walk to only the next node in x OR previously visited nodes
    possibles=np.zeros(len(a),dtype=int)
    possibles[[x[0],x[1],x[2]]] = 1
    
    while True:
        for i in range(sweeps):
            order=range(len(seed))
            random.shuffle(order)
            for j in order:
                if j==0:
                    possibles=np.nonzero(a[seed[j+1]])[0]  # if first element in walk
                elif j==(len(seed)-1):
                    possibles=np.nonzero(a[seed[j-1]])[0]  # if last element in walk
                else:
                    possibles=np.intersect1d(np.nonzero(a[seed[j-1]])[0],np.nonzero(a[seed[j+1]])[0])  # in the middle
                seed[j]=random.choice(possibles)

        newwalk=walk_from_path(seed)
        yield newwalk

# constrained random walk
# generate random walk that results in observed x
def genZfromX(x,a):
    # restrict walk to only the next node in x OR previously visited nodes
    possibles=np.zeros(len(a),dtype=int)
    possibles[[x[0],x[1],x[2]]] = 1

    walk=[(x[0], x[1])]      # add first two Xs to random walk
    pos=2                    # only allow items up to pos

    while len(x[pos:]) > 0:
        p=walk[-1][1]
        pruned_links=np.intersect1d(np.nonzero(a[p])[0],np.nonzero(possibles)[0])
        s=random.choice(pruned_links)
        walk.append((p,s))
        if s in x[pos:]:
            pos=pos+1
            if len(x[pos:]) > 0:
                possibles[x[pos]] = 1
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
    numnodes=20     # number of nodes in graph
    numlinks=4      # initial number of edges per node (must be even)
    probRewire=.75  # probability of re-wiring an edge
    
    # Generate small-world graph
    g,a=genG(numnodes,numlinks,probRewire) 
    
    # Generate fake participant data
    x=genX(a)
    
    # do some Gibbs sampling
    #sampler=gibbZ(x,a,10)
    #z1=sampler.next()
    #z2=sampler.next()
    #sampler.close()
    
    # Draw graph
    #pos=nx.spring_layout(g)
    #nx.draw(g,pos)
    #nx.draw_networkx_labels(g,pos,labels,font_size=12)
    #plt.show()
