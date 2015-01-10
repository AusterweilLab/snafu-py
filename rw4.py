#!/usr/bin/python

# V4

import networkx as nx
import numpy as np
import random
import operator
import math
import matplotlib.pyplot as plt

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

# first hitting times for each node
def firstHit(walk):
    firsthit=[]
    path=path_from_walk(walk)
    for i in observed_walk(walk):
        firsthit.append(path.index(i))
    return zip(observed_walk(walk),firsthit)

# generate random walk that results in observed x (using rho function)
def genZfromX(x, theta):
    x2=x[:]                  # make a local copy
    x2.reverse()
    
    path=[]                  # z to return
    path.append(x2.pop())    # add first two x's to z
    path.append(x2.pop())

    while len(x2) > 0:
        if random.random() < theta:
            # add random hidden node
            possibles=set(path) # choose equally from previously visited nodes
            possibles.discard(path[-1]) # but exclude last node (node cant link to itself)
            path.append(random.choice(list(possibles)))
        else:
            # first hit!
            path.append(x2.pop())
    return walk_from_path(path)

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

# returns both networkx graph G and link matrix A
def genGfromZ(walk):
    numnodes=len(observed_walk(walk))
    a=np.zeros((numnodes,numnodes))
    for i in set(walk):
        a[i[0],i[1]]=1
        a[i[1],i[0]]=1 # symmetry
    a=np.array(a.astype(int))
    g=nx.from_numpy_matrix(a)
    return g, a

# constrained random walk
# generate random walk on a that results in observed x
def genZfromXG(x,a):
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


# Draw graph
def drawG(g,save=False,display=True):
    pos=nx.spring_layout(g)
    nx.draw_networkx(g,pos,with_labels=True)
#    nx.draw_networkx_labels(g,pos,font_size=12)
#    for node in range(numnodes):                    # sometimes the above doesn't work
#        plt.annotate(str(node), xy=pos[node])       # here's a workaround
    plt.title(x)
    plt.axis('off')
    if save==True:
        plt.savefig('temp.png')
    if display==True:
        plt.show()

def changes(theta):
    Zs=[genZfromX(x, theta) for i in range(1000)]
    As=[genGfromZ(z)[1] for z in Zs]
    counts=[np.bincount(np.array(i-a).flatten()+1) for i in As] # [missing links, correct links, added links]
    missing=[i[0] for i in counts]
    correct=[i[1] for i in counts]
    added  =[i[2] for i in counts]
    return [np.mean(missing), np.mean(correct), np.mean(added)]

if __name__ == "__main__":
    numnodes=20     # number of nodes in graph
    numlinks=4      # initial number of edges per node (must be even)
    probRewire=.75  # probability of re-wiring an edge
    
    theta=.5        # probability of hiding node when generating z from x (rho function)
    
    # Generate small-world graph
    g,a=genG(numnodes,numlinks,probRewire) 
    
    # Generate fake participant data
    x=genX(a)
    
    Zs=[genZfromX(x,theta) for i in range(1000)]
    As=[genGfromZ(z)[1] for z in Zs]
    costs=[sum(sum(np.array(abs(i-a)))) for i in As]

    est_costs=[]
    q=1
    for graph in As:
        zGs=[genZfromXG(x,graph) for i in range(100)]
        probG=sum([logprobZ(i,graph) for i in zGs])/100
        est_costs.append(probG)
        print q
        q=q+1
