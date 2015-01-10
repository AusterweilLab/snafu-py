#!/usr/bin/python

# V6

import networkx as nx
import numpy as np
import random
import operator
import math
import matplotlib.pyplot as plt
import time
import scipy.stats as ss

# random walk given an adjacency matrix that hits every node; returns a list of tuples
def random_walk(g,s=None):
    if s is None:
        s=random.choice(range(len(a)))
    walk=[]
    unused_nodes=set(range(len(a)))
    unused_nodes.remove(s)
    while len(unused_nodes) > 0:
        p=s
        s=random.choice([x for x in nx.all_neighbors(g,s)]) # follow random edge
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

def genX(g):
    return observed_walk(random_walk(g))

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
    logProb=sum(logProbList)
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
    #numnodes=len(observed_walk(walk))
    a=np.zeros((numnodes,numnodes))
    for i in set(walk):
        a[i[0],i[1]]=1
        a[i[1],i[0]]=1 # symmetry
    a=np.array(a.astype(int))
    #g=nx.from_numpy_matrix(a)      # too slow, not necessary
    return a

# helper function for optimization
def timer(times):
    t1=time.time()
    for i in range(times):
        genZfromXG(x,a) # insert function to time here
    t2=time.time()
    return t2-t1

# constrained random walk
# generate random walk on a that results in observed x
def genZfromXG(x,a):
    # restrict walk to only the next node in x OR previously visited nodes
    possibles=np.zeros(len(a),dtype=int)
    possibles[[x[0],x[1],x[2]]] = 1
    walk=[(x[0], x[1])]      # add first two Xs to random walk
    pos=2                    # only allow items up to pos
    newa=np.copy(a)                          ## 
    for i in range(len(newa)):               ## these lines sped up code a lot but look a bit sloppy
        newa[i][np.where(possibles==0)[0]]=0 ##
    while len(x[pos:]) > 0:
        p=walk[-1][1]
        pruned_links=np.nonzero(newa[p])[0] ##
        s=random.choice(pruned_links)
        walk.append((p,s))
        if s in x[pos:]:
            pos=pos+1
            if len(x[pos:]) > 0:
                possibles[x[pos]] = 1
                newa=np.copy(a) ##
                for i in range(len(newa)): ##
                    newa[i][np.where(possibles==0)[0]]=0 ##
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

def prior(a):
    g_sm=nx.from_numpy_matrix(a)
    c_sm=nx.average_clustering(g_sm)
    l_sm=nx.average_shortest_path_length(g_sm)
    c_rand= (numedges*2.0)/(numnodes*(numnodes-1))     # same as edge density for a random graph
    l_rand= math.log(numnodes)/math.log(2*numlinks)  # see humphries & gurney (2006) eq 11
    #l_rand= (math.log(numnodes)-0.5772)/(math.log(2*numlinks)) + .5 # alternative from fronczak, fronczak & holyst (2004)
    s=(c_sm/c_rand)/(l_sm/l_rand)
    return s

# function for investigating role of theta. not really necessary.
def changes(theta):
    Zs=[genZfromX(x, theta) for i in range(1000)]
    As=[genGfromZ(z) for z in Zs]
    counts=[np.bincount(np.array(i-a).flatten()+1) for i in As] # [missing links, correct links, added links]
    missing=[i[0] for i in counts]
    correct=[i[1] for i in counts]
    added  =[i[2] for i in counts]
    return [np.mean(missing), np.mean(correct), np.mean(added)]

if __name__ == "__main__":
    numnodes=20             # number of nodes in graph
    numlinks=4              # initial number of edges per node (must be even)
    probRewire=.2          # probability of re-wiring an edge
    numedges=numlinks*10    # number of edges in graph
    
    theta=.5                # probability of hiding node when generating z from x (rho function)
    
    # Generate small-world graph
    g,a=genG(numnodes,numlinks,probRewire) 
    
    # Generate fake participant data
    x=genX(g)
    
    Zs=[genZfromX(x,theta) for i in range(1000)]
    As=[genGfromZ(z) for z in Zs]
    costs=[sum(sum(np.array(abs(i-a)))) for i in As]

    #params=(8.0780635773023164, -0.25699137175238229, 0.090526785535404941)
    #gamma=ss.gamma(params[0],loc=params[1],scale=params[2])
    #gamma.pdf(x)

    est_costs=[]
    #est_costs2=[]
    for q, graph in enumerate(As):
        zGs=[genZfromXG(x,graph) for i in range(100)] # look how correlation changes with number of zs
        probG=sum([logprobZ(i,graph) for i in zGs])/100
        est_costs.append(probG)
        #probG=probG + math.log(gamma.pdf(prior(graph)))
        #est_costs2.append(probG)
        print q

