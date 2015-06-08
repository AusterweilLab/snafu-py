import numpy as np
import rw.rw as rw
import math
import random

# http://stackoverflow.com/questions/350519/getting-the-lesser-n-elements-of-a-list-in-python
def maxs(items,n):
    maxs = items[:n]
    maxs.sort(reverse=True)
    for i in items[n:]:
        if i > maxs[-1]: 
            maxs.append(i)
            maxs.sort(reverse=True)
            maxs= maxs[:n]
    return maxs

# hill climbing with stochastic search
def genFromSeeds(seedgraphs,numperseed,nodestotweak):
    graphs=seedgraphs[:]
    for i in seedgraphs:
        for j in range(numperseed):
            new=np.copy(i)
            for k in range(random.choice(nodestotweak)):
                rand1=rand2=0
                while (rand1 == rand2):
                    rand1=random.randint(0,len(i)-1)
                    rand2=random.randint(0,len(i)-1)
                new[rand1,rand2]=1-new[rand1,rand2]
                new[rand2,rand1]=1-new[rand2,rand1]
            graphs.append(new)
    return graphs

# hill climbing with stochastic search
def xBest(graphs,numkeep):
    ours=[]
    his=[]
    true=[]

    for it, graph in enumerate(graphs):
        tmp=rw.probX(Xs,graph,expected_irts,numnodes,maxlen,jeff)
        ours.append(tmp)
        true.append(rw.cost(graph,a))
    
    maxvals=maxs(ours,numkeep)
    maxpos=[ours.index(i) for i in maxvals]
    maxgraphs=[]
    for i in maxpos:
        maxgraphs.append(graphs[i])
    print "MAX: ", max(ours), "COST: ", rw.cost(graphs[ours.index(max(ours))],a)
    return maxgraphs

numnodes=53                           # number of nodes in graph
numlinks=4                            # initial number of edges per node (must be even)
probRewire=.2                         # probability of re-wiring an edge
numedges=numnodes*(numlinks/2)        # number of edges in graph

theta=.5                # probability of hiding node when generating z from x (rho function)
numx=1
numsamples=100          # number of sample z's to estimate likelihood
numgraphs=100
trim=1
maxlen=20               # no closed form, number of times to sum over
jeff = .99
maxtokeep=10
numperseed=10
nodestotweak=[1]

g,a=rw.genG(numnodes,numlinks,probRewire) 
Xs=[rw.genX(g) for i in range(numx)]
[Xs,g,a,numnodes]=rw.trimX(trim,Xs,g,a,numnodes)

allnodes=[(i,j) for i in range(len(a)) for j in range(len(a)) if (i!=j) and (i>j)]
expected_irts=rw.expectedHidden(Xs,a,numnodes)
graphs=rw.genGraphs(numgraphs,theta,Xs,numnodes)

while True:
    graphs=xBest(graphs,1)
    graphs=genFromSeeds(graphs,100,nodestotweak)
