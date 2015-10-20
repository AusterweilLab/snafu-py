import numpy as np
import rw.rw as rw
import math
import random
from datetime import datetime
import networkx as nx
#import graphviz
import pygraphviz

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
    true=[]
     
    maxlen=numnodes # maybe?
    for it, graph in enumerate(graphs):
        tmp=rw.probX(Xs,graph,expected_irts,numnodes,maxlen,jeff)
        #tmp=rw.probXnoIRT(Xs,graph,numnodes)
        ours.append(tmp)

        true.append(rw.cost(graph,a))  # only on toy networks
    
    maxvals=maxs(ours,numkeep)
    #print maxvals
    maxpos=[ours.index(i) for i in maxvals]
    maxgraphs=[]
    for i in maxpos:
        maxgraphs.append(graphs[i])
    print "MAX: ", max(ours), "COST: ", rw.cost(graphs[ours.index(max(ours))],a)/2 # only on toy networks
    return maxgraphs, max(ours)

# treat Xs as if there are no hidden nodes and connect all observations to form graph
def noHidden(Xs, numnodes):
    a=np.zeros((numnodes,numnodes))
    for x in Xs:
        for i in range(len(x)-1):
            a[x[i]][x[i+1]]=1
            a[x[i+1]][x[i]]=1 # symmetry
    a=np.array(a.astype(int))
    return a

# read Xs in from user files
def readX(subj,category):
    if type(subj) == str:
        subj=[subj]
    game=-1
    cursubj=-1
    Xs=[]
    irts=[]
    items={}
    idx=0
    with open('exp/results_cleaned.csv') as f:
        for line in f:
            row=line.split(',')
            if (row[0] in subj) & (row[2] == category):
                if (row[1] != game) or (row[0] != cursubj):
                    Xs.append([])
                    irts.append([])
                    game=row[1]
                    cursubj=row[0]
                item=row[3]
                irt=row[4]
                if item not in items.values():
                    items[idx]=item
                    idx += 1
                itemval=items.values().index(item)
                if itemval not in Xs[-1]:   # ignore any duplicates in same list resulting from spelling corrections
                    Xs[-1].append(itemval)
                    irts[-1].append(int(irt)/1000.0)
    numnodes = len(items)
    return Xs, items, irts, numnodes

def drawDot(g, filename, labels={}):
    if type(g) == np.ndarray:
        g=nx.to_networkx_graph(g)
    if labels != {}:
        nx.relabel_nodes(g, labels, copy=False)
    nx.drawing.write_dot(g, filename)
       
numnodes=55                           # number of nodes in graph
numlinks=4                            # initial n/fivebeumber of edges per node (must be even)
probRewire=.2                         # probability of re-wiring an edge
numedges=numnodes*(numlinks/2)        # number of edges in graph
numx=3
trim=1

theta=.5                # probability of hiding node when generating z from x (rho function)
numgraphs=100
maxlen=20               # no closed form, number of times to sum over
jeff = .5
numperseed=50
nodestotweak=[1,1,1,2,3,4,5,6,7,8,9,10]
numkeep=3
beta=1.4             # for gamma distribution when generating IRTs from hidden nodes
offset=2           # for generating IRTs from hidden nodes

#graph_seed=None
#x_seed=None
graph_seed=65      # make sure same toy network is generated every time
x_seed=65          # make sure same Xs are generated every time

# toy data
g,a=rw.genG(numnodes,numlinks,probRewire,seed=graph_seed)
Xs=[rw.genX(g, seed=x_seed+i) for i in range(numx)]
[Xs,g,a,numnodes]=rw.trimX(trim,Xs,g,a,numnodes)
expected_irts=rw.expectedIRT(Xs,a,numnodes, beta, offset)

starttime=str(datetime.now())

# gen candidate graphs
graphs=rw.genGraphs(numgraphs,theta,Xs,numnodes)
graphs.append(noHidden(Xs,numnodes)) # probably best starting graph
#allnodes=[(i,j) for i in range(len(a)) for j in range(len(a)) if (i!=j) and (i>j)]

max_converge=25
converge=0
oldbestval=0
fivebest=[]
log=[]

log.append(starttime)
while converge < max_converge:
    graphs, bestval=xBest(graphs,numkeep)
    log.append(bestval)
    if bestval == oldbestval:
        converge += 1
    else:
        fivebest.append(graphs[0]) # TODO: make sure it's saving the 'best' of the returned graphs
        if len(fivebest) > 5:
            fivebest.pop(0)
        converge = 0
        oldbestval = bestval
    graphs=genFromSeeds(graphs,numperseed,nodestotweak)

gs=[nx.to_networkx_graph(i) for i in fivebest]

# record endtime
endtime=str(datetime.now())
log.append(endtime)
