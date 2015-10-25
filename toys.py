import numpy as np
import rw.rw as rw
import math
import random
from datetime import datetime
import networkx as nx
import pygraphviz

# PARAMETERS OF TOY SMALL-WORLD GRAPH
numnodes=55                           # number of nodes in graph
numlinks=4                            # initial number of edges per node (must be even)
probRewire=.2                         # probability of re-wiring an edge
numedges=numnodes*(numlinks/2)        # number of edges in graph
numx=4                                # How many observed lists?
trim=.7                               # ~ What proportion of graph does each list cover?

# PARAMETERS FOR RECONTRUCTING GRAPH
theta=.5                # probability of hiding node when generating z from x (rho function)
numgraphs=10            # number of seed graphs, excluding "no-censor" graph
maxlen=20               # no closed form, number of times to sum over
jeff = .5               # weighting of IRTs
numperseed=50           # number of candidate graphs to generate from each seed
numkeep=3               # how many graphs to keep as seeds for next iteration
beta=1.4                # for gamma distribution when generating IRTs from hidden nodes
offset=2                # for generating IRTs from hidden nodes
max_converge=25         # stop search when graph hasn't changed for this many iterations
edgestotweak=[1,1,1,2,3,4,5,6,7,8,9,10] # how many edges to flip in search?

#graph_seed=None
#x_seed=None
graph_seed=65      # make sure same toy network is generated every time
x_seed=65          # make sure same Xs are generated every time

# toy data
g,a=rw.genG(numnodes,numlinks,probRewire,seed=graph_seed)
Xs=[rw.genX(g, seed=x_seed+i) for i in range(numx)]
[Xs,g,a,numnodes]=rw.trimX(trim,Xs,g,a,numnodes)
expected_irts=rw.expectedIRT(Xs, a, numnodes, beta, offset)

starttime=str(datetime.now())

# gen candidate graphs
graphs=rw.genGraphs(numgraphs,theta,Xs,numnodes)
graphs.append(rw.noHidden(Xs,numnodes)) # probably best starting graph

converge=0
oldbestval=0
bestgraphs=[]
log=[]

log.append(starttime)
while converge < max_converge:
    graphs, bestval=rw.graphSearch(graphs,numkeep,Xs,numnodes,maxlen,jeff,expected_irts)
    log.append(bestval)
    if bestval == oldbestval:
        converge += 1
    else:
        bestgraphs.append(graphs[0]) # TODO: make sure it's saving the 'best' of the returned graphs
        if len(bestgraphs) > 5:
            bestgraphs.pop(0)
        converge = 0
        oldbestval = bestval
    graphs=rw.genFromSeeds(graphs,numperseed,edgestotweak)

gs=[nx.to_networkx_graph(i) for i in bestgraphs]

# record endtime
endtime=str(datetime.now())
log.append(endtime)
