import numpy as np
import rw.rw as rw
import math
import random
from datetime import datetime
import networkx as nx
#import graphviz
import pygraphviz

numnodes=25                           # number of nodes in graph
numlinks=4                            # initial number of edges per node (must be even)
probRewire=.2                         # probability of re-wiring an edge
numedges=numnodes*(numlinks/2)        # number of edges in graph
numx=3
trim=1

theta=.5                # probability of hiding node when generating z from x (rho function)
numgraphs=100
maxlen=20               # no closed form, number of times to sum over
jeff = .5
numperseed=50
edgestotweak=[1,1,1,2,3,4,5,6,7,8,9,10]
numkeep=3
beta=0.9             # for gamma distribution when generating IRTs from hidden nodes
offset=2             # for generating IRTs from hidden nodes
#graph_seed=None
#x_seed=None
graph_seed=65      # make sure same toy network is generated every time
x_seed=65          # make sure same Xs are generated every time

cost_irts=[]
cost_noirts=[]
cost_orig=[]

for seed_param in range(50):
    for irt_param in range(2):
        graph_seed=seed_param
        x_seed=seed_param

        # toy data
        g,a=rw.genG(numnodes,numlinks,probRewire,seed=graph_seed)
        Xs=[rw.genX(g, seed=x_seed+i) for i in range(numx)]
        [Xs,g,a,numnodes]=rw.trimX(trim,Xs,g,a,numnodes)
        expected_irts=rw.expectedIRT(Xs,a,numnodes, beta, offset)

        starttime=str(datetime.now())

        # gen candidate graphs
        graphs=rw.genGraphs(numgraphs,theta,Xs,numnodes)
        graphs.append(rw.noHidden(Xs,numnodes)) # probably best starting graph
        #allnodes=[(i,j) for i in range(len(a)) for j in range(len(a)) if (i!=j) and (i>j)]

        max_converge=5
        converge=0
        oldbestval=0
        bestgraphs=[]
        log=[]

        log.append(starttime)
        while converge < max_converge:
            if irt_param==1:
                graphs, bestval=rw.graphSearch(graphs,numkeep,Xs,numnodes,maxlen,jeff,expected_irts)
            else:
                graphs, bestval=rw.graphSearch(graphs,numkeep,Xs,numnodes,maxlen,jeff)
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

        if irt_param:
            cost_irts.append(rw.cost(bestgraphs[-1],a)/2)
        else:
            cost_noirts.append(rw.cost(bestgraphs[-1],a)/2)
            cost_orig.append(rw.cost(rw.noHidden(Xs,numnodes),a)/2)

        gs=[nx.to_networkx_graph(i) for i in bestgraphs]

        # record endtime
        endtime=str(datetime.now())
        log.append(endtime)
    print "FINAL COSTS:", cost_orig[-1], cost_noirts[-1], cost_irts[-1]
