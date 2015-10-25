import numpy as np
import rw.rw as rw
import math
import random
from datetime import datetime
import networkx as nx
#import graphviz
import pygraphviz


# TOY GRAPH PARAMETERS
numnodes=25                           
numlinks=4                            
probRewire=.2                         
numedges=numnodes*(numlinks/2)        
graph_seed=65                             # make sure same toy network is generated every time

# FAKE DATA PARAMETERS
numx=3
trim=1
x_seed=65                                 # make sure same Xs are generated every time

# FITTING PARAMETERS
theta=.5                                  # probability of hiding node when generating z from x (rho function)
numgraphs=100
maxlen=20                                 # no closed form, number of times to sum over
jeff = .5
numperseed=50
edgestotweak=[1,1,1,2,3,4,5,6,7,8,9,10]
numkeep=3
beta=0.9                                  # for gamma distribution when generating IRTs from hidden nodes
offset=2                                  # for generating IRTs from hidden nodes

# SIMULATION RESULTS
cost_irts=[]
cost_noirts=[]
cost_orig=[]

for seed_param in range(50):
    for irt_param in range(2):
        graph_seed=seed_param
        x_seed=seed_param

        # toy data
        g,a=rw.genG(numnodes,numlinks,probRewire,seed=graph_seed)
        [Xs,irts]=zip(*[rw.genX(g, seed=x_seed+i,use_irts=1) for i in range(numx)])
        Xs=list(Xs)
        irts=list(irts)
        [Xs,g,a,numnodes]=rw.trimX(trim,Xs,g)
        irts=rw.stepsToIRT(irts, beta, offset)

        starttime=str(datetime.now())

        # gen candidate graphs
        graphs=rw.genGraphs(numgraphs,theta,Xs,numnodes)
        graphs.append(rw.noHidden(Xs,numnodes)) # probably best starting graph

        max_converge=5
        converge=0
        oldbestval=0
        bestgraphs=[]
        log=[]

        log.append(starttime)
        while converge < max_converge:
            if irt_param==1:
                graphs, bestval=rw.graphSearch(graphs,numkeep,Xs,numnodes,maxlen,jeff,irts)
            else:
                graphs, bestval=rw.graphSearch(graphs,numkeep,Xs,numnodes,maxlen,jeff)
            log.append(bestval)
            if bestval == oldbestval:
                converge += 1
            else:
                bestgraphs.append(graphs[0])
                if len(bestgraphs) > 5:
                    bestgraphs.pop(0)
                converge = 0
                oldbestval = bestval
            graphs=rw.genFromSeeds(graphs,numperseed,edgestotweak)

        if irt_param:
            cost_irts.append(rw.cost(bestgraphs[-1],a))
        else:
            cost_noirts.append(rw.cost(bestgraphs[-1],a))
            cost_orig.append(rw.cost(rw.noHidden(Xs,numnodes),a))

        gs=[nx.to_networkx_graph(i) for i in bestgraphs]

        # record endtime
        endtime=str(datetime.now())
        log.append(endtime)
    print "FINAL COSTS:", cost_orig[-1], cost_noirts[-1], cost_irts[-1]
