import numpy as np
import rw.rw as rw
import math
import random
from datetime import datetime
import networkx as nx
#import graphviz
import pygraphviz


# TOY GRAPH PARAMETERS
numnodes=15                           
numlinks=4                            
probRewire=.4                         
numedges=numnodes*(numlinks/2)        
graph_seed=None                             # make sure same toy network is generated every time

# FAKE DATA PARAMETERS
numx=5
trim=0.7
x_seed=None                                 # make sure same Xs are generated every time

# FITTING PARAMETERS
theta=.5                                  # probability of hiding node when generating z from x (rho function)
numgraphs=100
maxlen=20                                 # no closed form, number of times to sum over
jeff = .9
numperseed=50
edgestotweak=[1,1,1,2,3,4,5,6,7,8,9,10]
numkeep=3
beta=0.9                                  # for gamma distribution when generating IRTs from hidden nodes
offset=2                                  # for generating IRTs from hidden nodes
max_converge=5

# SIMULATION RESULTS
cost_irts=[]
cost_noirts=[]
cost_orig=[]
time_irts=[]
time_noirts=[]
bestval_irts=[]
bestval_noirts=[]
bestval_orig=[]
bestgraph_irts=[]
bestgraph_noirts=[]

# WRITE DATA
outfile='sim_resultsx.csv'

f=open(outfile,'a', 0) # write/append to file with no buffering

for seed_param in range(100):
    for irt_param in range(2):
        graph_seed=seed_param
        x_seed=seed_param

        # toy data
        g,a=rw.genG(numnodes,numlinks,probRewire,seed=graph_seed)
        [Xs,irts]=zip(*[rw.genX(g, seed=x_seed+i,use_irts=1) for i in range(numx)])
        Xs=list(Xs)
        irts=list(irts)
        [Xs,alter_graph]=rw.trimX(trim,Xs,g)
        
        if irt_param:
            irts=rw.stepsToIRT(irts, beta, offset)

        starttime=datetime.now()

        # gen candidate graphs
        graphs=rw.genGraphs(numgraphs,theta,Xs,numnodes)
        graphs.append(rw.noHidden(Xs,numnodes)) # probably best starting graph

        converge=0
        oldbestval=0
        bestgraphs=[]

        while converge < max_converge:
            if irt_param:
                graphs, bestval=rw.graphSearch(graphs,numkeep,Xs,numnodes,maxlen,jeff,irts)
            else:
                graphs, bestval=rw.graphSearch(graphs,numkeep,Xs,numnodes,maxlen,jeff)
            if bestval == oldbestval:
                converge += 1
            else:
                bestgraphs.append(graphs[0])
                if len(bestgraphs) > 5:
                    bestgraphs.pop(0)
                converge = 0
                oldbestval = bestval
            graphs=rw.genFromSeeds(graphs,numperseed,edgestotweak)

        # record endtime
        elapsedtime=str(datetime.now()-starttime)

        if irt_param:
            cost_irts.append(rw.cost(bestgraphs[-1],a))
            time_irts.append(elapsedtime)
            bestval_irts.append(bestval)
            bestgraph_irts.append(rw.graphToHash(bestgraphs[-1]))
        else:
            cost_noirts.append(rw.cost(bestgraphs[-1],a))
            time_noirts.append(elapsedtime)
            bestval_noirts.append(bestval)
            bestgraph_noirts.append(rw.graphToHash(bestgraphs[-1]))

            orig=rw.noHidden(Xs,numnodes)
            cost_orig.append(rw.cost(orig,a))
            bestval_orig.append(rw.probXnoIRT(Xs, orig, numnodes))
            
        #gs=[nx.to_networkx_graph(i) for i in bestgraphs]

    # log stuff here
    f.write(str(max_converge) + ',' +
            str(theta) + ',' +
            str(numgraphs) + ',' +
            str(edgestotweak).replace(',','') + ',' +
            str(numkeep) + ',' + 
            str(maxlen) + ',' +
            str(jeff) + ',' +
            str(numperseed) + ',' +
            str(beta) + ',' +
            str(offset) + ',' +
            str(numnodes) + ',' +
            str(numlinks) + ',' +
            str(probRewire) + ',' +
            str(numedges) + ',' +
            str(graph_seed) + ',' +
            str(numx) + ',' +
            str(trim) + ',' +
            str(x_seed) + ',' +
            str(cost_orig[-1]) + ',' +
            str(cost_irts[-1]) + ',' +
            str(cost_noirts[-1]) + ',' +
            str(time_irts[-1]) + ',' +
            str(time_noirts[-1]) + ',' +
            str(bestgraph_irts[-1]) + ',' +
            str(bestgraph_noirts[-1]) + ',' +
            str(alter_graph) + '\n')
            # include hit, miss, fa, cr for each of 3 graph types
            # include LL

    print "FINAL COSTS:", cost_orig[-1], cost_noirts[-1], cost_irts[-1]

f.close()
