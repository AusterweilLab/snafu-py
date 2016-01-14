import numpy as np
import rw.rw as rw
import math
import random
from datetime import datetime
import networkx as nx
import pygraphviz

# PARAMETERS OF TOY SMALL-WORLD GRAPH
numnodes=15                           # number of nodes in graph
numlinks=4                            # initial number of edges per node (must be even)
probRewire=.3                         # probability of re-wiring an edge
numedges=numnodes*(numlinks/2)        # number of edges in graph
numx=5                                # How many observed lists?
trim=.7                               # ~ What proportion of graph does each list cover?

# PARAMETERS FOR RECONTRUCTING GRAPH
jeff=0.9           # 1-IRT weight
beta=1.1             # for gamma distribution when generating IRTs from hidden nodes

#graph_seed=None
#x_seed=None
graph_seed=65      # make sure same toy network is generated every time
x_seed=65          # make sure same Xs are generated every time

# toy data
g,a=rw.genG(numnodes,numlinks,probRewire,seed=graph_seed)
[Xs,steps]=zip(*[rw.genX(g, seed=x_seed+i,use_irts=1) for i in range(numx)])
Xs=list(Xs)
irts=list(irts)
[Xs,alter_graph]=rw.trimX(trim,Xs,g)
irts=rw.stepsToIRT(steps, beta, seed=x_seed) # TODO: also chop irts!

# Find best graph!
best_graph, bestval=rw.findBestGraph(Xs, irts, jeff, beta)

# convert best graph to networkX graph, write to file
g=nx.to_networkx_graph(best_graph)
nx.write_dot(g,subj+".dot")

