import numpy as np
import sys
sys.path.append('./rw')
import rw
import math
import random
import networkx as nx
import pygraphviz

# PARAMETERS OF TOY SMALL-WORLD GRAPH
numnodes=15                           # number of nodes in graph
numlinks=4                            # initial number of edges per node (must be even)
probRewire=.3                         # probability of re-wiring an edge
numedges=numnodes*(numlinks/2)        # number of edges in graph
numx=3                                # How many observed lists?
trim=.7                               # ~ What proportion of graph does each list cover?

# PARAMETERS FOR RECONTRUCTING GRAPH
jeff=0.5                              # 1-IRT weight
beta=(1/1.1)                            # for gamma distribution when generating IRTs from hidden nodes

# WRITE DATA
numgraphs=100                         # number of toy graphs to generate/reconstruct
outfile='sim_results_1.csv'

# optionally, pass a methods argument
# default is methods=['fe','rw','invite','inviteirt'] 
rw.toyBatch(numgraphs, numnodes, numlinks, probRewire, numx, trim, jeff, beta, outfile,start_seed=1,methods=['inviteirt'])
