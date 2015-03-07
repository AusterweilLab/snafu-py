import scipy
from scipy import stats
import pickle
from itertools import product

execfile('rw11.py')

numnodes_set=range(20,65,5)
numx_set=range(1,6)

numnodes=20                              # number of nodes in graph
numlinks=4                               # initial number of edges per node (must be even)
probRewire=.2                            # probability of re-wiring an edge
numedges=numlinks*(numnodes/2.0)         # number of edges in graph

theta=.5                # probability of hiding node when generating z from x (rho function)
numx=2                  # number of Xs to generate
numsamples=100          # number of sample z's to estimate likelihood

# Generate small-world graph
g,a=genG(numnodes,numlinks,probRewire) 

# Generate fake participant data
Xs=[genX(g) for i in range(numx)]

