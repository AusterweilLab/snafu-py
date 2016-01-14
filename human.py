import numpy as np
import rw.rw as rw
import math
import random
from datetime import datetime
import networkx as nx
#import graphviz
import pygraphviz
from itertools import *
import random

allsubs=["S101","S102","S103","S104","S105","S106","S107","S108","S109","S110",
         "S111","S112","S113","S114","S115","S116","S117","S118","S119","S120"]

# free parameters
jeff=0.9           # 1-IRT weight
beta=1.1             # for gamma distribution when generating IRTs from hidden nodes

subj="S111"
category="animals"
Xs, items, irts, numnodes=rw.readX(subj,category,'exp/results_cleaned.csv')

# Find best graph!
best_graph, bestval=rw.findBestGraph(Xs, irts, jeff, beta)

# convert best graph to networkX graph, add labels, write to file
g=nx.to_networkx_graph(best_graph)
nx.relabel_nodes(g, items, copy=False)
nx.write_dot(g,subj+".dot")

# write lists to file
#with open(subj+'_lists.csv','w') as f:
#    for i, x in enumerate(Xs):
#        for item in x:
#            print>>f, str(i)+","+items[item]

# items removed
#for i in np.argwhere(graph5-oldgraph==-1):
#    if i[0]>i[1]:   # skip duplicates
#        print items[i[0]],items[i[1]]

# items added
#for i in np.argwhere(graph5-oldgraph==1):
#    if i[0]>i[1]:   # skip duplicates
#        print items[i[0]],items[i[1]]


# MATLAB BEAGLE example
# [t,index]=ismember('ZEBU',BEAGLE_labels)
# BEAGLE_sim(100,100)
