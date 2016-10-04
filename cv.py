import numpy as np
import sys
sys.path.append('./rw')
import rw
import math
import random
from datetime import datetime
import networkx as nx
#import graphviz
import pygraphviz
from itertools import *
import random

allsubs=["S1","S2","S3","S4","S5","S7","S8","S9","S10","S11","S12","S13"]

# free parameters
#jeff=0.9             # 1-IRT weight
#beta=1.1             # for gamma distribution when generating IRTs from hidden nodes
 
category="animals"

for subj in allsubs:
    Xs, items, irts, numnodes=rw.readX(subj,category,'data/raw/data_cleaned.csv')

    # Find best graph!
    best_graph, bestval=rw.findBestGraph(Xs)
    best_rw=rw.noHidden(Xs, numnodes)

    g=nx.to_networkx_graph(best_graph)
    g2=nx.to_networkx_graph(best_rw)

    nx.relabel_nodes(g, items, copy=False)
    nx.relabel_nodes(g2, items, copy=False)

    rw.write_csv([g, g2],subj+".csv",subj) # write multiple graphs
