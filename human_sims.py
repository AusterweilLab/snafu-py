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

allsubs=["S101","S102","S103","S104","S105","S106","S107","S108","S109","S110",
         "S111","S112","S113","S114","S115","S116","S117","S118","S119","S120"]

# free parameters
beta=1.1             # for gamma distribution when generating IRTs from hidden nodes

subj="S120"
category="vegetables"

for subj in allsubs:
    Xs, items, irts, numnodes=rw.readX(subj,category,'exp/results_cleaned.csv')

    # Find best graphs!
    rw_graph = rw.noHidden(Xs,numnodes)
    rw_g=nx.to_networkx_graph(rw_graph)
    nx.relabel_nodes(rw_g, items, copy=False)
    nx.write_dot(rw_g,subj+"_rw.dot")

    invite_graph, val=rw.findBestGraph(Xs)
    invite_g=nx.to_networkx_graph(invite_graph)
    nx.relabel_nodes(invite_g, items, copy=False)
    nx.write_dot(invite_g,subj+"_invite.dot")

    jeff=0.9           # IRT weight
    irt_graph, val=rw.findBestGraph(Xs, irts, jeff, beta)
    irt_g=nx.to_networkx_graph(irt_graph)
    nx.relabel_nodes(irt_g, items, copy=False)
    nx.write_dot(irt_g,subj+"_irt9.dot")

    jeff=0.5
    irt_graph, val=rw.findBestGraph(Xs, irts, jeff, beta)
    irt_g=nx.to_networkx_graph(irt_graph)
    nx.relabel_nodes(irt_g, items, copy=False)
    nx.write_dot(irt_g,subj+"_irt5.dot")
