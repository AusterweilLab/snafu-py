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
types=['rw','invite','irt5','irt7','irt9']
onezero={True: '1', False: '0'}

# write edges from all graphs to file with no buffering
path='human_graphs/converge_1500'
outfile='network_edges.csv'
f=open(outfile,'w', 0)

for sub in allsubs:
    gs=[]

    for typ in types:
        gs.append(nx.read_dot(path+'/'+sub+'_'+typ+'.dot'))

    edges=set(rw.flatten_list([gs[i].edges() for i in range(len(gs))]))
    
    # write ALL edges
    for edge in edges:
        edgelist=""
        for g in gs:
            edgelist=edgelist+","+onezero[g.has_edge(edge[0],edge[1])]
        f.write(sub      + "," +
                edge[0]  + "," +
                edge[1]  + 
                edgelist + "\n")
    
    
