import rw
import networkx as nx
import numpy as np
import pickle
import sys
import numpy as np
import scipy.stats

# load graphs
uinvite_graph, uinvite_items = rw.read_csv("graph_plus_ratings.csv",cols=("node1","node2"),header=True,filters={"s2017_uinvite_hierarchical_zibb_a2_b1_p5": "1"})
#goni_graph, goni_items = rw.read_csv("graph_plus_ratings.csv",cols=("node1","node2"),header=True,filters={"s2017_goni": "1"})

gtom_mat = rw.gtom(uinvite_graph)

fi=open('graph_plus_ratings.csv','r')
fo=open('out.csv','w')

header=fi.readline()
fo.write(header)
for line in fi:
    try:
        item1 = uinvite_items.keys()[uinvite_items.values().index(line.split(',')[1])]
        item2 = uinvite_items.keys()[uinvite_items.values().index(line.split(',')[2])]
        gtom_rating = gtom_mat[item1,item2]
    except:
        gtom_rating = "NA"
    fo.write(line.split('\n')[0] + "," + str(gtom_rating) + "\n")
