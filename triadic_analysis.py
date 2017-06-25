import rw
import networkx as nx
import numpy as np
import pickle
import sys

usf_graph, usf_items = rw.read_csv("./snet/dedyne.snet")
print "graph loaded"

fout=open('triaidc.csv','w')

with open('dedyne_exp3.csv','r') as triadic_data:
    triadic_data.readline()
    for line in triadic_data:
        line = line.rstrip().split(',')
        triad=[line[0], line[1], line[2]]
        res = rw.triadicMonteCarlo(usf_graph, usf_items, triad)
        fout.write(",".join(line) + "," + ",".join([str(i) for i in res]) + "\n")
fh.close()
