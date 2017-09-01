import rw
import networkx as nx
import numpy as np
import pickle
import sys

usf_graph, usf_items = rw.read_csv("./snet/USF_full.snet")
animal_graph, animal_items = rw.read_csv("./snet/USF_animal_subset.snet")
print "graph loaded"
triads=[["horse","zebra","elephant"], ["dog","cat","zebra"], ["lion","cat","snail"], ["bird","fly","goat"]]
for triad in triads:
    #rw.triadicMonteCarlo(usf_graph, usf_items, triad, jumpval=.25) 
    #rw.triadicMonteCarlo(animal_graph, animal_items, triad, numsims=100000, jumpval=0.25)
    rw.triadicComparison(animal_graph, animal_items, triad, jumpval=0.25)
    print 

fout=open('tmp.csv','w')

with open('dedyne_triadic.csv','r') as triadic_data:
    triadic_data.readline()
    for line in triadic_data:
        line = line.rstrip().split(',')
        triad=[line[0], line[1], line[2]]
        res = rw.triadicMonteCarlo(usf_graph, usf_items, triad,jumpval=0.5)
        #res = rw.triadicComparison(usf_graph, usf_items, triad)
        lineout = ",".join(line) + "," + ",".join([str(i) for i in res]) + "\n"
        fout.write(lineout)
        print lineout
fout.close()
