# NOTE: requires space delimited matrix
# Matlab outputs csv

import numpy as np
import networkx as nx
from rw.rw import *

f=open('animalnet.csv')
animal=np.loadtxt(f)
animal=nx.from_numpy_matrix(animal)
animal=list(nx.connected_component_subgraphs(animal))[0] # largest connected subgraph

f=open('animalwords.csv')
names=f.read().split()
names=[names[i] for i in nx.nodes(animal)] # remove unconnected animals

labels={}
for i,j in enumerate(names):
    labels[i]=j

animal=nx.convert_node_labels_to_integers(animal) # re-label nodes to consecutive integers

pos=nx.spring_layout(animal)
nx.draw_networkx_nodes(animal,pos)
nx.draw_networkx_edges(animal,pos)
nx.draw_networkx_labels(animal,pos,labels)

g=animal
a=nx.to_numpy_matrix(g)
numnodes=nx.number_of_nodes(g)
numedges=nx.number_of_edges(g)

## now generate fake data from animal network
numx=3
Xs=[genX(g) for i in range(numx)]

## now generate fake graphs from fake data
numsamples=1 # to compute LL
thetas=[i*.1 for i in range(10)]
numgraphs=10
for theta in thetas:
gs=genGraphs(numgraphs,theta,Xs,len(a)) # gen graphs
gs.append(a)
c=computeCosts(gs,Xs,a,numsamples)      # compute costs
tc=c[0]
ec=c[1]
gs=[nx.from_numpy_matrix(g) for g in gs]    
spl=[nx.average_shortest_path_length(g) for g in gs]    # shortest path length
ac=[nx.average_clustering(g) for g in gs]               # average clustering coefficient


    

# average clustering
# shortest path length

#plt.show()
