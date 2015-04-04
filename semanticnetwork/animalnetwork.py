# NOTE: requires space delimited matrix
# Matlab outputs csv

import numpy as np
import networkx as nx

f=open('semanticnetwork/animalnet.csv')
animal=np.loadtxt(f)
animal=nx.from_numpy_matrix(animal)
animal=list(nx.connected_component_subgraphs(animal))[0] # largest connected subgraph

f=open('semanticnetwork/animalwords.csv')
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

#plt.show()
