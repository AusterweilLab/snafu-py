# script to generate USF' network; same as USF network except creates a direct edge between any two animals where
# indirect edge exists (path through exclusively non-animals)
# constraint: path length of 3

import sys
sys.path.append('./rw')
import rw
import numpy as np
import networkx as nx

usf,items=rw.read_csv('USF_full.csv')
usfprime=np.copy(usf) 

# animal list
with open('USFanimals.csv','r') as f:
    animals = f.read()
animals=animals.split('\n')
animals.pop()

animalidx=[items.values().index(i) for i in animals]
nonanimalidx=sorted([i for i in range(5018) if i not in animalidx],reverse=True)

G = nx.to_networkx_graph(usf)

cutoff=3

for node1 in items:
    print node1
    for node2 in items:
        if node2 > node1:       # loop through all pairs (could be optimized)
            if (items[node1] in animals) and (items[node2] in animals): # if both nodes are animals
                if usf[node1,node2] == 0.0: # if no direct path exists
                    # check for indirect path and add edge
                    paths=nx.all_simple_paths(G,source=node1,target=node2,cutoff=cutoff)
                    for path in paths:  # check all indirect paths (w/ cutoff)
                        items_to_check=[items[i] for i in path[1:-1]]
                        is_animal=[i in animals for i in items_to_check]
                        if any(qq)==False:      # there are no animals along indirect path
                            usfprime[node1,node2]=1.0   # therefore make a direct path
                            usfprime[node2,node1]=1.0
                            break

# remove non-animal nodes
usfprime=usfprime[animalidx,]
usfprime=usfprime[:,animalidx]

# relabeling nodes... this is way more complicated than it should/could be...

idxmap=dict(enumerate(animalidx))
for i in idxmap:
    idxmap[i]=items[idxmap[i]]


GG=nx.to_networkx_graph(usfprime)
primeedges=GG.edges()

for pairnum, pair in enumerate(primeedges):
    primeedges[pairnum] = idxmap[pair[0]],idxmap[pair[1]]

# write to file
fh = open('USF_prime.csv','w')
for i in primeedges:
    print >>fh, items[i[0]] + "," + items[i[1]]


