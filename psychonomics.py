# psychonomics AD poster

import rw
import networkx as nx
import numpy as np

usf_graph, usf_items = rw.read_csv("./snet/USF_animal_subset.snet")
usf_graph_nx = nx.from_numpy_matrix(usf_graph)
usf_numnodes = len(usf_items)

all_walks=[]
params=[]
rangestep=100

for cf in [i/float(rangestep) for i in range(0,rangestep)]:
    for ef in [j/float(rangestep) for j in range(0,rangestep)]:
        print cf, ef
        td=rw.Data({
                'numx': 25,
                'trim': 17,
                'censor_fault': cf,
                'emission_fault': ef })
        
        walks = rw.genX(usf_graph_nx, td)[0]
        walks = rw.numToItemLabel(walks, usf_items)
        all_walks.append(walks)
        params.append((cf,ef))

cluster_sizes=[]
cluster_switches=[]
persevs = []
numitems = []

for pnum, walkset in enumerate(all_walks):
    print pnum
    numitems.append(np.mean([len(set(i)) for i in walkset])) # num unique items only
    
    persev = rw.avgNumPerseverations(walkset)
    persevs.append(persev)
    
    # https://stackoverflow.com/questions/480214/how-do-you-remove-duplicates-from-a-list-in-whilst-preserving-order
    def f7(seq):
        seen = set()
        seen_add = seen.add
        return [x for x in seq if not (x in seen or seen_add(x))]

    walkset = [f7(i) for i in walkset]
    cluster_data = [rw.clusterSize(l,scheme="schemes/troyer_hills_zemla_animals.csv") for l in walkset]
    cluster_size = rw.avgClusterSize(cluster_data)
    cluster_switch = rw.avgNumClusterSwitches(cluster_data)
    cluster_sizes.append(cluster_size)
    cluster_switches.append(cluster_switch)

fo=open('simulate_ad_17.csv','w')
header="idx,cf,ef,numitems,persev,clustersize,clusterswitch\n"
fo.write(header)
for pnum, paramset in enumerate(params):
    towrite = str(pnum) + ","
    towrite = towrite + str(paramset[0]) + "," + str(paramset[1]) + ","
    towrite = towrite + str(numitems[pnum]) + ","
    towrite = towrite + str(persevs[pnum]) + ","
    towrite = towrite + str(cluster_sizes[pnum]) + ","
    towrite = towrite + str(cluster_switches[pnum])
    fo.write(towrite + "\n")

fo.close()

