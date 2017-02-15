# Generate toy data from USF animal subset network
# Re-construct network from toy data
# Check to see whether reconstructed graph produced correct/incorrect edges

import numpy as np
import rw
import math
import random
import networkx as nx
import pygraphviz

usf,items=rw.read_csv('USF_animal_subset.csv')
usfg=nx.to_networkx_graph(usf)
numnodes=len(items)

sdt_rws=[]
sdt_uis=[]

f=open('usf_sims.csv','a', 0)                # write/append to file with no buffering

for numlists in range(2,50):
    for listlength in [15,30,50,70]:
        for gnum in range(10):  # how many samples for each numlists/listlength combo
            print numlists, listlength, gnum
            # generate toy lists
            #numlists=5
            #listlength=50   # must be <= numnodes
            Xs=[rw.genX(usfg)[0:listlength] for i in range(numlists)]

            itemsinXs=np.unique(rw.flatten_list(Xs))    # list of items that appear in toy data

            # reconstruct graph
            rw_graph=rw.noHidden(Xs, numnodes)
            ui_graph, bestval=rw.findBestGraph(Xs, numnodes=numnodes)

            # remove nodes not in X from all graphs before comparison
            rw_graph=rw_graph[itemsinXs,]
            rw_graph=rw_graph[:,itemsinXs]
            ui_graph=ui_graph[itemsinXs,]
            ui_graph=ui_graph[:,itemsinXs]
            usfcopy=np.copy(usf)
            usfcopy=usfcopy[itemsinXs,]
            usfcopy=usfcopy[:,itemsinXs]

            sdt_rw = rw.costSDT(rw_graph, usfcopy)
            sdt_ui = rw.costSDT(ui_graph, usfcopy)

            sdt_rws.append(sdt_rw)
            sdt_uis.append(sdt_ui)

            towrite = rw.flatten_list([sdt_rw,sdt_ui])
            towrite = ','.join([str(i) for i in towrite])
            towrite = str(numlists) + ',' + str(listlength) + ',' + str(gnum) + ',' + towrite

            f.write(towrite)
            f.write('\n')

    






