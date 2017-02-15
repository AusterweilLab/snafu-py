# Results of previous simulation were hard to analyze since the graph was changing size as number of lists increased
# This file estimates the number of Misses for each numlists/listlength combo to impute into data

import numpy as np
import rw
import math
import random
import networkx as nx
import pygraphviz

usf,items=rw.read_csv('USF_animal_subset.csv')
usfg=nx.to_networkx_graph(usf)
numnodes=len(items)
usfsize=160

misses=[]

f=open('usf_sims_misses.csv','a', 0)                # write/append to file with no buffering

for numlists in range(2,18):
    for listlength in [15,30,50,70]:
        misses=0
        crs=0
        for gnum in range(100):  # how many samples for each numlists/listlength combo
            print numlists, listlength, gnum
            # generate toy lists
            Xs=[rw.genX(usfg)[0:listlength] for i in range(numlists)]

            itemsinXs=np.unique(rw.flatten_list(Xs))    # list of items that appear in toy data

            notinx=[]       # nodes not in trimmed X
            for i in range(usfsize):
                if i not in itemsinXs:
                    notinx.append(i)

            miss=sum([len([j for j in usfg.neighbors(i) if j > i]) for i in notinx])
            misses=misses+miss

            cr=sum([(usfsize-i-1) for i in range(len(notinx))])
            cr=cr-miss
            crs=crs+cr
        
        misses=misses/100.0 # 100 = gnum
        crs=crs/100.0

        towrite = str(numlists) + ',' + str(listlength) + ',' + str(misses) + ',' + str(crs)
        print towrite

        f.write(towrite)
        f.write('\n')

    







