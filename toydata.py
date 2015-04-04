import scipy
from scipy import stats
import pickle
from itertools import product
import csv
import time
from rw.rw import *

numsamples=100          # number of sample z's to estimate likelihood
numlinks=4              # initial number of edges per node (must be even)
probRewire=.2           # probability of re-wiring an edge
theta=.5                # probability of hiding node when generating z from x (rho function)

numnodes_set=range(20,65,5)
numx_set=range(1,5)
trim_set=[i/100.0 for i in range(30,110,10)] # proportion of graph used before trimming
startnode_set=["random","fixed"] # always start at 0 node, or always start at random node

numgraphs=100            # calculate rank and spearman on how many graphs per parameter set?

filename='toydata.csv'
picklefile='toydata.pickle'

loop=0 # iterate for pickle
dat={}

with open(filename,'w') as f:
    csvfile=csv.writer(f)
    for numnodes, numx, trim, startnode in product(numnodes_set, numx_set, trim_set, startnode_set):
        dat[loop]={}
        starttime=time.time()
        metric=[probRewire, numnodes, numlinks, theta, numx, trim, startnode, numsamples, numgraphs]
        metric.append(int(round(numnodes*trim)))          # number of items in each X

        g,a=genG(numnodes,numlinks,probRewire)
        dat[loop]["true"]=a

        if startnode=="random":
            Xs=[genX(g) for i in range(numx)]
        else:
            sn=random.choice(range(len(g)))
            Xs=[genX(g, sn) for i in range(numx)]

        [Xs,g,a,numnodes]=trimX(trim,Xs,g,a,numnodes)
        dat[loop]["Xs"]=Xs

        As=genGraphs(numgraphs)
        costs, est_costs=computeCosts(As)

        dat[loop]["As"]=As
        
        metric.append(spearman(costs, est_costs))
        metric.append(rank(est_costs)[1])
        metric.append(len(g))                            # actual graph size after trim
            
        g_sm=[nx.from_numpy_matrix(a) for a in As]
        c_sm=[nx.average_clustering(g) for g in g_sm]

        metric.append(spearman(est_costs, c_sm))

        endtime=time.time()
        metric.append(endtime-starttime)

        print metric
        csvfile.writerow(metric)
        loop=loop+1

with open(picklefile,'w') as p:
    pickle.dump(dat,p)

f.close()
p.close()
