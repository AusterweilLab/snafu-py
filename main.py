import numpy as np
import rw.rw as rw
import math
import random
from datetime import datetime
import networkx as nx
#import graphviz
import pygraphviz

# treat Xs as if there are no hidden nodes and connect all observations to form graph

# read Xs in from user files
def readX(subj,category):
    if type(subj) == str:
        subj=[subj]
    game=-1
    cursubj=-1
    Xs=[]
    irts=[]
    items={}
    idx=0
    with open('exp/results_cleaned.csv') as f:
        for line in f:
            row=line.split(',')
            if (row[0] in subj) & (row[2] == category):
                if (row[1] != game) or (row[0] != cursubj):
                    Xs.append([])
                    irts.append([])
                    game=row[1]
                    cursubj=row[0]
                item=row[3]
                irt=row[4]
                if item not in items.values():
                    items[idx]=item
                    idx += 1
                itemval=items.values().index(item)
                if itemval not in Xs[-1]:   # ignore any duplicates in same list resulting from spelling corrections
                    Xs[-1].append(itemval)
                    irts[-1].append(int(irt)/1000.0)
    numnodes = len(items)
    return Xs, items, irts, numnodes

allsubs=["S101","S102","S103","S104","S105","S106","S107","S108","S109","S110",
         "S111","S112","S113","S114","S115","S116","S117","S118","S119","S120"]

numnodes=30                           # number of nodes in graph
numlinks=4                            # initial n/fivebeumber of edges per node (must be even)
probRewire=.2                         # probability of re-wiring an edge
numedges=numnodes*(numlinks/2)        # number of edges in graph
numx=3
trim=1

theta=.5                # probability of hiding node when generating z from x (rho function)
numgraphs=100
maxlen=20               # no closed form, number of times to sum over
jeff = .5
numperseed=50
edgestotweak=[1,1,1,2,3,4,5,6,7,8,9,10]
numkeep=3
beta=1             # for gamma distribution when generating IRTs from hidden nodes

# record start time

# toy data
g,a=rw.genG(numnodes,numlinks,probRewire) 
Xs=[rw.genX(g) for i in range(numx)]
[Xs,g,a,numnodes]=rw.trimX(trim,Xs,g,a,numnodes)
expected_irts=rw.expectedIRT(Xs,a,numnodes, beta)

subj="S103"
category="animals"
starttime=str(datetime.now())

Xs, items, expected_irts, numnodes=readX(subj,category)

# gen candidate graphs
graphs=rw.genGraphs(numgraphs,theta,Xs,numnodes)
graphs.append(rw.noHidden(Xs,numnodes)) # probably best starting graph
#allnodes=[(i,j) for i in range(len(a)) for j in range(len(a)) if (i!=j) and (i>j)]

max_converge=25
converge=0
oldbestval=0
bestgraphs=[]
log=[]

log.append(starttime)
while converge < max_converge:
    graphs, bestval=rw.graphSearch(graphs,numkeep,Xs,numnodes,maxlen,jeff,expected_irts)
    log.append(bestval)
    if bestval == oldbestval:
        converge += 1
    else:
        bestgraphs.append(graphs[0]) # TODO: make sure it's saving the 'best' of the returned graphs
        if len(bestgraphs) > 5:
            bestgraphs.pop(0)
        converge = 0
        oldbestval = bestval
    graphs=rw.genFromSeeds(graphs,numperseed,edgestotweak)

gs=[nx.to_networkx_graph(i) for i in bestgraphs]

# record endtime
endtime=str(datetime.now())
log.append(endtime)

for i, j in enumerate(gs):
    nx.relabel_nodes(j, items, copy=False)
    nx.write_dot(j,subj+"_"+str(i)+".dot")

# write iterations and start/end time to log
with open(subj+'_log.txt','w') as f:
    for item in log:
        print>>f, item

with open(subj+'_lists.csv','w') as f:
    for i, x in enumerate(Xs):
        for item in x:
            print>>f, str(i)+","+items[item]

# write lists to file
with open(subj+'_lists.csv','w') as f:
    for i, x in enumerate(Xs):
        for item in x:
            print>>f, str(i)+","+items[item]
