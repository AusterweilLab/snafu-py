# some hard-coding for s2015!
# trying to prove null + what's proper comparison model?

from __future__ import division

import networkx as nx
import rw
import numpy as np
import math
import csv

subs=['S101','S102','S103','S104','S105','S106','S107','S108','S109','S110',
      'S111','S112','S113','S114','S115','S116','S117','S118','S119','S120']

toydata=rw.Toydata({
        'numx': 1,
        'trim': 1.0,
        'jump': 0.0,
        'jumptype': "stationary",
        'startX': "stationary"})

def numToAnimal(data, items):
    for lnum, l in enumerate(data):
        for inum, i in enumerate(l):
            data[lnum][inum]=items[i]
    return data

def freqToProp(subj, freqs, bylist=0):
    if bylist:
        totalitems=len(listlengths[subj])
    else:
        totalitems=sum(listlengths[subj])
        
    for i in freqs:
        freqs[i]=freqs[i]/totalitems
    return freqs

real_lists = './Spring2015/results_cleaned.csv'
real_graphs = './Spring2015/s2015_combined.csv'

## import real_ data
real_data={}
real_irts={}
listlengths={}
for subj in subs:
    data, items, irts, numnodes=rw.readX(subj,"animals",real_lists)
    listlengths[subj]=[len(x) for x in data]
    data = numToAnimal(data, items)
    real_data[subj]=data
    real_irts[subj]=irts

## generate fake_ data and irts yoked to real_ data
numsets=100     # number of sets of fake data per SS
fake_data={}
fake_irts={}

for subj in subs:
    graph, items = rw.read_csv(real_graphs,cols=('node1','node2'),header=1,
                               filters={'subj': str(subj), 'uinvite': '1'})
    graph=nx.from_numpy_matrix(graph)
   
    fake_data[subj]=[]
    fake_irts[subj]=[]
    for setnum in range(numsets):
        dataset=[]
        irtset=[]
        for trimval in listlengths[subj]:
            toydata.trim = trimval
            print str(subj), str(toydata.trim)
            [data,irts,alter_graph]=rw.genX(graph, toydata)
            data=numToAnimal(data, items)[0]
            dataset.append(data)
            irtset.append(irts)
        fake_data[subj].append(dataset)
        fake_irts[subj].append(irtset)
