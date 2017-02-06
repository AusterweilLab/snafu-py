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


real_freqs={}
fake_freqs={}
for subj in subs:
    real_freqs[subj]=rw.freq(rw.flatten_list(real_data[subj]))
    real_freqs[subj]=freqToProp(subj, real_freqs[subj])
    fake_freqs[subj]=[]
    for setnum in range(numsets):
        data=rw.freq(rw.flatten_list(fake_data[subj][setnum]))
        data=freqToProp(subj, data)
        fake_freqs[subj].append(data)


real_starts={}
fake_starts={}
for subj in subs:
    real_starts[subj]=rw.freq(rw.flatten_list([i[0] for i in real_data[subj]]))
    real_starts[subj]=freqToProp(subj, real_starts[subj], bylist=1)
    for item in real_freqs[subj].keys():
        if item not in real_starts[subj]:
            real_starts[subj][item]=0.0
    fake_starts[subj]=[]
    for setnum in range(numsets):
        data=rw.freq(rw.flatten_list([i[0] for i in fake_data[subj][setnum]]))
        data=freqToProp(subj, data, bylist=1)
        for item in real_freqs[subj].keys():
            if item not in data:
                data[item]=0.0
        fake_starts[subj].append(data)

diff_freqs={}
diff_starts={}
rmse_freqs={}
rmse_starts={}

with open('freqs.csv', 'wb') as csv_file:
    writer = csv.writer(csv_file)
    for subj in real_freqs.keys():
        for key, value in real_freqs[subj].items():
            writer.writerow([subj, 'real', 1, key, value])
        for setnum in range(numsets):
            for key, value in fake_freqs[subj][setnum].items():
                writer.writerow([subj, 'fake', setnum, key, value])

# dat2<-dat[,mean(freq),keyby=.(animal,subj,type)]
#for subj in subs:
#    diff_freqs[subj] = {key: real_freqs[subj][key] - fake_freqs[subj].get(key, 0) for key in real_freqs[subj].keys()}
#    diff_starts[subj] = {key: real_starts[subj][key] - fake_starts[subj].get(key, 0) for key in real_starts[subj].keys()}
#    rmse_freqs[subj] = math.sqrt(sum([i**2 for i in diff_freqs[subj].values()]))
#    rmse_starts[subj] = math.sqrt(sum([i**2 for i in diff_starts[subj].values()]))
#    
       
#freqs={1: [], 2: [], 3: []}
#for i in freqdata:
#    for numtimes in range(1,4):
#        try:
#            freqs[numtimes].append(i[numtimes])
#        except:
#            freqs[numtimes].append(0)
#
#for i in freqs:
#    freqs[i]=np.mean(freqs[i])

# (53.39, 18.25, 3.75)
# c.f. 
# dat2<-dat[category=="animals",.N,keyby=.(id,item,game)]
# dat3<-dat2[,.N,keyby=.(item,id)]
# dat4<-table(dat3[,N,keyby=id])
# mean(dat4[,1]) # [,2] [,3] --> (23.35, 17.25, 13.85)
