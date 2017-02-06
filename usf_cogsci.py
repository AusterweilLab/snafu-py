# some hard-coding for s2015!

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
        'jump': 0.6,
        'priming': 0.0,
        'jumptype': "stationary",
        'jumponcensored': None,
        'startX': "stationary"})

# should be same as above, but keep this one un-edited just in case :)
toydata_base=rw.Toydata({
        'numx': 1,
        'trim': 1.0,
        'jump': 0.0,
        'priming': 0.0,
        'jumptype': "stationary",
        'jumponcensored': None,
        'startX': "stationary"})

def numToAnimal(data, items):
    for lnum, l in enumerate(data):
        for inum, i in enumerate(l):
            data[lnum][inum]=items[i]
    return data

def freqToProp(sub, freqs, bylist=0):
    if bylist:
        totalitems=len(listlengths[sub])
    else:
        totalitems=sum(listlengths[sub])
        
    for i in freqs:
        freqs[i]=freqs[i]/totalitems
    return freqs

# from http://locallyoptimal.com/blog/2013/01/20/elegant-n-gram-generation-in-python/
def find_ngrams(input_list, n):
    return zip(*[input_list[i:] for i in range(n)])

real_lists = './Spring2015/results_cleaned.csv'
real_graphs = './Spring2015/s2015_combined.csv'
scheme='./categories/troyer_hills_extended.csv'

## import real data
real_data={}
real_irts={}
listlengths={}
for sub in subs:
    data, items, irts, numnodes=rw.readX(sub,"animals",real_lists)
    listlengths[sub]=[len(x) for x in data]
    data = numToAnimal(data, items)
    real_data[sub]=data
    real_irts[sub]=irts

## generate fake_data and irts yoked to real_data
numsets=100     # number of sets of fake data per SS
fake_data={}
fake_irts={}

USFnet, USFanimals = rw.read_csv('USF_animal_subset.snet')
USFnet = nx.to_networkx_graph(USFnet)

# methods: genX, frequency, cbdfs, sa
def genFakeData(td, method="genX", decay=0):
    for sub in subs:
        fake_data[sub]=[]
        fake_irts[sub]=[]
        for setnum in range(numsets):
            dataset=[]
            irtset=[]
            for listnum, trimval in enumerate(listlengths[sub]):
                td.trim = trimval

                ## SEED START
                # find animal key in USFanimals or substitute with closest animal
                startAnimal=real_data[sub][listnum][0]
                if startAnimal=='polarbear': startAnimal='bear'
                if startAnimal=='beagle': startAnimal='dog'
                if startAnimal=='bulldog': startAnimal='dog'
                if startAnimal=='cheetah': startAnimal='tiger'
                if startAnimal=='python': startAnimal='snake'
                if startAnimal=='ferret': startAnimal='possum'
                if startAnimal=='hedgehog': startAnimal='chipmunk'
                for key, val in USFanimals.iteritems():
                    if val==startAnimal:
                        td.startX=('specific',key)
                ## SEED END

                print str(sub), str(td.trim)
                if method=="genX":
                    [data,irts,alter_graph]=rw.genX(USFnet, td)
                    irtset.append(irts)
                elif method=="frequency":
                    data=[rw.nodeDegreeSearch(USFnet, td) for i in range(3)]
                elif method=="cbdfs":
                    data=[rw.cbdfs(USFnet, td) for i in range(3)]
                elif method=="sa":
                    data=[rw.spreadingActivationSearch(USFnet, td, decay) for i in range(3)]
                data=numToAnimal(data, USFanimals)[0]
                dataset.append(data)
            fake_data[sub].append(dataset)
            fake_irts[sub].append(irtset)
    return fake_data

# num unique animals named across lists
def ssUniqueAnimalsNamed(data, datatype, ngram=1):
    returnval=[]
    for sub in subs:
        if datatype=="fake":
            simAnimalsNamed=[]
            for setnum in range(numsets):
                animallists=[find_ngrams(data[sub][setnum][i],ngram) for i in range(len(data[sub][setnum]))]
                simAnimalsNamed.append(len(set(rw.flatten_list(animallists))))
            simAnimalsNamed=np.mean(simAnimalsNamed)
            returnval.append(simAnimalsNamed),
        elif datatype=="real":
            animallists=[find_ngrams(data[sub][i],ngram) for i in range(len(data[sub]))]
            realAnimalsNamed=len(set(rw.flatten_list(animallists)))
            returnval.append(realAnimalsNamed)
    return returnval

# avg cluster size and avg num cluster switches
def ssClusters(data, datatype, metric):
    returnval=[]
    for sub in subs:
        if datatype=="real":
            cc_real=rw.clusterSize(data[sub], scheme)
            if metric=="size":
                cc_real_avg=rw.avgClusterSize(cc_real)
                returnval.append(cc_real_avg)
            elif metric=="switches":
                cc_real_num=rw.avgNumClusterSwitches(cc_real)
                returnval.append(cc_real_num)
        elif datatype=="fake":
            cc_fake_avg=[]
            cc_fake_num=[]
            for setnum in range(numsets):
                cc_fake=rw.clusterSize(data[sub][setnum], scheme)
                if metric=="size":
                    cc_fake_avg.append(rw.avgClusterSize(cc_fake))
                elif metric=="switches":
                    cc_fake_num.append(rw.avgNumClusterSwitches(cc_fake))
            if metric=="size":
                cc_fake_avg=np.mean(cc_fake_avg)
                returnval.append(cc_fake_avg)
            elif metric=="switches":
                cc_fake_num=np.mean(cc_fake_num)
                returnval.append(cc_fake_num)
    return returnval

# avg num cluster types per list
def ssNumClusters(data, datatype):
    returnval=[]
    for sub in subs:
        if datatype=="real":
            numclusts_real=np.mean([len(set(rw.flatten_list([j.split(';') for j in i]))) for i in rw.labelClusters(data[sub],scheme)])
            returnval.append(numclusts_real)
        elif datatype=="fake":
            numclusts_fake=[]
            for setnum in range(numsets):
                numclusts_fake.append(np.mean([len(set(rw.flatten_list([j.split(';') for j in i]))) for i in rw.labelClusters(data[sub][setnum],scheme)]))
            numclusts_fake=np.mean(numclusts_fake)
            returnval.append(numclusts_fake)
    return returnval

def ssFreq(data, datatype, ngram=1):
    returnval=[]
    if datatype=="real":
        numlists=len(data[subs[0]])               # assumes same number of lists per subect!
    elif datatype=="fake":
        numlists=len(data[subs[0]][0])
    for sub in subs:
        if datatype=="real":
            animallists=[find_ngrams(data[sub][i],ngram) for i in range(len(data[sub]))]
            real_freq=rw.freq_stat(animallists)
            for i in range(1,numlists+1):            # if key doesn't exist add with value 0
                if i not in real_freq.keys():
                    real_freq[i]=0
            returnval.append(real_freq)
        elif datatype=="fake":
            fake_freqs=[]
            for setnum in range(numsets):
                animallists=[find_ngrams(data[sub][setnum][i],ngram) for i in range(len(data[sub][setnum]))]
                fake_freqs.append(rw.freq_stat(animallists))
                for i in range(1,numlists+1):            # if key doesn't exist add with value 0
                    if i not in fake_freqs[-1].keys():
                        fake_freqs[-1][i]=0
            fake_freq={}
            for i in range(1,numlists+1):
                fake_freq[i]=sum(d[i] for d in fake_freqs) / len(fake_freqs)
            returnval.append(fake_freq)
    return returnval
#
## real lists only, for now
## do people start with small clusters and move to larger clusters, vice versa, or random?
## report cluster size by item, averaging across all clusters to which an item belongs
## result: slight trend to move from smaller to larger clusters but nothing obvious
#def clusterSizeOrder(data):
#    returnval=[]
#    # http://stackoverflow.com/questions/36352300/python-compute-average-of-n-th-elements-in-list-of-lists-with-different-lengths
#    from itertools import izip_longest, imap
#    def avg(x):
#        x = filter(None, x)
#        return sum(x, 0.0) / len(x)
#    
#    ##### copied from rw.labelClusters
#    cf=open(scheme,'r')
#    cats={}
#    for line in cf:
#        line=line.rstrip()
#        cat, items = line.split(',', 1)
#        cat=cat.lower().replace(' ','')
#        for item in items.split(','):
#            if item != '':
#                item=item.lower().replace(' ','')
#                if item not in cats.keys():
#                    cats[item]=cat
#                else:
#                    if cat not in cats[item]:
#                        cats[item]=cats[item] + ';' + cat
#    #####
#    catfreqs=rw.freq(rw.flatten_list([i.split(';') for i in cats.values()]))
#    
#    for sub in subs:
#        clusts = rw.labelClusters(data[sub],scheme)
#        newclusts=[]
#        for l in clusts:    # long winded
#            cl=[]
#            for i in l:
#                tmp=[catfreqs[j] for j in i.split(';')]
#                cl.append(np.mean(tmp))     # could use max instead of np.mean
#            newclusts.append(cl)
#        newclusts = list(imap(avg, izip_longest(*newclusts)))
#        returnval.append(newclusts)
#    returnval = list(imap(avg, izip_longest(*returnval)))
#    return returnval

# for real lists only                     
# are n-grams more common in adjacent listssFreq()s compared to separated list?
# Spring2015 data: yes! 1-gram, 2-gram, 3-gram all more common in adjacent list
def ssNgramAdjacent(ngram=2):
    for sub in subs:
        l=[find_ngrams(real_data[sub][i], ngram) for i in range(len(real_data[sub]))]
        o01=o02=o12=0
        for i in l[0]:
            if i in l[1]:
                o01 += 1
            if i in l[2]:
                o02 += 1
        for i in l[1]:
            if i in l[2]:
                o12 += 1
        print (o01+o12)/2.0, o02

def realMandSD(data):
    # frequency
    uniqueUnigrams = ssUniqueAnimalsNamed(data, datatype="real")
    uniqueBigrams = ssUniqueAnimalsNamed(data, datatype="real", ngram=2)
    ufreqs = ssFreq(data, datatype="real")     # for 3 lists
    ufreqs1=[i[1] for i in ufreqs]
    ufreqs2=[i[2] for i in ufreqs]
    ufreqs3=[i[3] for i in ufreqs]
    
    # clustering
    clusterSize = ssClusters(data, datatype="real", metric="size")
    clusterSwitches = ssClusters(data, datatype="real", metric="switches")
    numClusters = ssNumClusters(data, datatype="real")
    
    z=dict()
    z['uniqueUnigrams'] = (np.mean(uniqueUnigrams), np.std(uniqueUnigrams))
    z['uniqueBigrams'] = (np.mean(uniqueBigrams), np.std(uniqueBigrams))
    z['ufreqs1'] = (np.mean(ufreqs1), np.std(ufreqs1))
    z['ufreqs2'] = (np.mean(ufreqs2), np.std(ufreqs2))
    z['ufreqs3'] = (np.mean(ufreqs3), np.std(ufreqs3))
    z['clusterSwitches'] = (np.mean(clusterSwitches), np.std(clusterSwitches))
    z['clusterSize'] = (np.mean(clusterSize), np.std(clusterSize))
    z['numClusters'] = (np.mean(numClusters), np.std(numClusters))
    return z

def findBestParam(data, ranges, method):
    zmax=[]
    for param in ranges:
        # priming
        if method=="priming":
            toydata_base.priming=param
            fake_data = genFakeData(toydata_base, method="genX")
        if method=="jumponcensored":
            toydata_base.jumponcensored=param
            fake_data = genFakeData(toydata_base, method="genX")
        if method=="randomjump":
            toydata_base.jump=param
            fake_data = genFakeData(toydata_base, method="genX")
        if method=="sa":
            fake_data = genFakeData(toydata_base, method="sa", decay=param)
        # generate predictions
        uniqueUnigrams = ssUniqueAnimalsNamed(data, datatype="fake")
        uniqueBigrams = ssUniqueAnimalsNamed(data, datatype="fake")
        ufreqs = ssFreq(data, datatype="fake")     # for 3 lists
        ufreqs1=[i[1] for i in ufreqs]
        ufreqs2=[i[2] for i in ufreqs]
        ufreqs3=[i[3] for i in ufreqs]
        clusterSwitches = ssClusters(data, datatype="fake", metric="switches")
        clusterSize = ssClusters(data, datatype="fake", metric="size")
        numClusters = ssNumClusters(data, datatype="fake")
        # generate z score s
        zs=[]
        zs.append(abs(np.mean(uniqueUnigrams) - z['uniqueUnigrams'][0])/z['uniqueUnigrams'][1])
        zs.append(abs(np.mean(uniqueBigrams) - z['uniqueBigrams'][0])/z['uniqueBigrams'][1])
        zs.append(abs(np.mean(ufreqs1) - z['ufreqs1'][0])/z['ufreqs1'][1])
        zs.append(abs(np.mean(ufreqs2) - z['ufreqs2'][0])/z['ufreqs2'][1])
        zs.append(abs(np.mean(ufreqs3) - z['ufreqs3'][0])/z['ufreqs3'][1])
        zs.append(abs(np.mean(clusterSwitches) - z['clusterSwitches'][0])/z['clusterSwitches'][1])
        zs.append(abs(np.mean(clusterSize) - z['clusterSize'][0])/z['clusterSize'][1])
        zs.append(abs(np.mean(numClusters) - z['numClusters'][0])/z['numClusters'][1])
        zmax.append(max(zs))
    return zmax

def genPredictions(data, datatype):
    metrics=[]
    uniqueUnigrams = ssUniqueAnimalsNamed(data, datatype=datatype)
    uniqueBigrams = ssUniqueAnimalsNamed(data, datatype=datatype, ngram=2)
    ufreqs = ssFreq(data, datatype=datatype)     # for 3 lists
    ufreqs1=[i[1] for i in ufreqs]
    ufreqs2=[i[2] for i in ufreqs]
    ufreqs3=[i[3] for i in ufreqs]
    clusterSwitches = ssClusters(data, datatype=datatype, metric="switches")
    clusterSize = ssClusters(data, datatype=datatype, metric="size")
    numClusters = ssNumClusters(data, datatype=datatype)
    metrics=[uniqueUnigrams, uniqueBigrams, ufreqs1, ufreqs2, ufreqs3, clusterSize, clusterSwitches, numClusters]
    return metrics

# FIND BEST PARAM

z=realMandSD(real_data)
#out=findBestParam(fake_data, [i/20.0 for i in range(0,21)], "priming")
#out=findBestParam(fake_data, range(10), "jumponcensored")
#out=findBestParam(fake_data, [i/20.0 for i in range(0,21)], "randomjump")
out=findBestParam(fake_data, [i/20.0 for i in range(0,21)], "sa")

# PREDICTIONS

fake_data=genFakeData(toydata, method="sa",decay=.25)
#fake_data=genFakeData(toydata)
#metrics=genPredictions(real_data, "real")
metrics=genPredictions(fake_data, "fake")
for m in metrics:
        for i in m:
            print i
