#!/usr/bin/python

import rw
import networkx as nx
import pickle

priortype="prob"

# for priortype=="count"
prior_a=0
prior_b=0
zib_p=.7
count_threshold=5

gs=[]
pickles=["prior.pickle"]

# if you forgot to save items use this
subs=["S"+str(i) for i in range(101,151)]
filepath = "../Spring2017/results_clean.csv"
category="animals"
Xs, items, irtdata, numnodes = rw.readX(subs,category,filepath,removePerseverations=True,spellfile="categories/zemla_spellfile.csv")

for filename in pickles:
    #if "_zibb" in filename:
    #    prior_a = 2
    #elif "_bb" in filename:
    #    prior_a = 1
    
    
    fh=open(filename,"r")
    
    # if you forgot to save items use this
    alldata={}
    alldata['graph']=pickle.load(fh)
    alldata['items']=items

    # else use this
    #alldata=pickle.load(fh)
    
    fh.close()
    if priortype=="graph":
        if filename=="s2017_uinvite_flat.pickle":
            g=nx.to_networkx_graph(alldata['graph'][0]) #oops
        else:
            g=nx.to_networkx_graph(alldata['graph'])
        nx.relabel_nodes(g, alldata['items'], copy=False)
        gs.append(g)
    elif priortype=="prob":
        for cut in [i/100.0 for i in range(50,70)]:
            g = rw.priorToGraph(alldata['graph'], alldata['items'], cutoff=cut)
            g=nx.to_networkx_graph(g)
            nx.relabel_nodes(g, alldata['items'], copy=False)
            gs.append(g)
    elif priortype=="count":
        # prior to prob after count thresholding
        priordict=alldata['graph']
        for item1 in priordict:
            if item1 != "DEFAULTPRIOR":
                for item2 in priordict[item1]:
                    a, b = priordict[item1][item2]      # a=number of participants without link, b=number of participants with link
                    b = b - prior_b
                    a = a - prior_a
                    if (a+b) >= count_threshold:
                        priordict[item1][item2] = float(b)/((1-zib_p)*a+b)
                    else:
                        priordict[item1][item2] = 0.0
        # then probability threshold
        for cut in [i/10.0 for i in range(5,10)]:
            g = rw.priorToGraph(priordict, alldata['items'], cutoff=cut)
            g=nx.to_networkx_graph(g)
            nx.relabel_nodes(g, alldata['items'], copy=False)
            gs.append(g)



rw.write_csv(gs,"newhierarchical.csv",subj="S100") #,extra_data=alldata['graph'])


#for i in alldata['graph']:
#    for j in alldata['graph'][i]:
#        if alldata['graph'][i][j] > .5:
#            print i, j, alldata['graph'][i][j]
