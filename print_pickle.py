#!/usr/bin/python

import rw
import networkx as nx
import pickle
import sys

priortype="count"

# for priortype=="count"
prior_a=2
prior_b=1
zibb_p=.5
count_threshold=2

gs=[]

inputname = sys.argv[1]
pickles=[inputname]

# temporary
pickles=["s2017_chan","s2017_fe","s2017_goni","s2017_kenett","s2017_rw","s2017_uinvite_flat","s2017_uinvite_hierarchical"]


# if you forgot to save items use this
#subs=["S"+str(i) for i in range(101,151)]
#filepath = "../Spring2017/results_clean.csv"
#category="animals"
#Xs, items, irtdata, numnodes, groupitems, groupnumnodes = rw.readX(subs,category,filepath,removePerseverations=True,spellfile="schemes/zemla_spellfile.csv")

for filename in pickles:
    print filename
    if "hierarchical" in filename:
        priortype="count"
    else:
        priortype="graph"
    #if "_zibb" in filename:
    #    prior_a = 2
    #elif "_bb" in filename:
    #    prior_a = 1
    
    
    fh=open(filename+".pickle","r")
    
    # if you forgot to save items use this
    #alldata={}
    #alldata['graph']=pickle.load(fh)['graph']
    #alldata['items']=groupitems

    # else use this
    alldata=pickle.load(fh)
    
    fh.close()
    if priortype=="graph":
        if filename=="s2017_uinvite_flat":
            g=nx.to_networkx_graph(alldata['graph'][0]) #oops
        else:
            g=nx.to_networkx_graph(alldata['graph'])
        nx.relabel_nodes(g, alldata['items'], copy=False)
        gs.append(g)
    elif priortype=="prob":
        for cut in [i/10.0 for i in range(4,8)]:
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
                    b_count = b - prior_b
                    a_count = a - prior_a
                    if b_count >= count_threshold:  # or (a+b)
                        priordict[item1][item2] = float(b)/((1-zibb_p)*a+b)     # zibb
                        #priordict[item1][item2] = float(b)/(a+b)               # bb
                    else:
                        priordict[item1][item2] = 0.0
        # then probability threshold
        for cut in [i/100.0 for i in range(50,51)]:
            g = rw.priorToGraph(priordict, alldata['items'], cutoff=cut)
            g=nx.to_networkx_graph(g)
            nx.relabel_nodes(g, alldata['items'], copy=False)
            gs.append(g)



rw.write_csv(gs,inputname+".csv",subj="S100") #,extra_data=alldata['graph'])


#for i in alldata['graph']:
#    for j in alldata['graph'][i]:
#        if alldata['graph'][i][j] > .5:
#            print i, j, alldata['graph'][i][j]
