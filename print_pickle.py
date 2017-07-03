#!/usr/bin/python

import rw
import networkx as nx
import pickle

gs=[]
pickles=["humans_uinvite_hierarchical.pickle","humans_rw.pickle","humans_fe.pickle","humans_goni.pickle","humans_chan.pickle","humans_kenett.pickle","humans_uinvite_hierarchical_bb.pickle","humans_uinvite.pickle","humans_uinvite_hierarchical_irt.pickle"]

for filename in pickles:
    fh=open(filename,"r")
    alldata=pickle.load(fh)
    fh.close()
    if filename=="humans_uinvite.pickle":
        g=nx.to_networkx_graph(alldata['graph'][0])  #oops
    else:
        g=nx.to_networkx_graph(alldata['graph'])
    nx.relabel_nodes(g, alldata['items'], copy=False)
    gs.append(g)

rw.write_csv(gs,"humans2017.csv",subj="S100")
