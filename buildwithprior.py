import os
import rw
import networkx as nx
import numpy as np
import pickle
import sys
import numpy as np
import scipy.stats
import sys

ss_num=int(sys.argv[2])

methods=['uinvite_hierarchical']

td=rw.Data({
        'startX': "stationary",
        'numx': 3 })

fitinfo_goni=rw.Fitinfo({
        'prior_method': "betabinomial",
        'prior_a': 1,
        'prior_b': 1,
        'startGraph': "goni_valid",
        'goni_size': 2,
        'goni_threshold': 2,
        'followtype': "avg", 
        'prune_limit': np.inf,
        'triangle_limit': np.inf,
        'other_limit': np.inf })

subs=["S"+str(i) for i in range(101,151)]
filepath = "results_clean.csv"
category="animals"

# read data from file (all as one)
Xs, items, irtdata, numnodes, groupitems, groupnumnodes = rw.readX(subs,category,filepath,removePerseverations=True,spellfile="zemla_spellfile.csv")
flatdata, groupitems, irtdata, groupnumnodes = rw.readX(subs,category,filepath,removePerseverations=True,spellfile="zemla_spellfile.csv",flatten=True)

goni_graph = rw.goni(flatdata, groupnumnodes, valid=False, fitinfo=fitinfo_goni)
priordict = rw.genGraphPrior([goni_graph], [groupitems], fitinfo_goni)
prior=(priordict, groupitems)

graph, ll = rw.uinvite(Xs[ss_num], td, numnodes[ss_num], fitinfo=fitinfo_goni, prior=prior)
    
fh=open("h_"+str(ss_num)+".pickle","w")
alldata={}
alldata['graph']=graph
alldata['items']=items[ss_num]
pickle.dump(alldata,fh)
fh.close()
