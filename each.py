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

fitinfo_zibb=rw.Fitinfo({
        'prior_method': "zeroinflatedbetabinomial",
        'zib_p': .3,
        'prior_a': 2,
        'prior_b': 1,
        'startGraph': "goni_valid",
        'goni_size': 2,
        'goni_threshold': 2,
        'followtype': "avg", 
        'prune_limit': np.inf,
        'triangle_limit': np.inf,
        'other_limit': np.inf })

fitinfo_bb=rw.Fitinfo({
        'prior_method': "betabinomial",
        'prior_a': 1,
        'prior_b': .75,
        'startGraph': "goni_valid",
        'goni_size': 2,
        'goni_threshold': 2,
        'followtype': "avg", 
        'prune_limit': np.inf,
        'triangle_limit': np.inf,
        'other_limit': np.inf })

filename='h_'+str(ss_num)+'.pickle'
if os.stat(filename).st_size > 0:
    fh=open(filename,'r')
    alldata = pickle.load(fh)
    fh.close()
    fitinfo_zibb.startGraph = alldata['graph']
    fitinfo_bb.startGraph = alldata['graph']

fh = open("prior.pickle",'r')
priordict = pickle.load(fh)['graph']  # grab only priordict on each iteration
fh.close()

subs=["S"+str(i) for i in range(101,151)]
filepath = "results_clean.csv"
category="animals"

# read data from file (all as one)
Xs, items, irtdata, numnodes, groupitems, groupnumnodes = rw.readX(subs,category,filepath,removePerseverations=True,spellfile="zemla_spellfile.csv")

graph, ll = rw.uinvite(Xs[ss_num], td, numnodes[ss_num], fitinfo=fitinfo_zibb, prior=(priordict,items[ss_num]))

fh=open("h_"+str(ss_num)+".pickle","w")
alldata={}
alldata['graph']=graph
alldata['items']=items[ss_num]
pickle.dump(alldata,fh)
fh.close()
