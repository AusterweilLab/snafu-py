import rw
import networkx as nx
import numpy as np
import pickle
import sys
import numpy as np
import scipy.stats

# these are different methods for generating graphs from fluency data
# our methods are the uinvite_* methods

#methods=['rw','fe','goni','chan','kenett','uinvite_flat','uinvite_hierarchical_bb','uinvite_hierarchical_zibb']
#methods=['rw','fe','goni','chan','kenett','uinvite_flat']
methods=['chan']

td=rw.Data({
        'startX': "stationary",
        'numx': 3,
        'jumptype': "stationary"
        })

fitinfo_zibb=rw.Fitinfo({
        'prior_method': "zeroinflatedbetabinomial",
        'zibb_p': .5,
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
        'prior_b': 1,
        'startGraph': "goni_valid",
        'goni_size': 2,
        'goni_threshold': 2,
        'followtype': "avg", 
        'prune_limit': np.inf,
        'triangle_limit': np.inf,
        'other_limit': np.inf })

# the hierarchical model will take a long time to run!! to test it you can fit a smaller number of participants, e.g. range(101,111)
subs=["S"+str(i) for i in range(101,151)]
filepath = "../Spring2017/results_clean.csv"
category="animals"

# read data from file (all as one)
flatdata, groupitems, irtdata, groupnumnodes = rw.readX(subs,category,filepath,removePerseverations=True,spellfile="spellfiles/zemla_spellfile.csv",flatten=True)

fo=open('individual_graphs2.csv','w')
fo.write('subj,method,item1,item2,edge\n')

for method in methods:
    for sub in subs:
        Xs, items, irtdata, numnodes = rw.readX(sub,category,filepath,removePerseverations=True,spellfile="spellfiles/zemla_spellfile.csv")
        if method=="rw":
            graph = rw.noHidden(Xs, numnodes)
        if method=="goni":
            graph = rw.goni(Xs, numnodes, valid=False, fitinfo=fitinfo_zibb)
        if method=="chan":
            graph = rw.chan(Xs, numnodes)
        if method=="kenett":
            graph = rw.kenett(Xs, numnodes)
        if method=="fe":
            graph = rw.firstEdge(Xs, numnodes)
        if method=="uinvite_flat":
            graph, ll = rw.uinvite(Xs, td, numnodes, fitinfo=fitinfo_zibb)

        for i in range(len(graph)):
            for j in range(len(graph)):
                if i>j:
                    item1=items[i]
                    item2=items[j]
                    itempair=np.sort([item1,item2])
                    fo.write(sub + "," + method + "," + itempair[0] + "," + itempair[1] +  "," + str(graph[i,j]) + "\n")
fo.close()
