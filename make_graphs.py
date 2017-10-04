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
methods=['uinvite_hierarchical_bb']

td=rw.Data({
        'startX': "stationary",
        'numx': 3 })

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

# the hierarchical model will take a long time to run!! to test it you can fit a smaller number of participants, e.g. ranmge(101,111)
subs=["S"+str(i) for i in range(101,151)]
filepath = "../Spring2017/results_clean.csv"
category="animals"

# read data from file (all as one)
Xs, items, irtdata, numnodes, groupitems, groupnumnodes = rw.readX(subs,category,filepath,removePerseverations=True,spellfile="schemes/zemla_spellfile.csv")
flatdata, groupitems, irtdata, groupnumnodes = rw.readX(subs,category,filepath,removePerseverations=True,spellfile="schemes/zemla_spellfile.csv",flatten=True)

for method in methods:
    if method=="rw":
        graph = rw.noHidden(flatdata, groupnumnodes)
    if method=="goni":
        c=0.05
        graph = rw.goni(flatdata, groupnumnodes, valid=False, fitinfo=fitinfo_bb, c=c)
    if method=="chan":
        graph = rw.chan(flatdata, groupnumnodes)
    if method=="kenett":
        graph = rw.kenett(flatdata, groupnumnodes)
    if method=="fe":
        graph = rw.firstEdge(flatdata, groupnumnodes)
    if method=="uinvite_flat":
        graph = rw.uinvite(flatdata, td, groupnumnodes, fitinfo=fitinfo_bb)
    if method=="uinvite_hierarchical_bb":
        sub_graphs, priordict = rw.hierarchicalUinvite(Xs, items, numnodes, td, fitinfo=fitinfo_bb)
        graph = rw.priorToGraph(priordict, groupitems)
    if method=="uinvite_hierarchical_zibb":
        sub_graphs, priordict = rw.hierarchicalUinvite(Xs, items, numnodes, td, fitinfo=fitinfo_zibb)
        graph = rw.priorToGraph(priordict, groupitems)

    # graph is an n x n matrix, where n is the number of items in the data set. if graph_{ij} = 1, that indicates an edge between those two items, else no edge exists
    # items is a dictionary that can be used to convert between matrix row/column numbers and animal names
    fh=open("humans_"+method+".pickle","w")
    alldata={}
    alldata['graph']=graph
    alldata['items']=groupitems
    pickle.dump(alldata,fh)
    fh.close()
