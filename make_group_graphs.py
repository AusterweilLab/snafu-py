import rw
import networkx as nx
import numpy as np
import pickle
import sys
import numpy as np
import scipy.stats

methods=['rw','fe','goni','chan','kenett']
methods=['uinvite_flat','uinvite_hierarchical']

# describe what your data should look like
toydata=rw.Data({
        'jump': 0.0,
        'jumptype': "stationary",
        'priming': 0.0,
        'jumponcensored': None,
        'censor_fault': 0.0,
        'emission_fault': 0.0,
        'startX': "stationary",     # stationary, uniform, or a specific node?
        'numx': 3,                  # number of lists per subject
        'trim': 1.0 })        

# some parameters of the fitting process
fitinfo=rw.Fitinfo({
        'startGraph': "goni_valid",
        'record': False,
        'directed': False,
        'prior_method': "zeroinflatedbetabinomial",
        'zibb_p': 0.5,
        'prior_a': 2,
        'prior_b': 1,
        'goni_size': 2,
        'goni_threshold': 2,
        'followtype': "avg", 
        'prune_limit': np.inf,
        'triangle_limit': np.inf,
        'other_limit': np.inf })

subs=["S"+str(i) for i in range(101,151)]
subs=["S"+str(i) for i in range(101,105)]
filepath = "../Spring2017/results_clean.csv"
category="animals"

# read in data from file, flattening all participants together
Xs_flat, groupitems, irtdata, groupnumnodes = rw.readX(subs,category,filepath,removePerseverations=True,spellfile="spellfiles/zemla_spellfile.csv",flatten=True)

# read data from file, preserving hierarchical structure
Xs_hier, items, irtdata, numnodes, groupitems, groupnumnodes = rw.readX(subs,category,filepath,removePerseverations=True,spellfile="spellfiles/zemla_spellfile.csv")

graphs=[]
for method in methods:

    # Estimate the best network using a Naive Random Walk
    if method=="rw":
        graph = rw.nrw(Xs_flat, groupnumnodes)

    # Estimate the best network using Goni
    if method=="goni":
        graph = rw.goni(Xs_flat, groupnumnodes, fitinfo=fitinfo)
        
    # Estimate the best network using Chan
    if method=="chan":
        graph = rw.chan(Xs_flat, groupnumnodes)

    # Estimate the best network using Kenett
    if method=="kenett":
        graph = rw.kenett(Xs_flat, groupnumnodes)

    # Estimate the best network using First-Edge
    if method=="fe":
        graph = rw.firstEdge(Xs_flat, groupnumnodes)
        
    # Estimate the best network using a non-hierarchical U-INVITE
    if method=="uinvite_flat":
        graph = rw.uinvite(Xs_flat, toydata, groupnumnodes, fitinfo=fitinfo)
        
    # Estimate the best network using hierarchical U-INVITE
    if method=="uinvite_hierarchical":
        sub_graphs, priordict = rw.hierarchicalUinvite(Xs_hier, items, numnodes, toydata, fitinfo=fitinfo)
        graph = rw.priorToGraph(priordict, groupitems)

    # convert numpy matrix to networkx graph and replace indices with semantic labels
    graph = nx.to_networkx_graph(graph)
    nx.relabel_nodes(graph, groupitems, copy=False)
    graphs.append(graph)

header=','.join(methods)
rw.write_graph(graphs, "human_graphs.csv",header=header)
