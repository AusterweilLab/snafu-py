import rw
import networkx as nx
import numpy as np
import pickle
import sys
import numpy as np
import scipy.stats

#methods=['rw','fe','goni','chan','kenett','uinvite_flat','uinvite_hierarchical_bb','uinvite_hierarchical_zibb']
#methods=['uinvite_hierarchical','uinvite']
methods=['goni']

td=rw.Data({
        'startX': "stationary",
        'numx': 3 })

fitinfo_zibb=rw.Fitinfo({
        'prior_method': "zeroinflatedbetabinomial",
        'zib_p': .5,
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

subs=["S"+str(i) for i in range(101,151)]
filepath = "../Spring2017/results_clean.csv"
category="animals"

# read data from file (all as one)
Xs, items, irtdata, numnodes = rw.readX(subs,category,filepath,ignorePerseverations=True,ignoreIntrusions=True,scheme="categories/troyer_hills_zemla_animals.csv",spellfile="categories/zemla_spellfile.csv")

for method in  methods:
    if method=="rw":
        graph = rw.noHidden(Xs, numnodes)
    if method=="goni":
        graphs=[]
        for c in [i/1000.0 for i in range(1,10)]:
            print c
            graph = rw.goni(Xs, numnodes, valid=False, fitinfo=fitinfo_bb, c=c)
            graphs.append(graph)
    if method=="chan":
        graph = rw.chan(Xs, numnodes)
    if method=="kenett":
        graph = rw.kenett(Xs, numnodes)
    if method=="fe":
        graph = rw.firstEdge(Xs, numnodes)
    if method=="uinvite_flat":
        graph = rw.uinvite(Xs, td, numnodes, fitinfo=fitinfo_bb)
    if (method=="uinvite_hierarchical_bb") or (method=="uinvite_hierarchical_zibb"):
        # renumber dictionary and item list
        start=0
        end=3
        ssnumnodes=[]
        itemsb=[]
        datab=[]
        for sub in range(len(subs)):
            subXs = Xs[start:end]
            itemset = set(rw.flatten_list(subXs))
            ssnumnodes.append(len(itemset))
                                                        
            ss_items = {}
            convertX = {}
            for itemnum, item in enumerate(itemset):
                ss_items[itemnum] = items[item]
                convertX[item] = itemnum
                                                        
            itemsb.append(ss_items)
                                                        
            subXs = [[convertX[i] for i in x] for x in subXs]
            datab.append(subXs)
            start += 3
            end += 3

        if method=="uinvite_hierarchical_bb":
            sub_graphs, priordict = rw.hierarchicalUinvite(datab, itemsb, ssnumnodes, td, fitinfo=fitinfo_bb)
        elif method=="uinvite_hierarchical_zibb":
            sub_graphs, priordict = rw.hierarchicalUinvite(datab, itemsb, ssnumnodes, td, fitinfo=fitinfo_zibb)
        graph = rw.priorToGraph(priordict, items)

    for graphnum, graph in enumerate(graphs):
        fh=open("humans_new_c"+str(graphnum)+"_"+method+".pickle","w")
        alldata={}
        alldata['graph']=graphs[graphnum]
        alldata['items']=items
        pickle.dump(alldata,fh)
        fh.close()
