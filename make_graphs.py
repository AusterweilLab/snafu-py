import rw
import networkx as nx
import numpy as np
import pickle
import sys
import numpy as np
import scipy.stats

#methods=['rw','fe','goni','chan','kenett','uinvite','uinvite_hierarchical']
methods=['uinvite_hierarchical','uinvite']

td=rw.Data({
        'startX': "stationary",
        'numx': 3 })

fitinfo=rw.Fitinfo({
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

subs=["S"+str(i) for i in range(101,151)]
filepath = "../Spring2017/results_clean.csv"
category="animals"

irts=[rw.Irts({}) for i in range(len(subs))]
for irt in irts:
    irt.irttype = "gamma"

irtgroup=rw.Irts({})
irtgroup.irttype="gamma"

# read data from file (all as one)
Xs, items, irtdata, numnodes = rw.readX(subs,category,filepath,ignorePerseverations=True,ignoreIntrusions=True,scheme="categories/troyer_hills_zemla_animals.csv",spellfile="categories/zemla_spellfile.csv")

for method in  methods:
    if method=="rw":
        graph = rw.noHidden(Xs, numnodes)
    if method=="goni":
        graph = rw.goni(Xs, numnodes, valid=False, fitinfo=fitinfo)
    if method=="chan":
        graph = rw.chan(Xs, numnodes)
    if method=="kenett":
        graph = rw.kenett(Xs, numnodes)
    if method=="fe":
        graph = rw.firstEdge(Xs, numnodes)
    if method=="uinvite":
        fit_alpha, fit_loc, fit_beta = scipy.stats.gamma.fit(rw.flatten_list(irtdata))
        irtgroup.data = irtdata
        irtgroup.gamma_beta = fit_beta
        graph = rw.uinvite(Xs, td, numnodes, irts=irtgroup, fitinfo=fitinfo)
    if method=="uinvite_hierarchical":
        # renumber dictionary and item list
        start=0
        end=3
        ssnumnodes=[]
        itemsb=[]
        datab=[]
        irtsb=[]
        for sub in range(len(subs)):
            subXs = Xs[start:end]
            irts[sub].data = irtdata[start:end]
            
            fit_alpha, fit_loc, fit_beta = scipy.stats.gamma.fit(rw.flatten_list(irts[sub].data))
            #print fit_alpha, fit_loc, fit_beta, np.mean(rw.flatten_list(irts[sub].data))
            irts[sub].gamma_beta = fit_beta

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

        #print datab
        #print itemsb
        #print ssnumnodes
        sub_graphs, priordict = rw.hierarchicalUinvite(datab, itemsb, ssnumnodes, td, irts=irts, fitinfo=fitinfo)
        graph = rw.priorToGraph(priordict, items)

    fh=open("humans_"+method+"_irt.pickle","w")
    alldata={}
    alldata['graph']=graph
    alldata['items']=items
    pickle.dump(alldata,fh)
    fh.close()
