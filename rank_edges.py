import rw
import networkx as nx
import numpy as np
import pickle
import sys
import numpy as np
import scipy.stats

# load graphs
uinvite_graph, uinvite_items = rw.read_csv("graph_plus_ratings.csv",cols=("node1","node2"),header=True,filters={"s2017_uinvite_hierarchical_zibb_a2_b1_p5": "1"})
#goni_graph, goni_items = rw.read_csv("graph_plus_ratings.csv",cols=("node1","node2"),header=True,filters={"s2017_goni": "1"})

# load data
subs=["S"+str(i) for i in range(101,151)]
filepath = "../Spring2017/results_clean.csv"
category="animals"
Xs, items, irtdata, numnodes, groupitems, groupnumnodes = rw.readX(subs,category,filepath,removePerseverations=True,spellfile="spellfiles/zemla_spellfile.csv")
flatdata, groupitems, irtdata, groupnumnodes = rw.readX(subs,category,filepath,removePerseverations=True,spellfile="spellfiles/zemla_spellfile.csv",flatten=True)



methods=['uinvite_hierarchical_zibb']

td=rw.Data({
        'startX': "stationary",
        'numx': 3,
        'jump': .1,
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

# ugh need to store individual graphs to compute change in LL for u-invite hierarchical...
#rw.probXhierarchical(Xs, 

listofedges=[]
fh=open('graph_plus_ratings.csv','r')
fh.readline()
for line in fh:
    linesplit = line.split(',')
    pair = (linesplit[1],linesplit[2])
    pair = (groupitems.keys()[groupitems.values().index(pair[0])], groupitems.keys()[groupitems.values().index(pair[1])])
    listofedges.append(pair)

fh.close()

# GONI

def goni_weights():
    w=fitinfo_zibb.goni_threshold
    c=0.05
    # frequency of co-occurrences within window (w)
    cooccur=np.zeros((groupnumnodes, groupnumnodes)).astype(int)         # empty graph
    for x in flatdata:                                         # for each list
        for pos in range(len(x)):                            # for each item in list
            for i in range(1, w+1):                          # for each window size
                if pos+i<len(x):
                    cooccur[x[pos],x[pos+i]] += 1
                    cooccur[x[pos+i],x[pos]] += 1

    setXs=[list(set(x)) for x in flatdata]                        # unique nodes in each list
    flatX=rw.flatten_list(setXs)                                  # flattened
    xfreq=[flatX.count(i) for i in range(groupnumnodes)]          # number of lists each item appears in (at least once)
    #listofedges=zip(*np.nonzero(graph))                          # list of edges in graph to check
    numlists=float(len(flatdata))
    meanlistlength=np.mean([len(x) for x in flatdata])
                                                                                                                         
    # Goni et al. (2011), eq. 10
    from statsmodels.stats.proportion import proportion_confint as pci
    p_adj = (2.0/(meanlistlength*(meanlistlength-1))) * ((w*meanlistlength) - ((w*(w+1))/2.0))
    p_linked_array=[]
    for i,j in listofedges:
        p_linked = (xfreq[i]/numlists) * (xfreq[j]/numlists) * p_adj
        ci=pci(cooccur[i,j],numlists,alpha=c,method="beta")[0]    # lower bound of Clopper-Pearson binomial CI
        p_linked_array.append(p_linked)

    fo=open('graph_plus_ratings_weighted.csv','w')
    fh=open('graph_plus_ratings.csv','r')

    fo.write(fh.readline())  # write header
    for linenum, line in enumerate(fh):
        newline = line.split('\n')
        newline = newline[0] + str(p_linked_array[linenum]) + '\n'
        fo.write(newline)

    fo.close()
    fh.close()
