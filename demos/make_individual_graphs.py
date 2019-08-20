import snafu
import networkx as nx
import numpy as np
import pickle
import sys
import numpy as np
import scipy.stats

# these are different methods for generating graphs from fluency data
# our methods are the uinvite_* methods

#methods=['rw','fe','goni','chan','kenett','uinvite_flat','uinvite_hierarchical']
#methods=['rw','fe','goni','chan','kenett','uinvite_flat']
methods=["uinvite_flat"]

# describe what your data should look like
toydata=snafu.DataModel({
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
fitinfo=snafu.Fitinfo({
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

# the hierarchical model will take a long time to run!! to test it you can fit a smaller number of participants, e.g. range(101,111)
subs=["A"+str(i) for i in range(101,102)]
filepath = "../fluency/snafu_sample.csv"
category="animals"

fo=open('individual_graphs.csv','w')
fo.write('subj,method,item1,item2,edge\n')

for method in methods:
    # add snafu.hierarhicalUinvite method here
    
    for sub in subs:
        filedata = snafu.load_fluency_data(filepath,category=category,removePerseverations=True,spellfile="../spellfiles/animals_snafu_spellfile.csv",subjects=sub)
        filedata.nonhierarchical()
        Xs = filedata.Xs
        items = filedata.items
        numnodes = filedata.numnodes
        
        if method=="rw":
            graph = snafu.nrw(Xs, numnodes)
        if method=="goni":
            graph = snafu.goni(Xs, numnodes, fitinfo=fitinfo)
        if method=="chan":
            graph = snafu.chan(Xs, numnodes)
        if method=="kenett":
            graph = snafu.kenett(Xs, numnodes)
        if method=="fe":
            graph = snafu.firstEdge(Xs, numnodes)
        if method=="uinvite_flat":
            graph, ll = snafu.uinvite(Xs, toydata, numnodes, fitinfo=fitinfo)

        for i in range(len(graph)):
            for j in range(len(graph)):
                if i>j:
                    item1=items[i]
                    item2=items[j]
                    itempair=np.sort([item1,item2])
                    fo.write(sub + "," + method + "," + itempair[0] + "," + itempair[1] +  "," + str(graph[i,j]) + "\n")
fo.close()
