import rw
import networkx as nx
import numpy as np
import pickle

numsubs = 50

# read graphs in from file
graphs=[]
for filenum in range(numsubs):
    filename="theta_graphs/theta_"+str(filenum)+".pickle"
    fh=open(filename,"r")
    graph=np.array(pickle.load(fh))
    graphs.append(graph)
    fh.close()
pnumnodes=len(graphs[0])

# hierarchical model needs a dictionary of node->labels, but we don't actually have labels
prior_items={}
for i in range(pnumnodes):
    prior_items[i] = i

# can change these
numlists = 3        # number of lists per person, you can change this
listlength = 25     # number of items per list, you can change this. must be smaller than number of nodes in graph.
numsims = 1         # one simulation runs 1,2...50 participants once (e.g. one line on the graph)

methods=['uinvite_flat','uinvite_hierarchical']
#methods=['uinvite_hierarchical']

toydata=rw.Data({
        'numx': numlists,
        'trim': listlength })

# model parameters you can change
fitinfo=rw.Fitinfo({
        'prior_method': "betabinomial",  # can also be zeroinflatedbetabinomial
        'prior_a': 1,                    # BB(1,1) is 50/50 prior, ZIBB(2,1,.5) is 50/50 prior
        'prior_b': 1,                    # note that graphs are sparse, so 50/50 isn't accurate...
        'zibb_p': .5,                    # only used for zibb
        'startGraph': "goni_valid",
        'goni_size': 2,
        'goni_threshold': 2,
        'followtype': "avg", 
        'prune_limit': np.inf,
        'triangle_limit': np.inf,
        'other_limit': np.inf })

# this is the quickest way to make a group graph in which edge exists if >50% of participants have edge
bb_agg = rw.Fitinfo({
        'prior_method': "betabinomial",
        'prior_a': 1,
        'prior_b': 1 })

priordict = rw.genGraphPrior(graphs, [prior_items]*50, fitinfo=bb_agg) # BB(1,1) ensures no bias in aggregating network
pgraph = rw.priorToGraph(priordict, prior_items)        # prior graph

# generate data for `numsub` participants, each having `numlists` lists of `listlengths` items
seednum=0

with open('theta_results.csv','w',0) as fh:
    fh.write("method,simnum,listnum,hit,miss,fa,cr,cost,startseed\n")

    for simnum in range(numsims):
        data = []       # Xs using group graph indices
        datab = []      # Xs using individual graph indices
        numnodes = []   # number of nodes in each participant graph
        items = []      # ss_items
        startseed = seednum # for recording
        
        # simulate data... some slop, but works
        for sub in range(numsubs):
            Xs = rw.genX(nx.to_networkx_graph(graphs[sub]), toydata, seed=seednum)[0]   # generate lists
            data.append(Xs)

            ####
            # renumber dictionary and item list per individual
            # this is done to avoid fitting full graph for each individual when data only covers a portion of the nodes
            itemset = set(rw.flatten_list(Xs))
            numnodes.append(len(itemset))

            ss_items = {}
            convertX = {}
            for itemnum, item in enumerate(itemset):
                ss_items[itemnum] = prior_items[item]
                convertX[item] = itemnum

            items.append(ss_items)

            Xs = [[convertX[i] for i in x] for x in Xs]
            datab.append(Xs)
            
            seednum += numlists
            ####
        
        for listnum in range(1,len(data)+1):
            print simnum, listnum
            flatdata = rw.flatten_list(data[:listnum])      # treat all lists as if they're from a single participant for non-hierarchical models
            if 'rw' in methods:
                rw_graph = rw.noHidden(flatdata, pnumnodes)
            if 'goni' in methods:
                goni_graph = rw.goni(flatdata, pnumnodes, td=toydata, valid=0, fitinfo=fitinfo)
            if 'chan' in methods:
                chan_graph = rw.chan(flatdata, pnumnodes)
            if 'kenett' in methods:
                kenett_graph = rw.kenett(flatdata, pnumnodes)
            if 'fe' in methods:
                fe_graph = rw.firstEdge(flatdata, pnumnodes)
            if 'uinvite_hierarchical' in methods:
                uinvite_graphs, priordict = rw.hierarchicalUinvite(datab[:listnum], items[:listnum], numnodes[:listnum], toydata, fitinfo=fitinfo)
                uinvite_group_graph = rw.priorToGraph(priordict, prior_items)
            if 'uinvite_flat' in methods:
                uinvite_flat_graph, ll = rw.uinvite(flatdata, toydata, pnumnodes, fitinfo=fitinfo)

            for method in methods:
                if method=="rw": costlist = [rw.costSDT(rw_graph, pgraph), rw.cost(rw_graph, pgraph)]
                if method=="goni": costlist = [rw.costSDT(goni_graph, pgraph), rw.cost(goni_graph, pgraph)]
                if method=="chan": costlist = [rw.costSDT(chan_graph, pgraph), rw.cost(chan_graph, pgraph)]
                if method=="kenett": costlist = [rw.costSDT(kenett_graph, pgraph), rw.cost(kenett_graph, pgraph)]
                if method=="fe": costlist = [rw.costSDT(fe_graph, pgraph), rw.cost(fe_graph, pgraph)]
                if method=="uinvite_hierarchical": costlist = [rw.costSDT(uinvite_group_graph, pgraph), rw.cost(uinvite_group_graph, pgraph)]
                if method=="uinvite_flat": costlist = [rw.costSDT(uinvite_flat_graph, pgraph), rw.cost(uinvite_flat_graph, pgraph)]
                costlist = rw.flatten_list(costlist)
                fh.write(method + "," + str(simnum) + "," + str(listnum))
                for i in costlist:
                    fh.write("," + str(i))
                fh.write("," + str(startseed))
                fh.write('\n')
