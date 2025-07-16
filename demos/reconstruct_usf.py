### What does this script do?
### 
### 1) Import the USF network from disk
### 2) Generate toy data (censored random walks) from that network for many pseudo-participants
### 3) Estimate the best network from the data using one or several methods, showing how the method
###    improves as more participants are added
### 4) Write data about those graphs to a CSV (cost, SDT measures)


import snafu
import networkx as nx
import numpy as np
import os

os.makedirs("demos_data", exist_ok=True)

# import the USF network and dictionary of items
usf_graph, usf_items = snafu.read_graph("../snet/USF_animal_subset.snet")

# convert network from adjacency matrix to networkx format
usf_graph_nx = nx.from_numpy_array(usf_graph)
usf_numnodes = len(usf_items)           # how many items are in the USF network

# parameters for simulating data from the USF network
numsubs = 30            # how many pseudo-participants? (one list per subj)
listlength = 30         # how many unique items should each list traverse?
numsims = 1             # how many simulations to perform?

# use all, or some subset of these methods below
# "uinvite" also work, but hierarchical uinvite does not work with this demo
methods = ['naiveRandomWalk','conceptualNetwork','pathfinder','correlationBasedNetwork']

# describe the generative process for the data
toydata = snafu.DataModel({
        'jump': 0.0,
        'jumptype': "stationary",
        'priming': 0.0,
        'jumponcensored': None,
        'censor_fault': 0.0,
        'emission_fault': 0.0,
        'startX': "stationary",     # stationary, uniform, or a specific node?
        'numx': 1,           # number of lists per subject
        'trim': listlength })       # 

# some parameters of the fitting process
fitinfo = snafu.Fitinfo({
        'startGraph': "cn_valid",
        'directed': False,
        'cn_size': 2,
        'cn_threshold': 2 })

seednum=0

with open('demos_data/usf_reconstruction_results.csv','w') as fh:
    fh.write("method,simnum,ssnum,hit,miss,falsealarms,correctrejections,cost,startseed\n")

    for simnum in range(numsims):
        data = []           # Xs using usf_item indices
        numnodes = []       # number of unique nodes traversed by each participant
        items = []          # individual participant index-label dictionaries
        startseed = seednum # random seed

        for sub in range(numsubs):
            
            # generate lists for each participant
            Xs = snafu.gen_lists(usf_graph_nx, toydata, seed=seednum)[0]
            data.append(Xs)

            # record number of unique nodes traversed by each participant's data
            itemset = set(snafu.flatten_list(Xs))
            numnodes.append(len(itemset))

            seednum += 1         # increment random seed so we dont get the same lists next time

        for ssnum in range(1,len(data)+1):
            print("simnum: ", simnum, " subj num: ", ssnum)
            flatdata = snafu.flatten_list(data[:ssnum])  # flatten list of lists
            
            # Generate Naive Random Walk graph from data
            if 'naiveRandomWalk' in methods:
                naiveRandomWalk_graph = snafu.naiveRandomWalk(flatdata, usf_numnodes)

            # Generate Goni graph from data
            if 'conceptualNetwork' in methods:
                conceptualNetwork_graph = snafu.conceptualNetwork(flatdata, usf_numnodes, fitinfo=fitinfo)

            # Generate pathfinder graph from data
            if 'pathfinder' in methods:
                pathfinder_graph = snafu.pathfinder(flatdata, usf_numnodes)
            
            # Generate correlationBasedNetwork from data
            if 'correlationBasedNetwork' in methods:
                correlationBasedNetwork_graph = snafu.correlationBasedNetwork(flatdata, usf_numnodes)

            # Generate First Edge graph from data
            if 'fe' in methods:
                fe_graph = snafu.firstEdge(flatdata, usf_numnodes)
               
            # Generate non-hierarchical U-INVITE graph from data
            if 'uinvite' in methods:
                uinvite_graph, ll = snafu.uinvite(flatdata, toydata, usf_numnodes, fitinfo=fitinfo)
            
            # Generate hierarchical U-INVITE graph from data
            #if 'uinvite_hierarchical' in methods:
            #    uinvite_graphs, priordict = snafu.hierarchicalUinvite(data_hier[:ssnum], items[:ssnum], numnodes[:ssnum], toydata, fitinfo=fitinfo)
            #    
            #    # U-INVITE paper uses an added "threshold" such that at least 2 participants must have an edge for it to be in the group network
            #    # So rather than using the same prior as the one used during fitting, we have to generate a new one
            #    priordict = snafu.genGraphPrior(uinvite_graphs, items[:ssnum], fitinfo=fitinfo, mincount=2)
            #
            #    # Generate group graph from the prior
            #    uinvite_group_graph = snafu.priorToGraph(priordict, usf_items)

            # Write data to file!
            for method in methods:
                if method=="naiveRandomWalk": costlist = [snafu.costSDT(naiveRandomWalk_graph, usf_graph), snafu.cost(naiveRandomWalk_graph, usf_graph)]
                if method=="conceptualNetwork": costlist = [snafu.costSDT(conceptualNetwork_graph, usf_graph), snafu.cost(conceptualNetwork_graph, usf_graph)]
                if method=="pathfinder": costlist = [snafu.costSDT(pathfinder_graph, usf_graph), snafu.cost(pathfinder_graph, usf_graph)]
                if method=="correlationBasedNetwork": costlist = [snafu.costSDT(correlationBasedNetwork_graph, usf_graph), snafu.cost(correlationBasedNetwork_graph, usf_graph)]
                if method=="fe": costlist = [snafu.costSDT(fe_graph, usf_graph), snafu.cost(fe_graph, usf_graph)]
                if method=="uinvite": costlist = [snafu.costSDT(uinvite_graph, usf_graph), snafu.cost(uinvite_graph, usf_graph)]
                costlist = snafu.flatten_list(costlist)
                fh.write(method + "," + str(simnum) + "," + str(ssnum))
                for i in costlist:
                    fh.write("," + str(i))
                fh.write("," + str(startseed))
                fh.write('\n')
