import snafu
import networkx as nx
import numpy as np
import pickle
import sys
import numpy as np

# These parameters are used for U-INVITE variants only
# They describe the hypothetical generative process used to produce your data
#
# jump:       For each step in the walk, jump to a random node with probability 'jump'
# jumptype:   The random node jumped to is chosen either 'uniform' or 'stationary' (proportional to number of edges attached to node)
# start_node: The probability of encountering the first node is either 'uniform' or 'stationary' (proportional to number of edges attached to node)

datamodel = snafu.DataModel({
        'jump': 0.0,
        'jumptype': "stationary",
        'start_node': "stationary" 
})

# These fitting parameters are used for U-INVITE and/or Community Network
fitinfo=snafu.Fitinfo({
        'startGraph': "goni_valid",
        'cn_alpha': 0.05,
        'cn_size': 2,
        'cn_threshold': 2,
        'prune_limit': 100,
        'triangle_limit': 100,
        'other_limit': 100 })

filepath = "fluency_data/snafu_sample.csv"
category="animals"

# read in data from file, flattening all participants together
filedata = snafu.load_fluency_data(filepath,category=category,removePerseverations=True,spell="spellfiles/animals_snafu_spellfile.csv",hierarchical=False)

filedata.nonhierarchical()
Xs_flat = filedata.Xs
groupnumnodes = filedata.numnodes
groupitems = filedata.groupitems
    
# Estimate the best network using a Naive Random Walk
graph = snafu.nrw(Xs_flat, groupnumnodes)

# Estimate the best network using Goni
graph = snafu.goni(Xs_flat, groupnumnodes, fitinfo=fitinfo)
    
# Estimate the best network using Chan
graph = snafu.chan(Xs_flat, groupnumnodes)

# Estimate the best network using Kenett
graph = snafu.kenett(Xs_flat, groupnumnodes)

# Estimate the best network using First-Edge
graph = snafu.firstEdge(Xs_flat, groupnumnodes)
    
# Estimate the best network using a non-hierarchical U-INVITE
graph, ll = snafu.uinvite(Xs_flat, toydata, groupnumnodes, fitinfo=fitinfo)
    
# Estimate the best network using hierarchical U-INVITE
#sub_graphs, priordict = snafu.hierarchicalUinvite(Xs_hier, items, numnodes, toydata, fitinfo=fitinfo)
#graph = snafu.priorToGraph(priordict, groupitems)

# convert numpy matrix to networkx graph and replace indices with semantic labels
#graph = nx.to_networkx_graph(graph)
#nx.relabel_nodes(graph, groupitems, copy=False)
#graphs.append(graph)

#header=','.join(methods)
#snafu.write_graph(graphs, "human_graphs.csv",header=header)
