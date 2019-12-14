import snafu
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

# These fitting parameters are used for U-INVITE and/or Conceptual Network
fitinfo = snafu.Fitinfo({
        'startGraph': "cn_valid",
        'cn_alpha': 0.05,
        'cn_size': 2,
        'cn_threshold': 2,
        'prune_limit': 100,
        'triangle_limit': 100,
        'other_limit': 100 })

filepath = "../fluency_data/snafu_sample.csv"
category="animals"

# read in data from file, flattening all participants together
fluencydata = snafu.load_fluency_data(filepath,category=category,removePerseverations=True,spell="../spellfiles/animals_snafu_spellfile.csv",hierarchical=False,group="Experiment1")

# Estimate the best network using a Naive Random Walk
nrw_graph = snafu.naiveRandomWalk(fluencydata.Xs, numnodes=fluencydata.groupnumnodes)

# Estimate the best network using Conceptual Network (Goni et al)
cn_graph = snafu.conceptualNetwork(fluencydata.Xs, numnodes=fluencydata.groupnumnodes, fitinfo=fitinfo)
    
# Estimate the best network using Pathfinder (Chan et al)
pf_graph = snafu.pathfinder(fluencydata.Xs, numnodes=fluencydata.groupnumnodes)

# Estimate the best network using correlation-based Network (Kenett et al)
# Requires the 'planarity' module for Python
cbn_graph = snafu.correlationBasedNetwork(fluencydata.Xs, numnodes=fluencydata.groupnumnodes)

# Estimate the best network using First-Edge (Abrahao et al)
fe_graph = snafu.firstEdge(fluencydata.Xs, numnodes=fluencydata.groupnumnodes)

# Estimate the best network using a non-hierarchical U-INVITE (Zemla et al)
uinvite_graph, ll = snafu.uinvite(fluencydata.Xs, datamodel, numnodes=fluencydata.groupnumnodes, fitinfo=fitinfo, debug=True)
    
# Estimate the best network using hierarchical U-INVITE
fluencydata.hierarchical()
individual_graphs, priordict = snafu.hierarchicalUinvite(fluencydata.Xs, fluencydata.items, fluencydata.numnodes, datamodel, fitinfo=fitinfo)
hierarchical_uinvite_graph = snafu.priorToGraph(priordict, fluencydata.groupitems)

# write Pathfinder graph, as an example
snafu.write_graph(pf_graph, "pathfinder_graph.csv", labels=fluencydata.groupitems)
