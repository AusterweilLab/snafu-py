import snafu
import numpy as np

# STEP 1: Choose a method!
#
# CHOICES: naiveRandomWalk, conceptualNetwork, pathfinder, correlationBasedNetwork, firstEdge, uinvite
#          Don't know which method to choose? Read Zemla & Austerweil (2018):
#          "Estimating semantic networks of groups and individuals from fluency data"
method= "uinvite"

# STEP 2: Where's your data?
filepath = "../fluency_data/snafu_sample.csv"

# STEP 3: Make networks for which subject?
subj = "A101"

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
        'startGraph': "cn_valid",
        'directed': False,
        'prior_method': "zeroinflatedbetabinomial",
        'zibb_p': 0.5,
        'prior_a': 2,
        'prior_b': 1,
        'cn_windowsize': 2,
        'cn_threshold': 2,
        'followtype': "avg", 
        'prune_limit': np.inf,
        'triangle_limit': np.inf,
        'other_limit': np.inf })


# Load the data, grab some info
filedata = snafu.load_fluency_data(filepath,
                                   removePerseverations=True,
                                   spell="../spellfiles/animals_snafu_spellfile.csv",
                                   subject=subj)
filedata.nonhierarchical()
Xs = filedata.Xs
items = filedata.items
numnodes = filedata.numnodes

# Make the network
if method=="naiveRandomWalk":
    graph = snafu.naiveRandomWalk(Xs, numnodes=numnodes)
if method=="conceptualNetwork":
    graph = snafu.conceptualNetwork(Xs, numnodes, fitinfo=fitinfo)
if method=="pathfinder":
    graph = snafu.pathfinder(Xs, numnodes=numnodes)
if method=="correlationBasedNetwork":
    graph = snafu.correlationBasedNetwork(Xs, numnodes=numnodes)
if method=="firstEdge":
    graph = snafu.firstEdge(Xs, numnodes=numnodes)
if method=="uinvite":
    graph, ll = snafu.uinvite(Xs, toydata, numnodes=numnodes, fitinfo=fitinfo, debug=True)

# Write the network to file
snafu.write_graph(graph, "individual_graph.csv", subj=subj, labels=items, sparse=True)
