# This file explains how to fit U-INVITE and hierarchical U-INVITE networks

# Q: U-INVITE NETWORK ESTIMATION IS SO SLOW...

# A: That's not really a question, but you're not wrong. Of all the network
# estimation methods implemented in SNAFU, U-INVITE is the most computationally
# intensive. The time needed to fit a network increases ~exponentially with the
# size of the network. Networks with a couple dozen nodes may take 2-30
# minutes, whereas a network with hundreds of nodes could take hours.

# Hierarchical U-INVITE fitting time is affected more by the number of
# participants (since each individual graph is small, but must be estimated
# iteratively many times). For very large datasets, we recommend using
# parallelization, but this implementation is not discussed here as it will
# depend on the system used for parallelization. If you need help, try
# e-mailing snafu-fluency@googlegroups.com and we'll figure something out.

# Q: I CAN'T FIGURE THIS HIERARCHICAL U-INVITE STUFF OUT. ISN'T THERE ANOTHER WAY?

# A: You can use a static prior instead. Instad of fitting all networks at
# once, what this does is only fit one network and assumes the rest of the
# networks are fixed. You can use a set of networks as a prior, or just one
# (such as the USF network or Small World of Words network). This process is
# much much faster and results in better networks, but (of course) this biases
# your network towards whatever you use as a prior. See Example 3 below.

# Q: WHAT IS A DATA MODEL?

# A: The DataModel is a dictionary of parameters that describe assumptions
# about how your fluency data is generated.

# Q: WHAT IS 'FIT INFO'?

# A: The estimation method for U-INVITE is stochastic. This is a dictionary of
# parameters that can be used to adjust the fitting procedure, sometimes
# resulting in faster estimation. Most of the time, you probably want to leave
# this alone, but some of the parameters could be useful when specifying a
# prior.

# Q: DO I NEED TO REMOVE PERSEVERATIONS FROM MY DATA?

# A: By default, U-INVITE assumes a censored random walk that does not produce
# perseverations. You can allow perseverations by changing 'censor_fault' in
# the data model or 'estimatePerseveration' in the fit info

# Q: DOES U-INVITE ALLOW FOR ANY POSSIBLE PAIRWISE TRANSITION?

# A: Some transitions might be impossible given a network, e.g., it might not
# be possible to transition from "dog" to "hippo" because that transition may
# have a zero probability under the U-INVITE model. If you allow for random
# jumps, then any transition will be possible. You can do this by changing the
# 'jump' parameter in the data model.

import snafu
import numpy as np
import pickle

# Default parameters are shown
datamodel = snafu.DataModel({
        'jump': 0.0,                        # The censored random walk should jump to a new node with what probability?
        'jumptype': "stationary",           # When jumping, should it jump to a new node with 'uniform' probability or according to the 'stationary' distribution?
        'start_node': "stationary",         # Is the start node in each list chosen with 'uniform' probability or according to the 'stationary' distribution?
        'priming': 0.0,                     # Accounts for increased bi-gram probability in repeated fluency task. See Zemla & Austerweil (2017; cogsci proceeeding)
        'censor_fault': 0.0                 # Repeated nodes are typically censore with probability 1, but can be emitted with some fixed probability
})

# These fitting parameters are used for U-INVITE and/or Conceptual Network
fitinfo = snafu.Fitinfo({
        # U-INVIE needs to be initialized. A modified ConceptualNetwork (see Zemla
        # & Austerweil, 2018) is a reasonable choice, but you can also use
        # NaiveRandomWalk ('nrw'), modified Pathfinder ('pf_valid') or a
        # 'fully_connected' network
        'startGraph': "cn_valid",
        
        'directed': False,                  # U-INVITE can fit directed networks, though it hasn't been tested extensively
        
        # Parameters for the ConceptualNetwork initialization, or when fitting ConceptualNetworks in their own right
        'cn_alpha': 0.05,                   # p-value for deciding if two nodes occur together more than chance. Goni et al (2011) Table 1 \alpha
        'cn_windowsize': 2,                 # Do two items in a list co-occur? See Goni et al (2011) Fig 1
        'cn_threshold': 2,                  # At least this many co-occurrences to add an edge. See Goni et al (2011) Table 1 'Hits'
        
        # U-INVITE will toggle edges in a network to see if the data are more
        # likely under that new network. When set to np.inf, it will
        # exhaustively toggle all edges. Because U-INVITE prioritizes togling
        # edges that are likely to affect the maximum-likelihood solution, you
        # can drastically cut down on compute time by setting a threshold.
        # These thresholds indicate how many edges to toggle in each phase
        # before moving on. See Zemla & Austerweil (2018) and its appendix.
        'prune_limit': 100,                 # ...when trying to remove nodes
        'triangle_limit': 100,              # ...when trying to add edges that form network triangles
        'other_limit': 100,                 # ...when trying to add edges that do not form triangles
        
        # When using a prior, how is the prior probability of each edge determined from the given prior networks?
        # See Zemla & Austerweil (2018) for more detail (ZA2018)
        
        # when using ZIBB, affects proportion of non-observed non-edges (ZA2018
        # Eq. 15 "p_hidden"). when set to 0.0, it becomes the beta-binomial
        # (not zero-inflated). this trio of parameters sets the prior
        # probability of an edge to 0.5 when no other info is known about that
        # edge.
        'zibb_p': 0.5,
        'prior_b': 1,                                 # affects probability of edge when no knowledge of edge exists (ZA2018 Eq. 16 "\beta_0")
        'prior_a': 2,                                 # affects probability of non-edge when no knowledge of edge exists (ZA2018 Eq. 15 "\alpha_0")

        # Instead of specifying a fixed censor_fault in the data model, you can
        # estimate the best fit parameter using maximum-likelihood and grid
        # search
        'estimatePerseveration': False                
        })
        
filepath = "../fluency_data/snafu_sample.csv"
category="animals"

# read in animal fluency data from Experiment 1
fluencydata = snafu.load_fluency_data(filepath,category=category,
                            removePerseverations=True,
                            spell="../spellfiles/animals_snafu_spellfile.csv",
                            hierarchical=True,
                            group="Experiment1")


# RUN A FUNCTION TO SEE ITS RESULTS

# Estimate the best network using a non-hierarchical U-INVITE for the first subject only
def example1():
    uinvite_network, ll = snafu.uinvite(fluencydata.lists[0],    # provide fluency lists
                                      datamodel,                # specify data model
                                      fitinfo=fitinfo,          # specify fit info
                                      debug=True)               # suppress print output to console when set to False
    return uinvite_network
       
# Estimate the best network using hierarchical U-INVITE
def example2():
    # estimate individual graph and return a prior
    individual_graphs, priordict = snafu.hierarchicalUinvite(fluencydata.lists, 
                                        fluencydata.items,
                                        fluencydata.numnodes,
                                        datamodel,
                                        fitinfo=fitinfo)
    # turn the prior edge probabilities into a network, as described in Zemla & Austerweil (2018)
    hierarchical_uinvite_graph = snafu.priorToGraph(priordict, fluencydata.groupitems)
    return individual_graphs, hierarchical_uinvite_graph

# Estimate the best network using a static prior (generated from University of South Florida free association norms) for the first subject only
def example3():
    usf_network, usf_items = snafu.load_network("../snet/USF_animal_subset.snet")
    # Here you can specify multiple networks as a prior; the first parameter is
    # a list of networks, the second parameter is a list of dictionaries that
    # map indices to items in each network
    usf_prior = snafu.genGraphPrior([usf_network], [usf_items])
    uinvite_network, ll = snafu.uinvite(fluencydata.lists[0],
                                    prior=(usf_prior, fluencydata.items[0]))
    return uinvite_network

def main():
    network1 = example1()
    print("\n[Network 1]")
    print("Shape:", network1.shape)
    print("Total edges:", np.sum(network1))
    print("Adjacency matrix:\n", network1)

    with open("uinvite_network1.pkl", "wb") as f:
        pickle.dump(network1, f)

    individual_graphs, group_network = example2()
    print("\n[Individual Graphs]")
    for i, graph in enumerate(individual_graphs):
        print(f"Graph {i}: shape={graph.shape}, total edges={np.sum(graph)}")

    with open("individual_graphs.pkl", "wb") as f:
        pickle.dump(individual_graphs, f)

    print("\n[Group Network]")
    print("Shape:", group_network.shape)
    print("Total edges:", np.sum(group_network))
    print("Adjacency matrix:\n", group_network)

    with open("group_network.pkl", "wb") as f:
        pickle.dump(group_network, f)

    network3 = example3()
    print("\n[Network 3]")
    print("Shape:", network3.shape)
    print("Total edges:", np.sum(network3))
    print("Adjacency matrix:\n", network3)

    with open("uinvite_network3.pkl", "wb") as f:
        pickle.dump(network3, f)

if __name__ == "__main__":
    main()