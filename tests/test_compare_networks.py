import snafu
import numpy as np
import random
import networkx as nx

def test_network_likelihoods():
    # Load the semantic network
    usf_network, usf_items = snafu.read_graph('..snet/usf_animal_subset.snet')

    # Create perturbed version of the network
    edges = list(zip(*np.where(np.triu(usf_network) == 1.0)))
    nonedges = list(zip(*np.where(np.triu(usf_network) == 0.0)))
    nonedges = [e for e in nonedges if e[0] != e[1]]
    n = round(len(edges) * 0.1)

    edges_to_flip = random.sample(edges, n)
    nonedges_to_flip = random.sample(nonedges, n)

    alt_network = np.copy(usf_network)
    alt_network[list(zip(*edges_to_flip))] = 0.0
    alt_network[list(zip(*nonedges_to_flip))] = 1.0
    alt_network = alt_network + alt_network.T - np.diag(alt_network.diagonal())

    datamodel = snafu.DataModel({
        'start_node': 'stationary',
        'jump': 0.05,
        'jump_type': 'stationary',
        'numx': 10,
        'trim': 20
    })

    usf_lists = snafu.gen_lists(nx.from_numpy_array(usf_network), datamodel)[0]
    alt_lists = snafu.gen_lists(nx.from_numpy_array(alt_network), datamodel)[0]

    loglike_usf_from_usf = snafu.probX(usf_lists, usf_network, datamodel)[0]
    loglike_usf_from_alt = snafu.probX(usf_lists, alt_network, datamodel)[0]
    loglike_alt_from_alt = snafu.probX(alt_lists, alt_network, datamodel)[0]
    loglike_alt_from_usf = snafu.probX(alt_lists, usf_network, datamodel)[0]

    # Assert that true network is a better fit for its own data
    assert loglike_usf_from_usf > loglike_usf_from_alt
    assert loglike_alt_from_alt > loglike_alt_from_usf

    print("Loglike USF from USF:", loglike_usf_from_usf)
    print("Loglike USF from ALT:", loglike_usf_from_alt)
    print("Loglike ALT from ALT:", loglike_alt_from_alt)
    print("Loglike ALT from USF:", loglike_alt_from_usf)
