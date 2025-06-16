import snafu
import numpy as np
import random
import networkx as nx
import pickle
import os

def symmetrize(a):
    return a + a.T - np.diag(a.diagonal())

def generate_likelihoods():
    random.seed(42)
    np.random.seed(42)

    usf_network, _ = snafu.read_graph('../snet/USF_animal_subset.snet')
    edges = list(zip(*np.where(np.triu(usf_network) == 1.0)))
    nonedges = [e for e in zip(*np.where(np.triu(usf_network) == 0.0)) if e[0] != e[1]]
    n = round(len(edges) * 0.1)

    edges_to_flip = random.sample(edges, n)
    nonedges_to_flip = random.sample(nonedges, n)

    alt_network = np.copy(usf_network)
    alt_network[list(zip(*edges_to_flip))] = 0.0
    alt_network[list(zip(*nonedges_to_flip))] = 1.0
    alt_network = symmetrize(alt_network)

    datamodel = snafu.DataModel({
        'start_node': 'stationary',
        'jump': 0.05,
        'jump_type': 'stationary',
        'numx': 20,
        'trim': 35
    })

    usf_lists = snafu.gen_lists(nx.from_numpy_array(usf_network), datamodel, 42)[0]
    alt_lists = snafu.gen_lists(nx.from_numpy_array(alt_network), datamodel, 42)[0]

    return {
        "p_usf_from_usf": round(snafu.probX(usf_lists, usf_network, datamodel)[0], 2),
        "p_usf_from_alternate": round(snafu.probX(usf_lists, alt_network, datamodel)[0], 2),
        "p_alternate_from_usf": round(snafu.probX(alt_lists, usf_network, datamodel)[0], 2),
        "p_alternate_from_alternate": round(snafu.probX(alt_lists, alt_network, datamodel)[0], 2),
    }

def test_network_likelihoods_tolerant_match():
    save_path = "../demos_data/expected_likelihoods.pkl"
    assert os.path.exists(save_path), "Expected likelihood file not found. Please generate it first."

    # Load expected values
    with open(save_path, "rb") as f:
        expected = pickle.load(f)

    # Regenerate new values
    current = generate_likelihoods()

    for key in expected:
        saved_val = float(expected[key])
        new_val = float(current[key])
        diff = abs(saved_val - new_val)

        print(f"{key}: expected={saved_val}, current={new_val}, Δ={diff}")

        # Strict check: fail if difference is not 0
        assert diff == 0.0, f" Mismatch for '{key}': expected {saved_val}, got {new_val} (Δ={diff})"