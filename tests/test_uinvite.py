import sys
import os
import pytest
import numpy as np
import snafu

# Add the 'snafu-py' directory to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'demos')))

from fit_uinvite_network import example1, example2, example3

def test_example1_runs():
    network = example1()
    assert isinstance(network, np.ndarray)
    assert network.shape[0] == network.shape[1]

def test_example2_runs():
    individual_graphs, group_graph = example2()
    assert isinstance(individual_graphs, list)
    assert all(isinstance(g, np.ndarray) for g in individual_graphs)
    assert isinstance(group_graph, np.ndarray)
    assert group_graph.shape[0] == group_graph.shape[1]

def test_example3_runs():
    network = example3()
    assert isinstance(network, np.ndarray)
    assert network.shape[0] == network.shape[1]