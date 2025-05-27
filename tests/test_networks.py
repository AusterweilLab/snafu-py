# tests/test_network_estimations.py

import os
import pytest
import snafu
import numpy as np

@pytest.fixture
def fluencydata():
    return snafu.load_fluency_data(
        "../fluency_data/snafu_sample.csv",
        category="animals",
        removePerseverations=True,
        spell="../spellfiles/animals_snafu_spellfile.csv",
        hierarchical=False,
        group="Experiment1"
    )

def test_network_estimations(fluencydata):
    fitinfo = snafu.Fitinfo({
        'cn_alpha': 0.05,
        'cn_windowsize': 2,
        'cn_threshold': 2
    })

    output_files = []

    # Naive Random Walk
    nrw_graph = snafu.naiveRandomWalk(fluencydata.Xs, numnodes=fluencydata.groupnumnodes)
    assert nrw_graph.shape[0] == nrw_graph.shape[1]
    output_files.append("nrw_graph.csv")
    snafu.write_graph(nrw_graph, output_files[-1], labels=fluencydata.groupitems, subj="GROUP")

    # Conceptual Network
    cn_graph = snafu.conceptualNetwork(fluencydata.Xs, numnodes=fluencydata.groupnumnodes, fitinfo=fitinfo)
    assert cn_graph.shape[0] == cn_graph.shape[1]
    output_files.append("cn_graph.csv")
    snafu.write_graph(cn_graph, output_files[-1], labels=fluencydata.groupitems, subj="GROUP")

    # Pathfinder
    pf_graph = snafu.pathfinder(fluencydata.Xs, numnodes=fluencydata.groupnumnodes)
    assert pf_graph.shape[0] == pf_graph.shape[1]
    output_files.append("pf_graph.csv")
    snafu.write_graph(pf_graph, output_files[-1], labels=fluencydata.groupitems, subj="GROUP")

    # First-Edge
    fe_graph = snafu.firstEdge(fluencydata.Xs, numnodes=fluencydata.groupnumnodes)
    assert fe_graph.shape[0] == fe_graph.shape[1]
    output_files.append("fe_graph.csv")
    snafu.write_graph(fe_graph, output_files[-1], labels=fluencydata.groupitems, subj="GROUP")

    # Check if output files were created
    for file in output_files:
        assert os.path.exists(file), f"{file} was not created."

    # Clean up files
    # for file in output_files:
    #     os.remove(file)
