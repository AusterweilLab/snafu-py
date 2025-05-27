import os
import snafu
import pytest

@pytest.fixture
def fluencydata():
    return snafu.load_fluency_data(
        "../fluency_data/snafu_sample.csv",
        category="foods",
        spell="../spellfiles/foods_snafu_spellfile.csv",
        hierarchical=True
    )

def test_foods_conceptual_network(fluencydata):
    fluencydata.nonhierarchical()
    fitinfo = snafu.Fitinfo({'cn_windowsize': 2, 'cn_threshold': 2, 'cn_alpha': 0.05})
    net = snafu.conceptualNetwork(fluencydata.Xs, fitinfo=fitinfo)
    
    assert net.shape[0] == net.shape[1], "Network should be square (adjacency matrix)"
    assert (net >= 0).all(), "Network should not contain negative values"
    
    # Write graph to file and check if it exists
    output_file = "foods_network.csv"
    snafu.write_graph(net, output_file, labels=fluencydata.groupitems, sparse=False, subj="Group")
    
    assert os.path.exists(output_file), "foods_network.csv should be written"
    
    # Clean up if needed
   # os.remove(output_file)