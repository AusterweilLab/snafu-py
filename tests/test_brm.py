import snafu
import pytest
import os
import numpy as np

# Adjust these paths as needed to your actual repo structure
DATA_PATH = "../fluency_data/snafu_sample.csv"
SCHEME_PATH = "../schemes/animals_snafu_scheme.csv"
SPELL_PATH = "../spellfiles/animals_snafu_spellfile.csv"
FREQ_PATH = "../frequency/subtlex-us.csv"
AOA_PATH = "../aoa/kuperman.csv"

@pytest.fixture
def fluencydata():
    return snafu.load_fluency_data(
        DATA_PATH,
        category="animals",
        removeNonAlphaChars=True,
        spell=SPELL_PATH,
        group=["Experiment1", "Experiment2"],
        hierarchical=True
    )

def test_data_loads(fluencydata):
    assert fluencydata is not None
    assert isinstance(fluencydata.labeledlists, list)
    assert len(fluencydata.labeledlists) > 0

def test_cluster_switches(fluencydata):
    switches = snafu.clusterSwitch(fluencydata.labeledlists, SCHEME_PATH)
    assert isinstance(switches, list)
    assert all(isinstance(val, (int, float)) for val in switches)

def test_perseverations(fluencydata):
    p = snafu.perseverations(fluencydata.labeledlists)
    print("perseverations:", p)
    assert isinstance(p, list)
    assert all(isinstance(val, (float, np.floating)) for val in p)

def test_intrusions(fluencydata):
    i = snafu.intrusions(fluencydata.labeledlists, SCHEME_PATH)
    print("intrusions:", i)
    assert isinstance(i, list)
    assert all(isinstance(val, (float, np.floating)) for val in i)

def test_word_stats(fluencydata):
    freq, _ = snafu.wordFrequency(fluencydata.labeledlists, data=FREQ_PATH, missing=0.5)
    aoa, _ = snafu.ageOfAcquisition(fluencydata.labeledlists, data=AOA_PATH, missing=None)
    assert isinstance(freq, list)
    assert isinstance(aoa, list)
    assert all(isinstance(f, float) for f in freq)