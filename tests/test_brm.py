import snafu
import pytest
import numpy as np
import pandas as pd
import json

# Adjust these paths as needed
DATA_PATH = "../fluency_data/snafu_sample.csv"
SCHEME_PATH = "../schemes/animals_snafu_scheme.csv"
SPELL_PATH = "../spellfiles/animals_snafu_spellfile.csv"
FREQ_PATH = "../frequency/subtlex-us.csv"
AOA_PATH = "../aoa/kuperman.csv"
STATS_PATH = "../demos_data/stats.csv"  # CSV with expected values
SEMANTIC_INTRUSIONS_PATH = "../demos_data/intrusions_list.json"
LETTER_INTRUSIONS_PATH = "../demos_data/intrusions_list_letter_a.json"

@pytest.fixture
def expected():
    return pd.read_csv(STATS_PATH).sort_values("sub_id").reset_index(drop=True)    

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

def test_cluster_switch(fluencydata, expected):
    actual = snafu.clusterSwitch(fluencydata.labeledlists, SCHEME_PATH, clustertype="static")
    assert np.allclose(actual, expected["cluster_switch"], equal_nan=True), "Mismatch in cluster_switch"   

def test_switch_rate(fluencydata, expected):
    actual = snafu.clusterSwitch(fluencydata.labeledlists, SCHEME_PATH, clustertype="static", switchrate=True)
    assert np.allclose(actual, expected["switch_rate"], equal_nan=True), "Mismatch in switch_rate"

def test_cluster_size(fluencydata, expected):
    actual = snafu.clusterSize(fluencydata.labeledlists, 2)
    assert np.allclose(actual, expected["cluster_size"], equal_nan=True), "Mismatch in cluster_size"

def test_perseverations(fluencydata, expected):
    actual = snafu.perseverations(fluencydata.labeledlists)
    assert np.allclose(actual, expected["num_perseverations"], equal_nan=True), "Mismatch in num_perseverations"

def test_intrusions(fluencydata, expected):
    actual = snafu.intrusions(fluencydata.labeledlists, SCHEME_PATH)
    assert np.allclose(actual, expected["num_intrusions"], equal_nan=True), "Mismatch in num_intrusions"

def test_intrusions_list_semantic_exact(fluencydata):
    with open(SEMANTIC_INTRUSIONS_PATH, "r") as f:
        expected_intrusions = json.load(f)
    actual_intrusions = snafu.intrusionsList(fluencydata.labeledlists, SCHEME_PATH)
    assert actual_intrusions == expected_intrusions, " Mismatch in semantic intrusions list"

def test_intrusions_list_letter_exact(fluencydata):
    with open(LETTER_INTRUSIONS_PATH, "r") as f:
        expected = json.load(f)

    actual = snafu.intrusionsList(fluencydata.labeledlists, "a")

    assert actual == expected, " Mismatch in letter-based intrusions list"    

def test_word_stats(fluencydata, expected):
    freq_actual, _ = snafu.wordFrequency(fluencydata.labeledlists, data=FREQ_PATH, missing=0.5)
    aoa_actual, _ = snafu.ageOfAcquisition(fluencydata.labeledlists, data=AOA_PATH, missing=None)

    assert np.allclose(freq_actual, expected["word_freq"], equal_nan=True), "Mismatch in word_freq"
    assert np.allclose(aoa_actual, expected["aoa"], equal_nan=True), "Mismatch in aoa"
