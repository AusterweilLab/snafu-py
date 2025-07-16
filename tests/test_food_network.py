import os
import pandas as pd
import snafu
import pytest

# --- Paths ---
DATA_PATH = "../fluency_data/snafu_sample.csv"
SPELL_PATH = "../spellfiles/foods_snafu_spellfile.csv"
SCHEME_PATH = "../schemes/foods_snafu_scheme.csv"
OUTPUT_DIR = "test_data"
EXPECTED_DIR = "test_data"
ACTUAL_CSV = os.path.join(OUTPUT_DIR, "foods_network.csv")
EXPECTED_CSV = os.path.join(EXPECTED_DIR, "foods_network_expected.csv")

# --- Setup & Generate ---
def generate_conceptual_network():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(EXPECTED_DIR, exist_ok=True)

    fluencydata = snafu.load_fluency_data(DATA_PATH, category="foods", spell=SPELL_PATH, hierarchical=True)
    fluencydata.nonhierarchical()

    fitinfo = snafu.Fitinfo({'cn_windowsize': 2, 'cn_threshold': 2, 'cn_alpha': 0.05})
    network = snafu.conceptualNetwork(fluencydata.Xs, fitinfo=fitinfo)

    snafu.write_graph(network, ACTUAL_CSV, labels=fluencydata.groupitems, sparse=False, subj="Group")

    if not os.path.exists(EXPECTED_CSV):
        df = pd.read_csv(ACTUAL_CSV)
        df.to_csv(EXPECTED_CSV, index=False)

# --- Test ---
def test_conceptual_network_matches_expected():
    generate_conceptual_network()

    assert os.path.exists(ACTUAL_CSV), f" Missing actual: {ACTUAL_CSV}"
    assert os.path.exists(EXPECTED_CSV), f" Missing expected: {EXPECTED_CSV}"

    actual_df = pd.read_csv(ACTUAL_CSV)
    expected_df = pd.read_csv(EXPECTED_CSV)

    actual_sorted = actual_df.sort_values(by=list(actual_df.columns)).reset_index(drop=True)
    expected_sorted = expected_df.sort_values(by=list(expected_df.columns)).reset_index(drop=True)

    try:
        pd.testing.assert_frame_equal(actual_sorted, expected_sorted, check_dtype=False, check_like=True)
    except AssertionError as e:
        pytest.fail(f"CSVs do not match:\n{e}")
