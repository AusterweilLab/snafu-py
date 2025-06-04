import snafu
import pandas as pd

DATA_PATH = "../fluency_data/snafu_sample.csv"
SPELL_PATH = "../spellfiles/animals_snafu_spellfile.csv"
SCHEME_PATH = "../schemes/animals_snafu_scheme.csv"
EXPECTED_SWITCHES_PATH = "../demos_data/switches.csv"

def compute_switchlists(clusterlabels):
    switchlists = []
    for fluencylist in clusterlabels:
        switchlist = []
        prev_clusters = []
        for item in fluencylist:
            if item == "intrusion":
                switchlist.append("intrusion")
                continue
            curr_clusters = item.split(';')
            switchlist.append("0" if not prev_clusters or set(prev_clusters) & set(curr_clusters) else "1")
            prev_clusters = curr_clusters
        switchlists.append(switchlist)
    return [s for sublist in switchlists for s in sublist]  # flatten

def test_switches_match():
    fluencydata = snafu.load_fluency_data(
        DATA_PATH,
        category="animals",
        removeNonAlphaChars=True,
        spell=SPELL_PATH,
        group=["Experiment1"]
    )
    clusterlabels = snafu.labelClusters(fluencydata.labeledXs, scheme=SCHEME_PATH, labelIntrusions=True)
    expected_df = pd.read_csv(EXPECTED_SWITCHES_PATH)

    actual_switches = compute_switchlists(clusterlabels)
    print(actual_switches)
    expected_switches = expected_df["switch"].astype(str).tolist()
    print(expected_switches)

    assert len(actual_switches) == len(expected_switches), "Mismatch in number of switch labels"
    assert actual_switches == expected_switches, "Mismatch in switch values"