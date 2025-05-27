import os
import snafu
import pytest

DATA_PATH = "../fluency_data/snafu_sample.csv"
SCHEME_PATH = "../schemes/animals_snafu_scheme.csv"
SPELL_PATH = "../spellfiles/animals_snafu_spellfile.csv"
OUTPUT_FILE = "switches.csv"

@pytest.fixture
def fluencydata():
    return snafu.load_fluency_data(
        DATA_PATH,
        category="animals",
        removeNonAlphaChars=True,
        spell=SPELL_PATH,
        group=["Experiment1"]
    )

def test_switch_computation_and_output(fluencydata):
    clusterlabels = snafu.labelClusters(fluencydata.labeledXs, scheme=SCHEME_PATH, labelIntrusions=True)
    switchlists = []

    for fluencylist in clusterlabels:
        switchlist = []
        prev_clusters = []
        for item in fluencylist:
            if item == "intrusion":
                switchlist.append("intrusion")
                continue
            curr_clusters = item.split(';')
            matching_clusters = list(set(prev_clusters) & set(curr_clusters))
            if len(matching_clusters) > 0 or len(prev_clusters) == 0:
                switchlist.append(0)
            else:
                switchlist.append(1)
            prev_clusters = curr_clusters
        switchlists.append(switchlist)

    # Write to file
    with open(OUTPUT_FILE, 'w') as fh:
        fh.write('id,listnum,category,item,switch\n')
        for eachlistnum, eachlist in enumerate(fluencydata.listnums):
            subj = eachlist[0]
            listnum = eachlist[1]
            for itemnum, item in enumerate(fluencydata.labeledXs[eachlistnum]):
                to_write = [subj, str(listnum), "animals", item, str(switchlists[eachlistnum][itemnum])]
                fh.write(','.join(to_write) + "\n")

    # Now test the file was written correctly
    assert os.path.exists(OUTPUT_FILE)

    with open(OUTPUT_FILE, 'r') as f:
        lines = f.readlines()

    assert lines[0].strip() == "id,listnum,category,item,switch"  # header
    for line in lines[1:]:
        parts = line.strip().split(',')
        assert len(parts) == 5
        assert parts[4] in ['0', '1', 'intrusion']

    os.remove(OUTPUT_FILE)  # clean up