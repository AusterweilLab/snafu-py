import pytest
import snafu
import networkx as nx
import numpy as np
import os
import csv

def test_regenerated_output_matches_saved_csv():
    saved_path = "../demos_data/usf_reconstruction_results.csv"
    regenerated_path = "../demos_data/temp_regenerated.csv"

    # Load USF graph
    usf_graph, usf_items = snafu.read_graph("../snet/USF_animal_subset.snet")
    usf_graph_nx = nx.from_numpy_array(usf_graph)
    usf_numnodes = len(usf_items)

    # Simulation config (same as original)
    numsubs = 1
    numlists = 3
    listlength = 35
    numsims = 10

    toydata = snafu.DataModel({
        'jump': 0.0,
        'jumptype': "stationary",
        'priming': 0.0,
        'jumponcensored': None,
        'censor_fault': 0.0,
        'emission_fault': 0.0,
        'startX': "stationary",
        'numx': numlists,
        'trim': listlength
    })

    fitinfo = snafu.Fitinfo({
        'startGraph': "cn_valid",
        'directed': False,
        'prior_method': "zeroinflatedbetabinomial",
        'zibb_p': 0.5,
        'prior_a': 2,
        'prior_b': 1,
        'goni_size': 2,
        'goni_threshold': 2,
        'followtype': "avg",
        'prune_limit': np.inf,
        'triangle_limit': np.inf,
        'other_limit': np.inf
    })

    seednum = 0

    # Generate and write regenerated values
    with open(regenerated_path, 'w') as fh:
        fh.write("method,simnum,ssnum,hit,miss,falsealarms,correctrejections,cost,startseed\n")
        for simnum in range(numsims):
            data, data_hier, numnodes, items = [], [], [], []
            startseed = seednum

            for _ in range(numsubs):
                Xs = snafu.gen_lists(usf_graph_nx, toydata, seed=seednum)[0]
                data.append(Xs)
                itemset = set(snafu.flatten_list(Xs))
                numnodes.append(len(itemset))
                ss_Xs, ss_items = snafu.groupToIndividual(Xs, usf_items)
                data_hier.append(ss_Xs)
                items.append(ss_items)
                seednum += numlists

            for ssnum in range(1, len(data) + 1):
                uinvite_graphs, priordict = snafu.hierarchicalUinvite(
                    data_hier[:ssnum], items[:ssnum], numnodes[:ssnum], toydata, fitinfo=fitinfo
                )
                priordict = snafu.genGraphPrior(uinvite_graphs, items[:ssnum], fitinfo=fitinfo, mincount=2)
                uinvite_group_graph = snafu.priorToGraph(priordict, usf_items)

                costlist = snafu.flatten_list([
                    snafu.costSDT(uinvite_group_graph, usf_graph),
                    snafu.cost(uinvite_group_graph, usf_graph)
                ])

                fh.write("uinvite_hierarchical," + str(simnum) + "," + str(ssnum))
                for val in costlist:
                    fh.write("," + str(val))
                fh.write("," + str(startseed))
                fh.write('\n')

    # Now compare both CSVs row by row
    with open(saved_path, newline='') as f_saved, open(regenerated_path, newline='') as f_new:
        reader_saved = list(csv.reader(f_saved))[1:]  # skip header
        reader_new = list(csv.reader(f_new))[1:]

        assert len(reader_saved) == len(reader_new), "Mismatch in number of rows."

        for i, (old_row, new_row) in enumerate(zip(reader_saved, reader_new)):
            assert old_row == new_row, f"Mismatch at row {i}:\nExpected: {old_row}\nGot:      {new_row}"