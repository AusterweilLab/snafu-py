import pytest
import snafu
import networkx as nx
import numpy as np
import os
import csv

def test_regenerated_output_matches_saved_csv():
    saved_path = "test_data/usf_reconstruction_results.csv"
    regenerated_path = "test_data/temp_regenerated.csv"
    
    # Load USF graph
    usf_graph, usf_items = snafu.read_graph("../snet/USF_animal_subset.snet")
    usf_graph_nx = nx.from_numpy_array(usf_graph)
    usf_numnodes = len(usf_items)
    
    # Simulation config (same as original)
    numsubs = 30
    listlength = 30
    numsims = 1
    
    toydata = snafu.DataModel({
            'jump': 0.0,
            'jumptype': "stationary",
            'priming': 0.0,
            'jumponcensored': None,
            'censor_fault': 0.0,
            'emission_fault': 0.0,
            'startX': "stationary", 
            'numx': 1,     
            'trim': listlength })  
    
    fitinfo = snafu.Fitinfo({
            'startGraph': "cn_valid",
            'directed': False,
            'cn_size': 2,
            'cn_threshold': 2 })
    
    seednum = 0
   
    methods = ['naiveRandomWalk','conceptualNetwork','pathfinder','correlationBasedNetwork'] 
   
    # Generate and write regenerated values
    with open(regenerated_path, 'w') as fh:
        fh.write("method,simnum,ssnum,hit,miss,falsealarms,correctrejections,cost,startseed\n")
        for simnum in range(numsims):
            data, numnodes, items = [], [], []
            startseed = seednum

            for _ in range(numsubs):
                Xs = snafu.gen_lists(usf_graph_nx, toydata, seed=seednum)[0]
                data.append(Xs)
                itemset = set(snafu.flatten_list(Xs))
                numnodes.append(len(itemset))
                seednum += 1

            for ssnum in range(1, len(data) + 1):
                flatdata = snafu.flatten_list(data[:ssnum])  # flatten list of lists
                
                # Generate Naive Random Walk graph from data
                if 'naiveRandomWalk' in methods:
                    naiveRandomWalk_graph = snafu.naiveRandomWalk(flatdata, usf_numnodes)

                # Generate Goni graph from data
                if 'conceptualNetwork' in methods:
                    conceptualNetwork_graph = snafu.conceptualNetwork(flatdata, usf_numnodes, fitinfo=fitinfo)

                # Generate pathfinder graph from data
                if 'pathfinder' in methods:
                    pathfinder_graph = snafu.pathfinder(flatdata, usf_numnodes)
                
                # Generate correlationBasedNetwork from data
                if 'correlationBasedNetwork' in methods:
                    correlationBasedNetwork_graph = snafu.correlationBasedNetwork(flatdata, usf_numnodes)

                # Generate First Edge graph from data
                if 'fe' in methods:
                    fe_graph = snafu.firstEdge(flatdata, usf_numnodes)
                   
                # Generate non-hierarchical U-INVITE graph from data
                if 'uinvite' in methods:
                    uinvite_graph, ll = snafu.uinvite(flatdata, toydata, usf_numnodes, fitinfo=fitinfo)
                
                for method in methods:
                    if method=="naiveRandomWalk": costlist = [snafu.costSDT(naiveRandomWalk_graph, usf_graph), snafu.cost(naiveRandomWalk_graph, usf_graph)]
                    if method=="conceptualNetwork": costlist = [snafu.costSDT(conceptualNetwork_graph, usf_graph), snafu.cost(conceptualNetwork_graph, usf_graph)]
                    if method=="pathfinder": costlist = [snafu.costSDT(pathfinder_graph, usf_graph), snafu.cost(pathfinder_graph, usf_graph)]
                    if method=="correlationBasedNetwork": costlist = [snafu.costSDT(correlationBasedNetwork_graph, usf_graph), snafu.cost(correlationBasedNetwork_graph, usf_graph)]
                    if method=="fe": costlist = [snafu.costSDT(fe_graph, usf_graph), snafu.cost(fe_graph, usf_graph)]
                    if method=="uinvite": costlist = [snafu.costSDT(uinvite_graph, usf_graph), snafu.cost(uinvite_graph, usf_graph)]
                    costlist = snafu.flatten_list(costlist)
                    fh.write(method + "," + str(simnum) + "," + str(ssnum))
                    for i in costlist:
                        fh.write("," + str(i))
                    fh.write("," + str(startseed))
                    fh.write('\n')

    # Now compare both CSVs row by row
    with open(saved_path, newline='') as f_saved, open(regenerated_path, newline='') as f_new:
        reader_saved = list(csv.reader(f_saved))[1:]  # skip header
        reader_new = list(csv.reader(f_new))[1:]

        assert len(reader_saved) == len(reader_new), "Mismatch in number of rows."

        for i, (old_row, new_row) in enumerate(zip(reader_saved, reader_new)):
            assert old_row == new_row, f"Mismatch at row {i}:\nExpected: {old_row}\nGot:      {new_row}"
