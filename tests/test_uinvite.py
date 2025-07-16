import sys
import os
import pytest
import numpy as np
import snafu

datamodel = snafu.DataModel({
        'jump': 0.0, 
        'jumptype': "stationary", 
        'start_node': "stationary",
        'priming': 0.0,   
        'censor_fault': 0.0     
})

fitinfo = snafu.Fitinfo({
        'startGraph': "cn_valid",
        'directed': False,  
        'cn_alpha': 0.05,  
        'cn_windowsize': 2, 
        'cn_threshold': 2,  
        'prune_limit': 100,  
        'triangle_limit': 100,     
        'other_limit': 100,    
        'zibb_p': 0.5,
        'prior_b': 1, 
        'prior_a': 2,   
        'estimatePerseveration': False                
        })

filepath = "../fluency_data/snafu_sample.csv"
category="animals"

fluencydata = snafu.load_fluency_data(filepath,category=category,
                            removePerseverations=True,
                            spell="../spellfiles/animals_snafu_spellfile.csv",
                            hierarchical=True,
                            group="Experiment1")

def test_example1_runs():
    uinvite_network, ll = snafu.uinvite(fluencydata.lists[0],
                                      datamodel,         
                                      fitinfo=fitinfo,   
                                      debug=True)        
    assert isinstance(uinvite_network, np.ndarray)
    assert uinvite_network.shape[0] == uinvite_network.shape[1]

def test_example2_runs():
    individual_graphs, priordict = snafu.hierarchicalUinvite(fluencydata.lists, 
                                        fluencydata.items,
                                        fluencydata.numnodes,
                                        datamodel,
                                        fitinfo=fitinfo)
    group_graph = snafu.priorToGraph(priordict, fluencydata.groupitems)
    
    assert isinstance(individual_graphs, list)
    assert all(isinstance(g, np.ndarray) for g in individual_graphs)
    assert isinstance(group_graph, np.ndarray)
    assert group_graph.shape[0] == group_graph.shape[1]

def test_example3_runs():
    usf_network, usf_items = snafu.load_network("../snet/USF_animal_subset.snet")
    usf_prior = snafu.genGraphPrior([usf_network], [usf_items])
    uinvite_network, ll = snafu.uinvite(fluencydata.lists[0],
                                    prior=(usf_prior, fluencydata.items[0]))
    assert isinstance(uinvite_network, np.ndarray)
    assert uinvite_network.shape[0] == uinvite_network.shape[1]
