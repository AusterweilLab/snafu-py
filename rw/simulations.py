import networkx as nx
import numpy as np
from core import *
#from helper import *

def toyBatch(tg, td, outfile, irts=Irts({}), fitinfo=Fitinfo({}), start_seed=0,
             methods=['rw','fe','uinvite','uinvite_irt','uinvite_prior','uinvite_irt_prior'],header=1,debug=False):
    np.random.seed(start_seed)

    # break out of function if using unknown method
    for method in methods:
        if method not in ['rw','fe','uinvite','uinvite_irt','uinvite_prior','uinvite_irt_prior','windowgraph',
                          'windowgraph_valid','threshold','threshold_valid']:
            raise ValueError('ERROR: Trying to fit graph with unknown method: ', method)

    # flag if using a prior method
    if ('uinvite_prior' in methods) or ('uinvite_irt_prior' in methods): use_prior=1
    else: use_prior=0

    # flag if using an IRT method
    if ('uinvite_irt' in methods) or ('uinvite_irt_prior' in methods): use_irt=1
    else: use_irt=0

    if use_prior:
        prior=genPrior(tg, fitinfo.prior_samplesize)

    # stuff to write to file
    globalvals=['numedges','graph_seed','x_seed','truegraph','ll_tg']      # same across all methods, updates with each seed
    methodvals=['method','cost','ll','time','bestgraph','hit','miss','fa','cr']     # differ per method

    f=open(outfile,'a', 0)                # write/append to file with no buffering
    if header==1:
        objs=[tg, td, irts, fitinfo]
        header_towrite=[str(i) for var in objs for i in var.keys() if i!='data'] + globalvals + methodvals
        f.write(','.join(header_towrite) + '\n')

    # write all parameters to file except irts.data (too long for csv file! if they're toy irts, they can be re-generated from seed)
    objs=[tg, td, irts, fitinfo]
    params_towrite=[str(var[i]) for var in objs for i in var.keys() if i!='data']

    # how many graphs to run?
    seed_param=start_seed
    last_seed=start_seed+tg.numgraphs

    while seed_param < last_seed:

        # generate toy graph and data
        # give up if it's hard to generate data that cover full graph

        # generate toy data
        graph_seed=seed_param
        g,a=genG(tg,seed=graph_seed)

        # ugly code -- who writes this shit? oh wait, it's me
        tries=0
        while True:
            x_seed=seed_param

            [Xs, irts.data, alter_graph]=genX(g, td, seed=x_seed)

            # generate IRTs if using IRT model
            if use_irt: irts.data=stepsToIRT(irts, seed=x_seed)

            if alter_graph==0:                      # only use data that covers the entire graph
                break
            else:
                tries=tries+1
                seed_param = seed_param + 1
                last_seed = last_seed + 1    # if data is unusable (doesn't cover whole graph), add another seed
                if tries >= 1000:
                    raise ValueError("Data doesn't cover full graph... Increase 'trim' or 'numx' (or change graph)")

        numedges=nx.number_of_edges(g)
        truegraph=nx.generate_sparse6(g,header=False)  # to write to file

        for method in methods:
            
            recordname="record_"+str(graph_seed)+"_"+str(x_seed)+"_"+method+".csv"
            if debug: print "SEED: ", seed_param, "method: ", method
            
            # Find best graph! (and log time)
            ll_tg=""        # only record TG LL for U-INVITE models; otherwise it's ambiguous whether it's using IRT/prior/etc
            starttime=datetime.now()
            if method == 'uinvite': 
                bestgraph, ll=uinvite(Xs, td, tg.numnodes, debug=debug, fitinfo=fitinfo, recordname=recordname)
                ll_tg=probX(Xs, a, td)[0]
            if method == 'uinvite_prior':
                bestgraph, ll=uinvite(Xs, td, tg.numnodes, prior=prior, debug=debug, fitinfo=fitinfo, recordname=recordname)
                ll_tg=probX(Xs, a, td, prior=prior)[0]
            if method == 'uinvite_irt':
                bestgraph, ll=uinvite(Xs, td, tg.numnodes, irts=irts, debug=debug, fitinfo=fitinfo, recordname=recordname)
                ll_tg=probX(Xs, a, td, irts=irts)[0]
            if method == 'uinvite_irt_prior':
                bestgraph, ll=uinvite(Xs, td, tg.numnodes, irts=irts, prior=prior, debug=debug, fitinfo=fitinfo, recordname=recordname)
                ll_tg=probX(Xs, a, td, prior=prior, irts=irts)[0]
            if method == 'windowgraph':
                bestgraph=windowGraph(Xs, tg.numnodes, fitinfo=fitinfo)
                ll=probX(Xs, bestgraph, td)[0]
            if method == 'windowgraph_valid':
                bestgraph=windowGraph(Xs, tg.numnodes, td=td, valid=1, fitinfo=fitinfo)
                ll=probX(Xs, bestgraph, td)[0]
            if method=='threshold':
                bestgraph=windowGraph(Xs, tg.numnodes, fitinfo=fitinfo, c=1)
                ll=probX(Xs, bestgraph, td)[0]
            if method=='threshold_valid':
                bestgraph=windowGraph(Xs, tg.numnodes, td=td, valid=1, fitinfo=fitinfo, c=1)
                ll=probX(Xs, bestgraph, td)[0]
            if method == 'rw':
                bestgraph=noHidden(Xs, tg.numnodes)
                ll=probX(Xs, bestgraph, td)[0]
            if method == 'fe':
                bestgraph=firstEdge(Xs, tg.numnodes)
                ll=probX(Xs, bestgraph, td)[0]
            elapsedtime=str(datetime.now()-starttime)
            if debug: 
                print elapsedtime
                print "COST: ", cost(bestgraph,a)
                print nx.generate_sparse6(nx.to_networkx_graph(bestgraph),header=False)

            # Record cost, time elapsed, LL of best graph, hash of best graph, and SDT measures
            graphcost=cost(bestgraph,a)
            hit, miss, fa, cr = costSDT(bestgraph,a)
            graphhash=nx.generate_sparse6(nx.to_networkx_graph(bestgraph),header=False)

            global_towrite=[str(i) for i in [numedges, graph_seed, x_seed, truegraph, ll_tg]]
            method_towrite=[str(i) for i in [method, graphcost, ll, elapsedtime, graphhash, hit, miss, fa, cr]]

            # log stuff here
            f.write(','.join(params_towrite) + ',')
            f.write(','.join(global_towrite) + ',')
            f.write(','.join(method_towrite) + '\n')

        seed_param = seed_param + 1
    f.close()
