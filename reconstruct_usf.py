import rw
import networkx as nx
import numpy as np

usf_graph,usf_items = rw.read_csv("./snet/USF_animal_subset.snet")
usf_graph_nx = nx.from_numpy_matrix(usf_graph)
numnodes = len(usf_items)

numsubs = 50
numlists = 3
listlength = 35
numsims = 50
methods=['rw','goni','chan','kenett','fe']

toydata=rw.Data({
        'numx': numlists,
        'trim': listlength })

fitinfo=rw.Fitinfo({
        'startGraph': "goni_valid",
        'goni_size': 2,
        'goni_threshold': 2,
        'followtype': "avg", 
        'prune_limit': np.inf,
        'triangle_limit': np.inf,
        'other_limit': np.inf})

#toygraphs=rw.Graphs({
#        'numgraphs': 1,
#        'graphtype': "steyvers",
#        'numnodes': 280,
#        'numlinks': 6,
#        'prob_rewire': .3})

# generate data for `numsub` participants, each having `numlists` lists of `listlengths` items
seednum=0
with open('sim_methods.csv','w',0) as fh:
    fh.write("method,simnum,listnum,hit,miss,fa,cr,cost\n")

    for simnum in range(numsims):
        data = []
        for sub in range(numsubs):
            Xs = rw.genX(usf_graph_nx, toydata, seed=seednum)
            data.append(Xs[0])
            seednum += numlists


        for listnum in range(1,len(data)+1):
            print simnum, listnum
            flatdata = rw.flatten_list(data[:listnum])
            rw_graph = rw.noHidden(flatdata, numnodes)
            goni_graph = rw.goni(flatdata, numnodes, td=toydata, valid=0, fitinfo=fitinfo)
            chan_graph = rw.chan(flatdata, numnodes)
            kenett_graph = rw.kenett(flatdata, numnodes)
            fe_graph = rw.firstEdge(flatdata, numnodes)
            #uinvite_graphs, priordict = rw.hierarchicalUinvite(data[:listnum], [usf_items]*numsubs, [numnodes]*numsubs, toydata)

            for method in methods:
                if method=="rw": costlist = [rw.costSDT(rw_graph, usf_graph), rw.cost(rw_graph, usf_graph)]
                if method=="goni": costlist = [rw.costSDT(goni_graph, usf_graph), rw.cost(goni_graph, usf_graph)]
                if method=="chan": costlist = [rw.costSDT(chan_graph, usf_graph), rw.cost(chan_graph, usf_graph)]
                if method=="kenett": costlist = [rw.costSDT(kenett_graph, usf_graph), rw.cost(kenett_graph, usf_graph)]
                if method=="fe": costlist = [rw.costSDT(fe_graph, usf_graph), rw.cost(fe_graph, usf_graph)]
                if method=="uinvite": costlist = [rw.costSDT(uinvite_graph, usf_graph), rw.cost(uinvite_graph, usf_graph)]
                costlist = rw.flatten_list(costlist)
                fh.write(method + "," + str(simnum) + "," + str(listnum))
                for i in costlist:
                    fh.write("," + str(i))
                fh.write('\n')
