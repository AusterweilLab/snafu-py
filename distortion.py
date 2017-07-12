import rw
import networkx as nx
import numpy as np
import pickle
import sys

filename=sys.argv[1]

usf_graph, usf_items = rw.read_csv("./snet/USF_animal_subset.snet")
usf_graph_nx = nx.from_numpy_matrix(usf_graph)
usf_numnodes = len(usf_items)

numsubs = 50
numlists = 3
listlength = 35
methods=['rw','goni','chan','kenett','fe']
methods=['uinvite_flat_noirt','uinvite_noirt']

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
        'other_limit': np.inf })

fitinfo_zibb=rw.Fitinfo({
        'prior_method': "zeroinflatedbetabinomial",
        'zib_p': .5,
        'prior_a': 2,
        'prior_b': 1,
        'startGraph': "goni_valid" })

fitinfo_bb=rw.Fitinfo({
        'prior_method': "betabinomial",
        'prior_a': 1,
        'prior_b': 1,
        'startGraph': "goni_valid" })

gamma_beta = 1.0

irts=[rw.Irts({'irttype': 'gamma',
               'data': [],
               'gamma_beta': gamma_beta}) for i in range(numsubs)]

irtgroup=rw.Irts({
    'irttype': 'gamma',
    'data': [],
    'gamma_beta': gamma_beta
})


simfiles = ['./graphs_tofit/simnum_'+str(i)+'.pickle' for i in range(10)]

with open('./graphs_tofit/'+filename,'w',0) as fh:
    fh.write("method,simnum,listnum,hit,miss,fa,cr,cost\n")

    for simnum, sim in enumerate(simfiles):
        pickled=open(sim,'r')
        alldata=pickle.load(pickled)
        data=alldata['data']
        datab=alldata['datab']
        items=alldata['ss_items']
        numnodes=alldata['numnodes']
        ss_irts=alldata['ss_irts']

        for listnum in range(1,len(data)+1):
            print simnum, listnum
            flatdata = rw.flatten_list(data[:listnum])
            flatirt = rw.flatten_list(ss_irts[:listnum])
            irtgroup.data=flatirt
            irts[listnum-1].data=ss_irts[listnum-1]

            if 'rw' in methods:
                rw_graph = rw.noHidden(flatdata, usf_numnodes)
            if 'goni' in methods:
                goni_graph = rw.goni(flatdata, usf_numnodes, td=toydata, valid=0, fitinfo=fitinfo)
            if 'chan' in methods:
                chan_graph = rw.chan(flatdata, usf_numnodes)
            if 'kenett' in methods:
                kenett_graph = rw.kenett(flatdata, usf_numnodes)
            if 'fe' in methods:
                fe_graph = rw.firstEdge(flatdata, usf_numnodes)
            if 'uinvite_flat_noirt' in methods:
                uinvite_flat_graph, ll = rw.uinvite(flatdata, toydata, usf_numnodes, fitinfo=fitinfo)
            if 'uinvite_flat_irt' in methods:
                uinvite_flat_irt_graph, ll = rw.uinvite(flatdata, toydata, usf_numnodes, irts=irtgroup, fitinfo=fitinfo)
            if 'uinvite_noirt' in methods:
                pf=open('uinvite_noirt_'+str(simnum)+"_"+str(listnum),'w')
                uinvite_graphs, priordict = rw.hierarchicalUinvite(datab[:listnum], items[:listnum], numnodes[:listnum], toydata, fitinfo=fitinfo_zibb)
                uinvite_group_graph = rw.priorToGraph(priordict, usf_items)
                pickle.dump(priordict, pf)
                pf.close()
            if 'uinvite_irt' in methods:
                pf=open('uinvite_irt_'+str(simnum)+"_"+str(listnum),'w')
                uinvite_irt_graphs, priordict_irt = rw.hierarchicalUinvite(datab[:listnum], items[:listnum], numnodes[:listnum], toydata, irts=irts[:listnum], fitinfo=fitinfo_zibb)
                uinvite_group_irt_graph = rw.priorToGraph(priordict, usf_items)
                pickle.dump(priordict, pf)
                pf.close()
            
            for method in methods:
                if method=="rw": costlist = [rw.costSDT(rw_graph, usf_graph), rw.cost(rw_graph, usf_graph)]
                if method=="goni": costlist = [rw.costSDT(goni_graph, usf_graph), rw.cost(goni_graph, usf_graph)]
                if method=="chan": costlist = [rw.costSDT(chan_graph, usf_graph), rw.cost(chan_graph, usf_graph)]
                if method=="kenett": costlist = [rw.costSDT(kenett_graph, usf_graph), rw.cost(kenett_graph, usf_graph)]
                if method=="fe": costlist = [rw.costSDT(fe_graph, usf_graph), rw.cost(fe_graph, usf_graph)]
                if method=="uinvite_flat_noirt": costlist = [rw.costSDT(uinvite_flat_graph, usf_graph), rw.cost(uinvite_flat_graph, usf_graph)]
                if method=="uinvite_flat_irt": costlist = [rw.costSDT(uinvite_flat_irt_graph, usf_graph), rw.cost(uinvite_flat_irt_graph, usf_graph)]
                if method=="uinvite_noirt": costlist = [rw.costSDT(uinvite_group_graph, usf_graph), rw.cost(uinvite_group_graph, usf_graph)]
                if method=="uinvite_irt": costlist = [rw.costSDT(uinvite_group_irt_graph, usf_graph), rw.cost(uinvite_group_irt_graph, usf_graph)]
                costlist = rw.flatten_list(costlist)
                fh.write(method + "," + str(simnum) + "," + str(listnum))
                for i in costlist:
                    fh.write("," + str(i))
                fh.write('\n')
