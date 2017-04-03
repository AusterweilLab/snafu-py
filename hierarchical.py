import networkx as nx
import rw
import numpy as np

subs=['S101','S102','S103','S104','S105','S106','S107','S108','S109','S110',
      'S111','S112','S113','S114','S115','S116','S117','S118','S119','S120']
#subs=['S1','S2','S3','S4','S5','S7','S8','S9','S10','S11','S12','S13']

toydata=rw.Toydata({
        'numx': 3,
        'trim': 1,
        'jump': 0.0,
        'jumptype': "stationary",
        'priming': 0.0,
        'startX': "stationary"})

fitinfo=rw.Fitinfo({
        'startGraph': "windowgraph_valid",
        'windowgraph_size': 2,
        'windowgraph_threshold': 2,
        'followtype': "avg", 
        'prior_samplesize': 10000,
        'recorddir': "records/",
        'directed': False,
        'prune_limit': np.inf,
        'triangle_limit': np.inf,
        'other_limit': np.inf})

toygraphs=rw.Toygraphs({
        'numgraphs': 1,
        'graphtype': "steyvers",
        'numnodes': 280,
        'numlinks': 6,
        'prob_rewire': .3})

irts=rw.Irts({
        'data': [],
        'irttype': "exgauss",
        'lambda': 0.721386887,
        'sigma': 6.58655566,
        'irt_weight': 0.95,
        'rcutoff': 20})

prior_graphs=[]
prior_items=[]
ss_Xs=[]
ss_numnodes=[]

# generate starting graphs
for subj in subs:
    category="animals"
    Xs, items, irts.data, numnodes=rw.readX(subj,category,'./Spring2015/results_cleaned.csv',ignorePerseverations=True)

    window_graph=rw.windowGraph(Xs, numnodes, td=toydata, valid=True, fitinfo=fitinfo)
    prior_graphs.append(window_graph)
    prior_items.append(items)
    ss_Xs.append(Xs)
    ss_numnodes.append(numnodes)

# initialize prior
priordict = rw.genGraphPrior(prior_graphs, prior_items)
changesmade = 1
rnd = 1
original_order=subs[:]

while changesmade > 0:
    print "round:", rnd, " graphs changed: ", changesmade
    rnd += 1
    changesmade=0
    np.random.shuffle(subs)

    for subj in subs:
        print "ss: ", subj
        subj_idx = original_order.index(subj)
        toydata.numx = len(ss_Xs[subj_idx])
        fitinfo.startGraph = prior_graphs[subj_idx]

        prior = (priordict, prior_items[subj_idx])

        # find best graph
        uinvite_graph, bestval=rw.uinvite(ss_Xs[subj_idx], toydata, ss_numnodes[subj_idx], fitinfo=fitinfo, prior=prior)

        ## update prior if graph has changed
        if not np.array_equal(uinvite_graph, prior_graphs[subj_idx]):
            changesmade += 1
            prior_graphs[subj_idx] = uinvite_graph
            priordict = rw.genGraphPrior(prior_graphs, prior_items)

for subj in range(len(subs)):
    g=nx.to_networkx_graph(prior_graphs[subj])
    g2=nx.to_networkx_graph(rw.noHidden(ss_Xs[subj],ss_numnodes[subj]))
    nx.relabel_nodes(g, prior_items[subj], copy=False)
    nx.relabel_nodes(g2, prior_items[subj], copy=False)
    rw.write_csv([g, g2],subj+".csv",subj) # write multiple graphs
    
