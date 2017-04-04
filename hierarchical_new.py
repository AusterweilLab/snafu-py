import networkx as nx
import rw
import numpy as np
import pickle


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

graphs=[]
items=[]
Xs=[]
numnodes=[]

# generate starting graphs
for subj in subs:
    category="animals"
    Xs, items, irts.data, numnodes=rw.readX(subj,category,'./Spring2015/results_cleaned.csv',ignorePerseverations=True)
    items.append(items)
    Xs.append(Xs)
    numnodes.append(numnodes)

graphs, priordict = hierarchicalUinvite(Xs, items, numnodes, toydata, fitinfo=Fitinfo({}), seed=None):

for subj in subs:
    g=nx.to_networkx_graph(graphs[subj])
    g2=nx.to_networkx_graph(rw.noHidden(Xs[subj],numnodes[subj]))
    nx.relabel_nodes(g, items[subj], copy=False)
    nx.relabel_nodes(g2, items[subj], copy=False)
    rw.write_csv([g, g2],subj+".csv",subj) # write multiple graphs
    
with open('prior.pickle','w') as fh:
    pickle.dump(priordict, fh)
