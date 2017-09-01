import rw
import networkx as nx
import numpy as np
import pickle
import sys

#filename=sys.argv[1]

def distort(graph, numtoreplace=39, stayConnected=True):
    if stayConnected: newgraphlen = 0
    else: newgraphlen = len(graph)  # it's a lie.
    
    while newgraphlen != len(graph):
        newgraph=np.copy(graph)
        
        zeros = zip(*np.where(newgraph==0))
        ones = zip(*np.where(newgraph==1))
        zeros = [(i,j) for (i,j) in zeros if i > j] # avoid self-links and directed edges
        ones = [(i,j) for (i,j) in ones if i > j]
        
        changetoone=np.random.choice(range(len(zeros)),numtoreplace,replace=False)
        changetozero=np.random.choice(range(len(ones)),numtoreplace,replace=False)
        
        for i in changetoone:
            idx=zeros[i]
            idx2=zeros[i][::-1]
            newgraph[idx]=1
            newgraph[idx2]=1
        
        for i in changetozero:
            idx=ones[i]
            idx2=ones[i][::-1]
            newgraph[idx]=0
            newgraph[idx2]=0
        
        newgraph = np.array(nx.to_numpy_matrix(max(nx.connected_component_subgraphs(nx.to_networkx_graph(newgraph)), key=len))).astype(int)
        if stayConnected: newgraphlen = len(newgraph)
    return newgraph

usf_graph, usf_items = rw.read_csv("./snet/USF_animal_subset.snet")
usf_graph_nx = nx.from_numpy_matrix(usf_graph)
usf_numnodes = len(usf_items)

numsubs = 50
numlists = 3
listlength = 35
numsims = 10
methods=['rw','goni','chan','kenett','fe']
#methods=['uinvite']

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

irtdata=[]
ss_graphs=[]
data=[]
items=[]
numnodes=[]
datab=[]

# generate data for `numsub` participants, each having `numlists` lists of `listlengths` items
seednum=0    # seednum=150 (numsubs*numlists) means start at second sim, etc.

for simnum in range(numsims):
    irtdata=[]
    ss_graphs=[]
    data=[]
    items=[]
    numnodes=[]
    datab=[]
    for sub in range(numsubs):
        ss_graphs.append(distort(usf_graph))
        ss_graph_nx = nx.from_numpy_matrix(ss_graphs[-1])
        
        Xs, irts[sub].data = rw.genX(ss_graph_nx, toydata, seed=seednum)[0:2]
        data.append(Xs)
        
        irts[sub].data = rw.stepsToIRT(irts[sub])
        irtdata.append(irts[sub].data)
        
        # renumber dictionary and item list
        itemset = set(rw.flatten_list(Xs))
        numnodes.append(len(itemset))
        
        ss_items = {}
        convertX = {}
        for itemnum, item in enumerate(itemset):
            ss_items[itemnum] = usf_items[item]
            convertX[item] = itemnum
        
        items.append(ss_items)
        
        Xs = [[convertX[i] for i in x] for x in Xs]
        datab.append(Xs)
            
        seednum += numlists
    alldata=dict()
    alldata['usf_graph']=usf_graph
    alldata['usf_items']=usf_items
    alldata['data']=data        # Xs using usf_items as keys
    alldata['datab']=datab      # Xs using recoded keys
    alldata['ss_items'] = items # recoded item dicts
    alldata['ss_irts'] = irtdata 
    alldata['gamma_beta'] = gamma_beta
    alldata['numnodes'] = numnodes
    
    fh=open('./graphs_tofit/simnum_'+str(simnum)+'.pickle',"w")
    pickle.dump(alldata,fh)
    fh.close()

