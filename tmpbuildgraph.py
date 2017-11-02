import rw
import pickle
import numpy as np

graphs=[]
items=[]

fitinfo=rw.Fitinfo({
        'prior_method': "betabinomial",  # can also be zeroinflatedbetabinomial
        'prior_a': 1,                    # BB(1,1) is 50/50 prior, ZIBB(2,1,.5) is 50/50 prior
        'prior_b': 1,                    # note that graphs are sparse, so 50/50 isn't accurate...
        'startGraph': "goni_valid",
        'goni_size': 2,
        'goni_threshold': 2,
        'followtype': "avg", 
        'prune_limit': np.inf,
        'triangle_limit': np.inf,
        'other_limit': np.inf })

subs=range(50)
for i in subs:
    fh=open('goniprior/h_'+str(i)+'.pickle')
    alldata=pickle.load(fh)
    graphs.append(alldata['graph'])
    items.append(alldata['items'])

subs=["S"+str(i) for i in range(101,151)]
filepath = "../Spring2017/results_clean.csv"
category="animals"

Xs, noitems, irtdata, numnodes, groupitems, groupnumnodes = rw.readX(subs,category,filepath,removePerseverations=True,spellfile="schemes/zemla_spellfile.csv")

priordict = rw.genGraphPrior(graphs, items, fitinfo)
#groupgraph = rw.priorToGraph(priordict, groupitems)

alldata={}
alldata['graph']=priordict
alldata['items']=groupitems

fh=open('goniprior.pickle','w')
pickle.dump(alldata,fh)
fh.close()
