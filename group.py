import pickle
import rw
import numpy as np

# slop
fh=open("count.txt","r")
count=fh.readline()
count=float(count.rstrip())
fh.close()
fh=open("count.txt","w")
newcount=count+0.1

if newcount > 0.5:
    newcount = 0.5

fh.write(str(newcount))
fh.close()

fitinfo_zibb=rw.Fitinfo({
        'prior_method': "zeroinflatedbetabinomial",
        'zib_p': .3,
        'prior_a': 2,
        'prior_b': 1,
        'startGraph': "goni_valid",
        'goni_size': 2,
        'goni_threshold': 2,
        'followtype': "avg", 
        'prune_limit': np.inf,
        'triangle_limit': np.inf,
        'other_limit': np.inf })

fitinfo_bb=rw.Fitinfo({
        'prior_method': "betabinomial",
        'prior_a': 1,
        'prior_b': .75,
        'startGraph': "goni_valid",
        'goni_size': 2,
        'goni_threshold': 2,
        'followtype': "avg", 
        'prune_limit': np.inf,
        'triangle_limit': np.inf,
        'other_limit': np.inf })

graphs=[]
items=[]
numsubs=50
for sub in range(numsubs):
    filename = "subs/h_" + str(sub) + ".pickle"
    fh=open(filename,"r")
    alldata=pickle.load(fh)
    fh.close()
    graphs.append(alldata['graph'])
    items.append(alldata['items'])

priordict = rw.genGraphPrior(graphs, items, fitinfo=fitinfo_zibb, returncounts=True)

# only want groupitems
subs=["S"+str(i) for i in range(101,151)]
filepath = "../Spring2017/results_clean.csv"
category="animals"
Xs, noitems, irtdata, numnodes, groupitems, groupnumnodes = rw.readX(subs,category,filepath,removePerseverations=True,spellfile="categories/zemla_spellfile.csv")

fh=open("prior.pickle","w")
alldata={}
alldata['graph']=priordict
alldata['items']=groupitems
pickle.dump(alldata,fh)
fh.close()

