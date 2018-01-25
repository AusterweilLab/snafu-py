import rw
import numpy as np
import pickle

subfiles=["h_"+str(i)+".pickle" for i in range(50)]
category="animals"
filepath="../Spring2017/results_clean.csv"

td=rw.Data({
        'startX': "stationary",
        'numx': 3,
        'jumptype': "stationary",
        'jump': 0.1
        })

ll_change = dict()

# load prior
with open('prior.pickle','r') as priorfile:
    prior = pickle.load(priorfile)
    prior = (prior['graph'], prior['items'])

for subnum, subfile in enumerate(subfiles):
    sub = "S" + str(101+subnum)   # fluency subs start here
    
    # load fluency data
    Xs, items, irtdata, numnodes = rw.readX(sub,category,filepath,removePerseverations=True,spellfile="schemes/zemla_spellfile.csv")

    # load graph
    fh=open('subs/'+subfile,'r')
    alldata=pickle.load(fh)
    fh.close()
    graph=alldata['graph']
    items=alldata['items']

    ll_orig, uinvite_probs = rw.probX(Xs, graph, td, prior=prior)

    # try each edge flip
    for item1 in range(len(graph)):
        for item2 in range(len(graph)):
            if item1 > item2:
                item1label = items[item1]
                item2label = items[item2]
                
                # new graph flip edge
                a=np.copy(graph)
                a[item1][item2] = 1-a[item1][item2]
                a[item2][item1] = 1-a[item2][item1]
                ll, uinvite_probs = rw.probX(Xs, a, td, prior=prior)
                
                if item1label not in ll_change:
                    ll_change[item1label]={}
                if item2label not in ll_change[item1label]:
                    ll_change[item1label][item2label] = []

                #print item1label, item2label, ll_orig, ll
                ll_change[item1label][item2label].append(ll_orig - ll)    # lower is better


# 182 entries with inf???
# find average change in graph -LL
for i in ll_change:
    for j in ll_change[i]:
        ll_change[i][j] = np.mean(ll_change[i][j])

outfile="llchanges.csv"
fo=open(outfile,'w')

for i in ll_change:
    for j in ll_change[i]:
        if i<j:
            line="S100,"+i+","+j+","
        else:
            line="S100,"+j+","+i+","
        line += str(ll_change[i][j]) + "," + "\n"
        fo.write(line)

fo.close()
