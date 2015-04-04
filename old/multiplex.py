# how many Xs do we need?
# load rw#.py first
# code not designed for re-usability, edit with caution

import scipy
from scipy import stats
import pickle

numsamples=100
numx=3
theta=.5
numgraphs=100 # graphs per sample
costs=[]
est_costs=[]

def gen_set(numsamples):
    galist=[genG(numnodes,numlinks,probRewire) for i in range(numsamples)]
    glist=[i[0] for i in galist]
    alist=[i[1] for i in galist]

    xlist=[[genX(g) for i in range(numx)] for g in glist]
    return alist, xlist

def load_set():
    print "Loading data... (this may take some time)"
    f=open('data.pickle')
    pickleobj=pickle.load(f)
    #[alist,xlist,z0,z1,z2,a0,a1,a2,costs0,costs1,costs1,est_costs0,est_costs1,est_costs2,realgraphcost0,realgraphcost1,realgraphcost2,spearman,rank]
    return pickleobj

def write_set():
    f=open('data.pickle','w')
    pickleobj=[alist,xlist,z0,z1,z2,a0,a1,a2,costs0,costs1,costs1,est_costs0,est_costs1,est_costs2,realgraphcost0,realgraphcost1,realgraphcost2,spearman,rank]
    pickle.dump(pickleobj,f)
    f.close()

# onex is index of x in xlist, or pass nothing to use entire list
def multipleX(numsamples,numgraphs,onex='all'):
    # estimated cost of graph
    if onex=='all':
        realgraphcost=[logprobG(a,xlist[i]) for i, a in enumerate(alist)]
        zok=[[reduce(operator.add,[genZfromX(lx,theta) for lx in x]) for i in range(numgraphs)] for x in xlist]
    else:
        realgraphcost=[logprobG(a,[xlist[i][onex]]) for i, a in enumerate(alist)]
        zok=[[genZfromX(x[onex],theta) for i in range(numgraphs)] for x in xlist]
    aok=[[genGfromZ(z) for z in zlist] for zlist in zok]
    costs=[[sum(sum(np.array(abs(i-a)))) for i in aok[j]] for j, a in enumerate(alist)]
    est_costs=[[] for i in range(numgraphs)]
    for i in range(numsamples):
        for g in aok[i]:
            if onex=='all':
                est_costs[i].append(logprobG(g,xlist[i]))
            else:
                est_costs[i].append(logprobG(g,[xlist[i][onex]]))
        print i

    spearman=[]
    for i in range(numgraphs):
        a=scipy.stats.spearmanr(costs[i],est_costs[i])[0]
        spearman.append(a)
    avg=sum([spearman[i] for i in range(numgraphs)])/float(numsamples)

    rank=[]
    for i in range(numgraphs):
        a=sum([realgraphcost[i]>j for j in est_costs[i]])
        rank.append(a)
    r=sum([rank[i] for i in range(numgraphs)])/float(numsamples)
    return avg, r

#>>> avg0
#-0.31271166618283736
#>>> avg1
#-0.32304235899238548
#>>> avg2
#-0.45245124017521454

#r0 -> ~33, r1 -> ~33, r2 -> ~95.5

# using 3 Xs
#>>> r
#99.99
#>>> avg
#-0.46667022916577111

