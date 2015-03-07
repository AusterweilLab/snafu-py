# how many Xs do we need?
# load rw#.py first`
# code not designed for re-usability

import scipy
import pickle

def logprobG(graph,Xs):
    probG=0
    for x in Xs:
        result=[]
        zGs=genz.threadZ(x,graph,numsamples)
        loglist=[logprobZ(i,graph) for i in zGs]
        logmax=max(loglist)
        loglist=[i-logmax for i in loglist]                          # log trick: subtract off the max
        probZG=math.log(sum([math.e**i for i in loglist])) + logmax  # add it back on
        probG=probG+probZG
    return probG

galist=[]
for numnodes in range(10,41):
    galist.append(genG(numnodes,numlinks,probRewire))

glist=[i[0] for i in galist]
alist=[i[1] for i in galist]

numx=2
xlist=[[genX(g) for i in range(numx)] for g in glist]

# real graph cost is 0, these are estimated costs
realgraphcost0=[logprobG(a,[xlist[i][0]]) for i, a in enumerate(alist)]
realgraphcost1=[logprobG(a,[xlist[i][1]]) for i, a in enumerate(alist)]
realgraphcost2=[logprobG(a,xlist[i]) for i, a in enumerate(alist)]

theta=.5
numgraphs=100
z0=[[genZfromX(x[0],theta) for i in range(numgraphs)] for x in xlist]
z1=[[genZfromX(x[1],theta) for i in range(numgraphs)] for x in xlist]
z2=[[reduce(operator.add,[genZfromX(lx,theta) for lx in x]) for i in range(numgraphs)] for x in xlist]

a0=[[genGfromZ(z) for z in zlist] for zlist in z0]
a1=[[genGfromZ(z) for z in zlist] for zlist in z1]
a2=[[genGfromZ(z) for z in zlist] for zlist in z2]
    
costs0=[[sum(sum(np.array(abs(i-a)))) for i in a0[j]] for j, a in enumerate(alist)]
costs1=[[sum(sum(np.array(abs(i-a)))) for i in a1[j]] for j, a in enumerate(alist)]
costs2=[[sum(sum(np.array(abs(i-a)))) for i in a2[j]] for j, a in enumerate(alist)]

est_costs0=[[] for i in range(numgraphs)]
est_costs1=[[] for i in range(numgraphs)]
est_costs2=[[] for i in range(numgraphs)]

for i in range(100):
    for g in a0[i]:
        est_costs0[i].append(logprobG(g,[xlist[i][0]]))
    print i

for i in range(100):
    for g in a1[i]:
        est_costs1[i].append(logprobG(g,[xlist[i][1]]))
    print i

for i in range(100):
    for g in a2[i]:
        est_costs2[i].append(logprobG(g,xlist[i]))
    print i

spearman=[]
for i in range(100):
    a=scipy.stats.spearmanr(costs0[i],est_costs0[i])[0]
    b=scipy.stats.spearmanr(costs1[i],est_costs1[i])[0]
    c=scipy.stats.spearmanr(costs2[i],est_costs2[i])[0]
    spearman.append((a,b,c))

avg0=sum([spearman[i][0] for i in range(100)])/100
avg1=sum([spearman[i][1] for i in range(100)])/100
avg2=sum([spearman[i][2] for i in range(100)])/100

#>>> avg0
#-0.31271166618283736
#>>> avg1
#-0.32304235899238548
#>>> avg2
#-0.45245124017521454

rank=[]
for i in range(100):
    a=sum([realgraphcost0[i]>j for j in est_costs0[i]])
    b=sum([realgraphcost1[i]>j for j in est_costs1[i]])
    c=sum([realgraphcost2[i]>j for j in est_costs2[i]])
    rank.append((a,b,c))

r0=sum([rank[i][0] for i in range(100)])/100
r1=sum([rank[i][1] for i in range(100)])/100
r2=sum([rank[i][2] for i in range(100)])/100

#>>> r0
#33
#>>> r1
#32
#>>> r2
#95

f=open('data.pickle','w')
pickleobj=[alist,xlist,z0,z1,z2,a0,a1,a2,costs0,costs1,costs1,est_costs0,est_costs1,est_costs2,realgraphcost0,realgraphcost1,realgraphcost2,spearman,rank]
pickle.dump(pickleobj,f)
f.close()
