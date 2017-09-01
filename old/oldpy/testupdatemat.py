graph_seed=1
x_seed=1
g,a=rw.genG(toygraphs,seed=graph_seed)
[Xs,irts.data]=zip(*[rw.genX(g, td, seed=x_seed+i) for i in range(td.numx)])
Xs=list(Xs)
irts.data=list(irts.data)
[Xs,irts.data,alter_graph]=rw.trimX(td.trim,Xs,irts.data)      # trim data when necessary
irts.data=rw.stepsToIRT(irts, seed=x_seed)

import datetime
prior=rw.genSWPrior(toygraphs)

p1, origmat=rw.probX(Xs, a, td, returnmat=1, irts=irts, prior=prior)
a[11,20]=1-a[11,20]
a[20,11]=1-a[20,11]
changed=[11,20]
datetime.datetime.now().time()
p2, n1=rw.probX(Xs, a, td, returnmat=1, irts=irts, prior=prior)
datetime.datetime.now().time()
p3, n2=rw.probX(Xs, a, td, returnmat=1, changed=changed, origmat=origmat, irts=irts, prior=prior)
datetime.datetime.now().time()
