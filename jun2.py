import numpy as np
from numpy.linalg import inv
from rw.rw import *
#from operator import mul

def probX(Xs, a, irts):
    probs=[]
    expecteds=[]
    for xnum, x in enumerate(Xs):
        prob=[]
        expected=[]
        for curpos in range(1,len(x)-1):
            irt=irts[xnum][curpos-1]
            t=a/sum(a.astype(float))            # transition matrix (from: column, to: row)
            Q=np.copy(t)
    
            notinx=[]       # nodes not in trimmed X
            for i in range(numnodes):
                if i not in x:
                    notinx.append(i)

            startindex=x[curpos-1]
            deletedlist=sorted(x[curpos:]+notinx,reverse=True)
            notdeleted=[i for i in range(53) if i not in deletedlist]

            for i in deletedlist:  # to form Q matrix
                Q=np.delete(Q,i,0) # delete row
                Q=np.delete(Q,i,1) # delete column
            startindex = startindex-sum([startindex > i for i in deletedlist])

            numcols=np.shape(Q)[1]
            beta=1  # free parameter
            flist=[]
            for r in range(1,maxlen):
                rollingsum=0
                for k in range(numcols):
                    rollingsum = rollingsum + Q[k,startindex] * t[x[curpos],notdeleted[k]]
                    print Q[k,startindex]
                
                gamma=math.log(beta**r)-math.lgamma(r)+(r-1)*math.log(irt)-beta*irt
                print rollingsum, gamma
                flist.append(gamma+math.log(rollingsum))
                Q=Q*Q
            logmax=max(flist)
            flist=[i-logmax for i in flist]                          # log trick: subtract off the max
            f=math.log(sum([math.e**i for i in flist])) + logmax  # add it back on
            print f
            prob.append(f)
        if 0.0 in prob: 
            print "Warning: Zero-probability transition? Check graph to make sure X is possible."
            raise
        probs.append(prob)
    return probs

def expectedHidden(Xs, a):
    expecteds=[]
    for x in Xs:
        expected=[]
        for curpos in range(1,len(x)-1):
            t=a/sum(a.astype(float))            # transition matrix (from: column, to: row)
            Q=np.copy(t)
             
            notinx=[]       # nodes not in trimmed X
            for i in range(numnodes):
                if i not in x:
                    notinx.append(i)
            
            startindex=x[curpos-1]
            deleted=0
            for i in sorted(x[curpos:]+notinx,reverse=True):   # to form Q matrix
                if i < startindex:
                    deleted += 1
                Q=np.delete(Q,i,0) # delete row
                Q=np.delete(Q,i,1) # delete column
            I=np.identity(len(Q))
            N=inv(I-Q)
            expected.append(sum(N[:,startindex-deleted]))
        expecteds.append(expected)        
    return expecteds

numnodes=53                           # number of nodes in graph
numlinks=4                            # initial number of edges per node (must be even)
probRewire=.2                         # probability of re-wiring an edge
numedges=numnodes*(numlinks/2)        # number of edges in graph

theta=.5                # probability of hiding node when generating z from x (rho function)
numx=3                  # number of Xs to generate
numsamples=100          # number of sample z's to estimate likelihood
numgraphs=1000
trim=.7
match_numnodes=1        # make sure trimmed graph has numnodes... else it causes problems. fix later. or keep, maybe not a bad idea.
maxlen=100              # no closed form, number of times to sum over

while match_numnodes:
    g,a=genG(numnodes,numlinks,probRewire) 
    Xs=[genX(g) for i in range(numx)]
    [Xs,g,a,numnodes]=trimX(trim,Xs,g,a,numnodes)
    if 0 not in sum(a):
        match_numnodes=0

ours=[]
true=[]

for it, graph in enumerate(genGraphs(numgraphs,theta,Xs,numnodes)):
    expected_irts=expectedHidden(Xs,a)
    ours=probX(Xs,graph,expected_irts)
    #for i in range(len(tmp)):
    #    tmp[i]=[math.log(j) for j in tmp[i]]
    #    tmp[i]=sum(tmp[i])
    #tmp=sum(tmp)
    #his.append(tmp)
    true.append(cost(graph,a))
    
    print it

