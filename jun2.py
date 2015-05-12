import numpy as np
from numpy.linalg import inv
from rw.rw import *
#from operator import mul

def logTrick(loglist):
    logmax=max(loglist)
    loglist=[i-logmax for i in loglist]                     # log trick: subtract off the max
    p=math.log(sum([math.e**i for i in loglist])) + logmax  # add it back on
    return p
        
def origProbX(Xs, a):
    probs=[]
    expecteds=[]
    for x in Xs:
        prob=[]
        expected=[]
        for curpos in range(1,len(x)):
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

            R=np.copy(t)
            for i in reversed(range(numnodes)):
                if i in notinx:
                    R=np.delete(R,i,1)
                    R=np.delete(R,i,0)
                elif i in x[curpos:]:
                    R=np.delete(R,i,1) # columns are already visited nodes
                else:
                    R=np.delete(R,i,0) # rows are absorbing/unvisited nodes
            B=np.dot(R,N)
            startindex=sorted(x[:curpos]).index(x[curpos-1])
            absorbingindex=sorted(x[curpos:]).index(x[curpos])
            prob.append(B[absorbingindex,startindex])
        if 0.0 in prob: 
            print "Warning: Zero-probability transition? Check graph to make sure X is possible."
            raise
        probs.append(prob)
        expecteds.append(expected)
    return probs

def probX(Xs, a, irts):
    underflow=0
    probs=[]
    for xnum, x in enumerate(Xs):
        prob=[]
        for curpos in range(1,len(x)):
            irt=irts[xnum][curpos-1]
            t=a/sum(a.astype(float))            # transition matrix (from: column, to: row)
            Q=np.copy(t)
    
            notinx=[]       # nodes not in trimmed X
            for i in range(numnodes):
                if i not in x:
                    notinx.append(i)

            startindex=x[curpos-1]
            deletedlist=sorted(x[curpos:]+notinx,reverse=True) # Alternatively: x[curpos:]+notinx OR [x[curpos]]
            notdeleted=[i for i in range(numnodes) if i not in deletedlist]
            for i in deletedlist:  # to form Q matrix
                Q=np.delete(Q,i,0) # delete row
                Q=np.delete(Q,i,1) # delete column
            startindex = startindex-sum([startindex > i for i in deletedlist])

            numcols=np.shape(Q)[1]
            beta=1  # free parameter
            flist=[]
            oldQ=np.copy(Q)

            for r in range(1,maxlen):
                Q=np.linalg.matrix_power(oldQ,r)
                sumlist=[]
                for k in range(numcols):
                    num1=Q[k,startindex]
                    num2=t[x[curpos],notdeleted[k]]
                    if ((num1>0) and (num2>0)):
                        tmp=num1*num2
                        if (tmp==0): 
                            print "Warning: Underflow"
                            #tmp=math.e**logTrick([math.log(num1),math.log(num2)])
                            #if (tmp==0):
                            #    print "Warning: Double Underflow"
                            #    underflow=1
                            tmp=0 # just ignore it for now...
                        sumlist.append(tmp)
                innersum=sum(sumlist)
                gamma=r*math.log(beta)-math.lgamma(r)+(r-1)*math.log(irt)-beta*irt
                if innersum > 0:
                    flist.append(gamma+math.log(innersum))
                else:
                    flist.append(gamma)
            f_tmp=[math.e**i for i in flist]
            if 0.0 in f_tmp:
                print "bad"
            f=sum(f_tmp)
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
        for curpos in range(1,len(x)):
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
numgraphs=100
trim=.7
match_numnodes=1        # make sure trimmed graph has numnodes... else it causes problems. fix later. or keep, maybe not a bad idea.
maxlen=20               # no closed form, number of times to sum over

while match_numnodes:
    g,a=genG(numnodes,numlinks,probRewire) 
    Xs=[genX(g) for i in range(numx)]
    [Xs,g,a,numnodes]=trimX(trim,Xs,g,a,numnodes)
    if 0 not in sum(a):
        match_numnodes=0

ours=[]
his=[]
true=[]

expected_irts=expectedHidden(Xs,graph)

for it, graph in enumerate(genGraphs(numgraphs,theta,Xs,numnodes)):
    tmp=probX(Xs,graph,expected_irts)
    for i in range(len(tmp)):
        tmp[i]=sum(tmp[i])
    tmp=sum(tmp)
    ours.append(tmp)

    tmp=origProbX(Xs,graph)
    for i in range(len(tmp)):
        tmp[i]=sum(tmp[i])
    tmp=sum(tmp)
    his.append(tmp)

    true.append(cost(graph,a))
    print it

