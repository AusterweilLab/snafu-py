from numpy.linalg import inv
from rw.rw import *
#from operator import mul

def probX(Xs, a):
    probs=[]
    for x in Xs:
        prob=[]
        for curpos in range(1,len(x)-1):
            t=a/sum(a.astype(float))            # transition matrix (from: column, to: row)
            Q=np.copy(t)
            
            for i in sorted(x[curpos:],reverse=True):   # to form Q matrix
                Q=np.delete(Q,i,0) # delete row
                Q=np.delete(Q,i,1) # delete column
            I=np.identity(len(Q))
            N=inv(I-Q)
            
            R=np.copy(t)
            for i in reversed(range(len(x))):
                if i in x[curpos:]:
                    R=np.delete(R,i,1) # columns are already visited nodes
                else:
                    R=np.delete(R,i,0) # rows are absorbing/unvisited nodes
            print R
            print N
            B=np.dot(R,N)
            startindex=sorted(x[:curpos]).index(x[curpos-1])
            absorbingindex=sorted(x[curpos:]).index(x[curpos])
            prob.append(B[absorbingindex,startindex])
        if 0.0 in prob: 
            print "Warning: Zero-probability transition? Check graph to make sure X is possible."
            raise
        probs.append(prob)
    return probs

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

while match_numnodes:
    g,a=genG(numnodes,numlinks,probRewire) 
    Xs=[genX(g) for i in range(numx)]
    #[Xs,g,a,numnodes]=trimX(trim,Xs,g,a,numnodes)
    if 0 not in sum(a):
        match_numnodes=0

his=[]
ours=[]
true=[]

for it, graph in enumerate(genGraphs(numgraphs,theta,Xs,numnodes)):
    tmp=probX(Xs,graph)
    for i in range(len(tmp)):
        tmp[i]=[math.log(j) for j in tmp[i]]
        tmp[i]=sum(tmp[i])
    tmp=sum(tmp)
    his.append(tmp)
    true.append(cost(graph,a))
    ours.append(logprobG(graph,Xs,numsamples))
    print it
