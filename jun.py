from numpy.linalg import inv
from rw.rw import *
#from operator import mul

def probX(Xs, a):
    probs=[]
    expecteds=[]
    for x in Xs:
        prob=[]
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
    return probs, expecteds

def probIRT(Xs, a):
    expected_hidden=[]
    for x in Xs:
        num_hidden=[]
        for curpos in range(1,len(x)-1):
            # want expected number of hidden nodes given the next in
            # sequence... so make sure there is only one absorbing node
            Q=np.copy(a)
           
            notinx=[]       # nodes not in trimmed X
            for i in range(numnodes):
                if i not in x:
                    notinx.append(i)
            
            absorbingnode=x[curpos]
            startingnode=x[curpos-1]
            absorb_offset=0
            start_offset=0
            for i in sorted(x[curpos+1:]+notinx,reverse=True):   # to form Q matrix (note curpos+1 until transition matrix is formed)
                if i < absorbingnode:
                    absorb_offset=absorb_offset+1
                if i < startingnode:
                    start_offset=start_offset+1
                Q=np.delete(Q,i,0) # delete row
                Q=np.delete(Q,i,1) # delete column

            if absorbingnode < startingnode:
                start_offset += 1
            absorbingnode=absorbingnode-absorb_offset
            
            # shitty code is easier to write
            t=Q/sum(Q.astype(float))            # transition matrix (from: column, to: row)
            t=np.delete(t,absorbingnode,1)
            t=np.delete(t,absorbingnode,0)
            
            I=np.identity(len(t))
            N=inv(I-t)
            startindex=startingnode-start_offset
            
            num_hidden.append(sum(N[:,startindex]))
        expected_hidden.append(num_hidden)
    return expected_hidden

def probGamma(expected_hidden,observed_irts):
    beta=1  # free parameter
    prob=0
    for i in range(len(observed_irts)): # alpha is expected hidden nodes
        for (alpha, x) in zip(expected_hidden[i],observed_irts[i]):
            #prob=prob+((beta**alpha)/math.gamma(alpha))*x**(alpha-1)*math.e**(-beta*x)
            prob=prob+math.log(beta**alpha)-math.lgamma(alpha)+(alpha-1)*math.log(x)-beta*x
    return prob

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
    [Xs,g,a,numnodes]=trimX(trim,Xs,g,a,numnodes)
    if 0 not in sum(a):
        match_numnodes=0

garbage, observed_irts=probX(Xs,a) # fake IRT data; expected number of hidden nodes between each X
print "real:", probGamma(observed_irts,observed_irts)
alt_irt=probIRT(Xs,a)

his=[]
ours=[]
true=[]
gamma=[]
gamma2=[]

for it, graph in enumerate(genGraphs(numgraphs,theta,Xs,numnodes)):
    tmp, irts=probX(Xs,graph)
    for i in range(len(tmp)):
        tmp[i]=[math.log(j) for j in tmp[i]]
        tmp[i]=sum(tmp[i])
    tmp=sum(tmp)
    his.append(tmp)
    true.append(cost(graph,a))
    ours.append(logprobG(graph,Xs,numsamples))
    
    alt_exp=probIRT(Xs,graph)
    gamma.append(probGamma(irts, observed_irts))
    gamma2.append(probGamma(alt_exp,alt_irt))

    print it

