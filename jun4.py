import numpy as np
from numpy.linalg import inv
from rw.rw import *
#from operator import mul

def logTrick(loglist):
    logmax=max(loglist)
    loglist=[i-logmax for i in loglist]                     # log trick: subtract off the max
    p=math.log(sum([math.e**i for i in loglist])) + logmax  # add it back on
    return p

# http://stackoverflow.com/questions/350519/getting-the-lesser-n-elements-of-a-list-in-python
def maxs(items,n):
    maxs = items[:n]
    maxs.sort(reverse=True)
    for i in items[n:]:
        if i > maxs[-1]: 
            maxs.append(i)
            maxs.sort(reverse=True)
            maxs= maxs[:n]
    return maxs

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
    for i in range(len(probs)):
        probs[i]=sum([math.log(j) for j in probs[i]])
    probs=sum(probs)
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

            for r in range(0,maxlen):
                Q=np.linalg.matrix_power(oldQ,r)
                sumlist=[]
                for k in range(numcols):
                    num1=Q[k,startindex]
                    num2=t[x[curpos],notdeleted[k]]
                    if ((num1>0) and (num2>0)):
                        tmp=num1*num2
                        sumlist.append(tmp)
                innersum=sum(sumlist)
                alpha=r+1
                gamma=alpha*math.log(beta)-math.lgamma(alpha)+(alpha-1)*math.log(irt)-beta*irt*beta
                if innersum > 0:
                    flist.append(gamma*(1-jeff)+jeff*math.log(innersum))
                    #print "innersum=", math.log(innersum), " gamma=", gamma, " ratio=", math.log(innersum)/gamma
            f=sum([math.e**i for i in flist])
            prob.append(f)
        if 0.0 in prob: 
            #print "Warning: Zero-probability transition; graph cannot produce X"
            return -1000
        probs.append(prob)
    for i in range(len(probs)):
        probs[i]=sum([math.log(j) for j in probs[i]])
    probs=sum(probs)
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

#numnodes=53                           # number of nodes in graph
numnodes=25
numlinks=4                            # initial number of edges per node (must be even)
probRewire=.2                         # probability of re-wiring an edge
numedges=numnodes*(numlinks/2)        # number of edges in graph

theta=.5                # probability of hiding node when generating z from x (rho function)
#numx=3                  # number of Xs to generate
numx=1
numsamples=100          # number of sample z's to estimate likelihood
numgraphs=100
trim=1
maxlen=20               # no closed form, number of times to sum over
jeff = .99
maxtokeep=10
numperseed=10
nodestotweak=[1]
graphtype='star' # 'star' or 'grid'

star_g=nx.star_graph(numnodes-1)                                         
star_a=np.array(nx.adjacency_matrix(star_g).todense(), dtype=np.int32)  
grid_g=nx.grid_graph(dim=[5,5])
grid_a=np.array(nx.adjacency_matrix(grid_g).todense(), dtype=np.int32) 

if graphtype=='star':
    g=star_g
    a=star_a
else:
    g=grid_g
    a=grid_a

Xs=[genX(g) for i in range(numx)]
#[Xs,g,a,numnodes]=trimX(trim,Xs,g,a,numnodes)

allnodes=[(i,j) for i in range(len(a)) for j in range(len(a)) if (i!=j) and (i>j)]
expected_irts=expectedHidden(Xs,a)

graphs=genGraphs(numgraphs,theta,Xs,numnodes)

# hill climbing with stochastic search
def genFromSeeds(seedgraphs,numperseed,nodestotweak):
    graphs=seedgraphs[:]
    for i in seedgraphs:
        for j in range(numperseed):
            new=np.copy(i)
            for k in range(random.choice(nodestotweak)):
                rand1=rand2=0
                while (rand1 == rand2):
                    rand1=random.randint(0,len(i)-1)
                    rand2=random.randint(0,len(i)-1)
                new[rand1,rand2]=1-new[rand1,rand2]
                new[rand2,rand1]=1-new[rand2,rand1]
            graphs.append(new)
    return graphs

# hill climbing with stochastic search
def xBest(graphs,numkeep):
    ours=[]
    his=[]
    true=[]

    for it, graph in enumerate(graphs):
        tmp=probX(Xs,graph,expected_irts)
        ours.append(tmp)

        #tmp=origProbX(Xs,graph)
        #for i in range(len(tmp)):
        #    tmp[i]=sum(tmp[i])
        #tmp=sum(tmp)
        #his.append(tmp)

        true.append(cost(graph,a))
        #if (it % 10) == 0:
        #    print it
    
    maxvals=maxs(ours,numkeep)
    maxpos=[ours.index(i) for i in maxvals]
    maxgraphs=[]
    for i in maxpos:
        maxgraphs.append(graphs[i])
    print "MAX: ", max(ours), "COST: ", cost(graphs[ours.index(max(ours))],a)
    return maxgraphs, max(ours)

#def beam(graphs,keep):
#    for i in 
#    pass 

rep=0
prev_maxnum=0
while rep < 5:
    [graphs,maxnum]=xBest(graphs,1)
    if maxnum==prev_maxnum:
        rep += 1
    else:
        rep = 0
    prev_maxnum=maxnum
    graphs=genFromSeeds(graphs,100,nodestotweak)
    print "REP: ", rep

