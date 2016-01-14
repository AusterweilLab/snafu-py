from numpy.linalg import inv
import rw.rw as rw
import numpy as np
import random
import math
import sys

numnodes=20
numlinks=4
probRewire=.3
numx=3
graph_seed=1
x_seed=graph_seed

g,a=rw.genG(numnodes,numlinks,probRewire,seed=graph_seed)
Xs=[rw.genX(g, seed=x_seed+i) for i in range(numx)]

def PfromB(b):
    P=np.empty([numnodes,numnodes],dtype=float)  # set P from b (transition matix P[to,from])
    for colnum, col in enumerate(b.T):
        for rownum, row in enumerate(col):
            P[rownum,colnum]=math.exp(row)/sum([math.exp(i) for i in col])
    return P

def regml():
    # free parameters
    c=.75       # value used by Kwang, and Bottou, 2012
    epochs=20   # value?
    cb=100       # regularization term (value?)
    gamma=.005    # value?

    # fixed parameters
    a=cb/len(Xs)
    
    b=np.random.rand(numnodes,numnodes)          # initialize b0
    avg_b=np.zeros((numnodes,numnodes))            # represents best solution at each step
    for i in range(len(b)):                      # set diagonal to infinity
        b[i,i]=-np.inf
    
    t=1
    while t < epochs:
        print "epoch = ", t
        random.shuffle(Xs) 
        eta= gamma * (1 + gamma*a*t)**(-c)
        derivative_mat = np.zeros((numnodes,numnodes))
         
        for xpos, x in enumerate(Xs):
            print "x =", xpos

            for curpos in range(1,len(x)):
                print "curpos = ", curpos
                P=PfromB(b)  # set transition matrix from b

                # create Q matrix
                Q=np.copy(P)
                notinx=[]       # nodes not in x
                for i in range(numnodes):
                    if i not in x:
                        notinx.append(i)
                startindex=x[curpos-1]
                deletedlist=sorted(x[curpos:]+notinx,reverse=True)
                notdeleted=[i for i in range(numnodes) if i not in deletedlist]
                for i in deletedlist:  # to form Q matrix
                    Q=np.delete(Q,i,0) # delete row
                    Q=np.delete(Q,i,1) # delete column

                # I, N, R
                I=np.identity(len(Q))
                reg=(1+1e-5)                   # nuisance parameter to prevent errors
                N=inv(I*reg-Q)
                R=np.copy(P)
                for i in reversed(range(numnodes)):
                    if i in notinx:
                        R=np.delete(R,i,1)
                        R=np.delete(R,i,0)
                    elif i in x[curpos:]:
                        R=np.delete(R,i,1) # columns are already visited nodes
                    else:
                        R=np.delete(R,i,0) # rows are absorbing/unvisited nodes

                # derivative
                # need tranpose for matrix multiply (switch column and row)
                P=np.transpose(P)
                Q=np.transpose(Q)
                N=np.transpose(N)
                R=np.transpose(R)
                NR=np.dot(N,R)
                QNR=np.dot(Q,NR)
                QNR=-1*QNR
                k=curpos-1
                denominator = NR[k,0]
                
                for si in range(1,curpos):
                    for sj in range(numnodes):
                        i=x[si]
                        j=x[sj]

                        numerator=N[k,si]*P[i,j]
                        first=numerator/denominator

                        if sj>k:
                            derivative_mat[i,j] = derivative_mat[i,j] - QNR[si,0] * first
                        else:
                            if (x[curpos])!=j:
                                derivative_mat[i,j] = derivative_mat[i,j] - first*(QNR[si,0] - NR[sj,0]*P[i,x[curpos]])
                            else:
                                derivative_mat[i,j] = derivative_mat[i,j] - first*(QNR[si,0] + NR[sj,0]*P[i,x[curpos]]*(1/P[i,j]-1))

                    #    print derivative_mat[i,j]
                    #    print i,j,si,sj
                    #    if np.isnan(derivative_mat[i,j]):
                    #        tmp= np.transpose(b)
                    #        print tmp[i,j]
                    #        raise ValueError('NO')

        dermat=np.transpose(derivative_mat)
        b = b - eta * dermat
        avg_b = ((t-1.0)/t)*avg_b + (1.0/t)*b
        t += 1
    return avg_b


## Objective function

#sumlog=0
#for i in range(len(x)-1):
#    sumlog += math.log(P[i+1,i])

# summation on regularizer term
#b2=b**2
#cterm=0
#for i in range(len(b)):
#    for j in range(len(b)):
#        if i!=j:
#            cterm += b2[i,j]

#cterm = cterm * cb/(2*len(Xs))
#obj = -1*sumlog + cterm

#for i in range(len(P):
#    for j in range(len(P)):

newmat=regml()
print PfromB(newmat)