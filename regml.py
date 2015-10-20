Xs=[rw.genX(g, seed=x_seed+i) for i in range(numx)]

def PfromB(b):
    P=np.empty([numnodes,numnodes],dtype=float)  # set P from b (transition matix P[to,from])
    for colnum, col in enumerate(b.T):
        for rownum, row in enumerate(col):
            P[rownum,colnum]=math.exp(row)/sum([math.exp(i) for i in col])
    return P

def regml():
    # free parameters
    c=.75       # value used by Kwang and Bottou, 2012
    cb=.5       # regularization term (value?)
    gamma=.5    # value?
    epochs=100  # value?

    # fixed parameters
    a=cb/len(Xs)
    
    b=np.random.rand(numnodes,numnodes)          # initialize b0
    for i in range(len(b)):                      # set diagonal to infinity
        b[i,i]=-np.inf
    avg_b=np.copy(b)                             # represents best solution at each step
    
    t=1
    while t < epochs:
        random.shuffle(Xs) 
        eta= gamma * (1 + gamma*a*t)**(-c)
        P=PfromB(b)  # set transition matrix from b
        
        for x in Xs:

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

            for i in range(len(P):
                for j in range(len(P)):

                    # matrix of non-absorbing states
                    Q=np.delete(P, i, 0) # delete absorbing row
                    Q=np.delete(P, i, 1) # delete absorbing column

                    # matrix of absorbing states
                    I=1         # only one absorbing state. does treating as scalar mess things up though?

                    # fundamental matrix
                    N=(I-Q)**-1

                    QNR=-1*Q*N*R
                    QNR[i,1] +

                

        b = b - eta * {derivative}
        avg_b = ((t-1)/t)*avg_b + (1/t)*b
        t += 1


    
    
