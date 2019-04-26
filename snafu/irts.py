import numpy as np

# returns a vector of how many hidden nodes to expect between each Xi for each X in Xs
def expectedHidden(Xs, a):
    numnodes=len(a)
    expecteds=[]
    t=a/sum(a.astype(float))                      # transition matrix (from: column, to: row)
    identmat=np.identity(numnodes) * (1+1e-10)    # pre-compute for tiny speed-up
    for x in Xs:
        x2=np.array(x)
        t2=t[x2[:,None],x2]                       # re-arrange transition matrix to be in list order
        expected=[]
        for curpos in range(1,len(x)):
            Q=t2[:curpos,:curpos]
            I=identmat[:len(Q),:len(Q)]
            N=np.linalg.solve(I-Q,I[-1])
            expected.append(sum(N))
            #N=inv(I-Q)         # old way, a little slower
            #expected.append(sum(N[:,curpos-1]))
        expecteds.append(expected)        
    return expecteds

# generates fake IRTs from # of steps in a random walk, using gamma distribution
def stepsToIRT(irts, seed=None):
    nplocal=np.random.RandomState(seed)        # to generate the same IRTs each time
    new_irts=[]
    for irtlist in irts.data:
        if irts.irttype=="gamma":
            newlist=[nplocal.gamma(irt, (1.0/irts.gamma_beta)) for irt in irtlist]  # beta is rate, but random.gamma uses scale (1/rate)
        if irts.irttype=="exgauss":
            newlist=[rand_exg(irt, irts.exgauss_sigma, irts.exgauss_lambda) for irt in irtlist] 
        new_irts.append(newlist)
    return new_irts

# ** this function is not really needed anymore since moving functionality to genX, 
# ** but there may be some niche cases where needed...
# trim Xs to proportion of graph size, the trim graph to remove any nodes that weren't hit
# used to simulate human data that doesn't cover the whole graph every time
def trim_lists(trimprop, Xs, steps):
    numnodes=len(Xs[0])             # since Xs haven't been trimmed, we know list covers full graph
    alter_graph_size=0              # report if graph size changes-- may result in disconnected graph!

    if trimprop <= 1:
        numtrim=int(round(numnodes*trimprop))       # if <=1, paramater is proportion of a list
    else:
        numtrim=trimprop                            # else, parameter is length of a list

    Xs=[i[0:numtrim] for i in Xs]
    steps=[i[0:(numtrim-1)] for i in steps]
    for i in range(numnodes):
        if i not in set(flatten_list(Xs)):
            alter_graph_size=1
    return Xs, steps, alter_graph_size

