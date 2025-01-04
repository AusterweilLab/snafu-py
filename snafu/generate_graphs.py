from . import *

# See Steyvers & Tenenbaum (2005)
# Include an extra parameter 'tail' which allows m-1 "null" nodes in
# neighborhood of every node to better match scale-free distribution.
def generate_tenenbaum_steyvers_network(n, m, tail=True, seed=None):
    nplocal = np.random.RandomState(seed)
    a = np.zeros((n,n))                                  # initialize matrix
    for i in range(m):                                   # complete m x m graph
        for j in range(m):
            if i != j:
                a[i,j] = 1
    for i in range(m,n):                                 # for the rest of nodes, preferentially attach
        nodeprob = sum(a) / sum(sum(a))                  # choose node to differentiate with this probability distribution
        diffnode = nplocal.choice(n,p=nodeprob)          # node to differentiate
        h = list(np.where(a[diffnode])[0]) + [diffnode]  # neighborhood of diffnode
        if tail==True:
            h = h + [-1] * (m-1)
        #hprob=sum(a[:,h])/sum(sum(a[:,h]))              # attach proportional to node degree?
        #tolink=nplocal.choice(h,m,replace=False,p=hprob)
        tolink = nplocal.choice(h,m,replace=False)       # or attach randomly
        for j in tolink:
            if j != -1:
                a[i,j] = 1
                a[j,i] = 1
    return nx.to_networkx_graph(a)
