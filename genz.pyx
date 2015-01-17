#cython: wraparound=False

import numpy as np
cimport numpy as np
import random
cimport cython

DTYPE = np.int
ctypedef np.int_t DTYPE_t

# using np.ndarray[DTYPE_t, ndim=2] slowed things down... why?
# @cython.boundscheck(False) causes crash

# restrict walk to only the next node in x OR previously visited nodes
def genZfromXG(list x, np.ndarray a):

    cdef unsigned int pos
    cdef unsigned int p, s
    cdef list walk
    cdef set x_left
    cdef np.ndarray j=np.zeros(len(a),dtype=np.int32)
    cdef np.ndarray possibles=np.zeros(len(a),dtype=np.int32)
    cdef np.ndarray newa=np.copy(a)

    possibles[[x[0],x[1],x[2]]] = 1
    j=np.where(possibles==0)[0]
    walk=[(x[0], x[1])]      # add first two Xs to random walk
    pos=2                    # only allow items up to pos
    newa[:,j]=0
    x_left=set(x[pos:])
    pl=np.nonzero(newa) # indices of pruned links (lists possible paths from each node)
    p=walk[-1][1]
    while len(x_left) > 0:
        s=random.choice(pl[1][pl[0]==p])
        walk.append((p,s))
        p=s
        if s in x_left:
            pos=pos+1
            x_left=set(x[pos:])
            if len(x[pos:]) > 0:
                possibles[x[pos]] = 1
                j=np.where(possibles==0)[0]
                newa=np.copy(a) 
                pl=np.nonzero(newa)
                newa[:,j]=0
    return walk

def threadZ(x,graph,numz):
    zarr=[]
    for i in range(numz):
        zarr.append(genZfromXG(x,graph))
    return zarr
