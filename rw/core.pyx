from __future__ import division

import multiprocessing as mp
import networkx as nx
import operator
import math
import scipy.stats
import sys
import copy
import csv

from numpy.linalg import inv
from itertools import *
from datetime import datetime

# sibling packages
from helper import *
from structs import *

print "CYTHON!"
import cython

cimport numpy as np
cfloat=np.float
ctypedef np.float_t cfloat_t

# TODO: make recording optional
# TODO: when doing same phase twice in a row, don't re-try same failures
    # (pass dict of failures, don't try if numchanges==0)

# mix U-INVITE with random jumping model
def addJumps(probs, td, numnodes=None, statdist=None, Xs=None):
    if (td.jumptype=="uniform") and (numnodes==None):
        raise ValueError("Must specify 'numnodes' when jumptype is uniform [addJumps]")
    if (td.jumptype=="stationary") and ((statdist==None) or (Xs==None)):
        raise ValueError("Must specify 'statdist' and 'Xs' when jumptype is stationary [addJumps]")

    if td.jumptype=="uniform":
        jumpprob=float(td.jump)/numnodes                      # uniform jumping
    
    for l in range(len(probs)):                              # loop through all lists (l)
        for inum, i in enumerate(probs[l][1:]):              # loop through all items (i) excluding first (don't jump to item 1)
            if td.jumptype=="stationary":
                jumpprob=statdist[Xs[l][inum]]                # stationary probability jumping
            probs[l][inum]=jumpprob + (1-td.jump)*i       # else normalize existing probability and add jumping probability
            if probs[l][inum] == 0.0:                    # if item can't be reached by RW or jumping...
                return -np.inf

    return probs

# objective graph cost
# returns the number of links that need to be added or removed to reach the true graph
def cost(graph,a):
    return sum(sum(np.array(abs(graph-a))))/2

# graph=estimated graph, a=target/comparison graph
# speed should be improved for large graphs if possible
def costSDT(graph, a):          
    Alinks=zip(*np.where(a==1))
    Glinks=zip(*np.where(graph==1))
    Anolinks=zip(*np.where(a==0))
    Gnolinks=zip(*np.where(graph==0))
    hit=sum([i in Alinks for i in Glinks])
    fa=len(Glinks)-hit
    cr=sum([i in Anolinks for i in Gnolinks])
    miss=len(Gnolinks)-cr
    cr=cr-len(a)            # don't count node self-transitions as correct rejections
    return [hit/2, miss/2, fa/2, cr/2]

# calculate P(SW_graph|graph type) using pdf generated from genPrior
def evalPrior(val, prior):
    # unpack dict for convenience
    kde=prior['kde']
    binsize=prior['binsize']

    prob=kde.integrate_box_1d(val-(binsize/2.0),val+(binsize/2.0))
    return prob

# returns a vector of how many hidden nodes to expect between each Xi for each X in Xs
def expectedHidden(Xs, a):
    numnodes=len(a)
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

# won't work with dotdict
def pivot2(graph, Xs, td, irts=Irts({}), prior=0, vmin=1, vmaj=0, best_ll=None, probmat=None, limit=np.inf, method=""):
    record=[method] 
    numchanges=0     # number of changes in single pivot() call

    if (best_ll == None) or (probmat == None):
        best_ll, probmat = probX(Xs,graph,td,irts=irts,prior=prior)   # LL of best graph found
    nxg=nx.to_networkx_graph(graph)

    # generate dict where v[i] is a list of nodes where (i, v[i]) is an existing edge in the graph
    if (method=="prune") or (method==0):
        print "Pruning", str(vmaj) + "." + str(vmin), "... ", # (len(edges)/2)-len(firstedges), "possible:",
        sys.stdout.flush()
        listofedges=np.where(graph==1)
        v=dict()
        for i in range(numnodes):
            v[i]=[]
        for i in zip(*listofedges):
            if ((i[0], i[1]) not in firstedges) and ((i[1], i[0]) not in firstedges): # don't flip first edges (FE)!
                v[i[0]].append(i[1])
    
    # generate dict where v[i] is a list of nodes where (i, v[i]) would form a new triangle
    if (method=="triangles") or (method==1):
        print "Adding triangles", str(vmaj) + "." + str(vmin), "... ", # (len(edges)/2), "possible:",
        sys.stdout.flush()
        nn=dict()
        for i in range(len(graph)):
            nn[i]=neighborsofneighbors(i, nxg)
        v=nn
    
    # generate dict where v[i] is a list of nodes where (i, v[i]) is NOT an existing an edge and does NOT form a triangle
    if (method=="nonneighbors") or (method==2):
        # list of a node's non-neighbors (non-edges) that don't form triangles
        print "Adding other edges", str(vmaj) + "." + str(vmin), "... ",
        sys.stdout.flush()
        nonneighbors=dict()
        for i in range(numnodes):
            nn=neighborsofneighbors(i, nxg)
            # non-neighbors that DON'T form triangles 
            nonneighbors[i]=[j for j in range(numnodes) if j not in nx.all_neighbors(nxg,i) and j not in nn] 
            nonneighbors[i].remove(i) # also remove self
        v=nonneighbors

    count=[0.0]*numnodes
    avg=[-np.inf]*numnodes
    finishednodes=0
    loopcount=0

    while (finishednodes < numnodes) and (loopcount < limit):
        loopcount += 1          # number of failures before giving up on this pahse
        maxval=max(avg)             
        bestnodes=[i for i, j in enumerate(avg) if j == maxval]  # most promising nodes based on avg logprob of edges with each node as vertex
        node1=np.random.choice(bestnodes)

        if len(v[node1]) > 0:
            n2avg=[avg[i] for i in v[node1]]
            maxval=max(n2avg)
            bestnodes=[v[node1][i] for i, j in enumerate(n2avg) if j == maxval]
            #node2=np.random.choice(v[node1])
            node2=np.random.choice(bestnodes)

            edge=(node1, node2)
            graph=swapEdges(graph,[edge])
            graph_ll, newprobmat=probX(Xs,graph,td,irts=irts,prior=prior,origmat=probmat,changed=[node1,node2])
            if best_ll > graph_ll:
                record.append(graph_ll)
                graph=swapEdges(graph,[edge])
            else:
                record.append(-graph_ll)
                best_ll = graph_ll
                probmat = newprobmat
                numchanges += 1
                loopcount = 0
            v[node1].remove(node2)   # remove edge from possible choices
            v[node2].remove(node1)
       
            # increment even if graph prob = -np.inf for implicit penalty
            count[node1] += 1
            count[node2] += 1
            if graph_ll != -np.inf:
                if avg[node1] == -np.inf:
                    avg[node1] = graph_ll
                else:
                    avg[node1] = avg[node1] * ((count[node1]-1)/count[node1]) + (1.0/count[node1]) * graph_ll
                if avg[node2] == -np.inf:
                    avg[node2] = graph_ll
                else:
                    avg[node2] = avg[node2] * ((count[node2]-1)/count[node2]) + (1.0/count[node2]) * graph_ll
        else:                       # no edges on this node left to try!
            avg[node1]=-np.inf      # so we don't try it again...
            finishednodes += 1

    print numchanges, "changes"

    records.append(record)
    return graph, best_ll, probmat, numchanges

#@profile
def findBestGraph(Xs, td, numnodes, irts=Irts({}), fitinfo=Fitinfo({}), prior=0, debug="T"):
    
    # return list of neighbors of neighbors of i, that aren't themselves neighbors of i
    # i.e., an edge between i and any item in nn forms a triangle
    #@profile
    def neighborsofneighbors(i, nxg):
        nn=[]                                   # neighbors of neighbors (nn)
        n=list(nx.all_neighbors(nxg,i))
        for j in n:
            nn=nn+list(nx.all_neighbors(nxg,j))
        nn=list(set(nn))
        for k in n:             # remove neighbors
            if k in nn:
                nn.remove(k)
        nn.remove(i)    # remove self
        return nn
        
    # toggle links back, should be faster than making graph copy
    #@profile
    def swapEdges(graph,links):
        for link in links:
            graph[link[0],link[1]] = 1 - graph[link[0],link[1]]
            graph[link[1],link[0]] = 1 - graph[link[1],link[0]] 
        return graph
        
    #@timer
    #@profile
    def pivot(graph, vmin=1, vmaj=0, best_ll=None, probmat=None, limit=np.inf, method=""):
        record=[method] 
        numchanges=0     # number of changes in single pivot() call

        if (best_ll == None) or (probmat == None):
            best_ll, probmat = probX(Xs,graph,td,irts=irts,prior=prior)   # LL of best graph found
        nxg=nx.to_networkx_graph(graph)

        # generate dict where v[i] is a list of nodes where (i, v[i]) is an existing edge in the graph
        if (method=="prune") or (method==0):
            print "Pruning", str(vmaj) + "." + str(vmin), "... ", # (len(edges)/2)-len(firstedges), "possible:",
            sys.stdout.flush()
            listofedges=np.where(graph==1)
            v=dict()
            for i in range(numnodes):
                v[i]=[]
            for i in zip(*listofedges):
                if ((i[0], i[1]) not in firstedges) and ((i[1], i[0]) not in firstedges): # don't flip first edges (FE)!
                    v[i[0]].append(i[1])
        
        # generate dict where v[i] is a list of nodes where (i, v[i]) would form a new triangle
        if (method=="triangles") or (method==1):
            print "Adding triangles", str(vmaj) + "." + str(vmin), "... ", # (len(edges)/2), "possible:",
            sys.stdout.flush()
            nn=dict()
            for i in range(len(graph)):
                nn[i]=neighborsofneighbors(i, nxg)
            v=nn
        
        # generate dict where v[i] is a list of nodes where (i, v[i]) is NOT an existing an edge and does NOT form a triangle
        if (method=="nonneighbors") or (method==2):
            # list of a node's non-neighbors (non-edges) that don't form triangles
            print "Adding other edges", str(vmaj) + "." + str(vmin), "... ",
            sys.stdout.flush()
            nonneighbors=dict()
            for i in range(numnodes):
                nn=neighborsofneighbors(i, nxg)
                # non-neighbors that DON'T form triangles 
                nonneighbors[i]=[j for j in range(numnodes) if j not in nx.all_neighbors(nxg,i) and j not in nn] 
                nonneighbors[i].remove(i) # also remove self
            v=nonneighbors

        count=[0.0]*numnodes
        avg=[-np.inf]*numnodes
        finishednodes=0
        loopcount=0

        while (finishednodes < numnodes) and (loopcount < limit):
            loopcount += 1          # number of failures before giving up on this pahse
            maxval=max(avg)             
            bestnodes=[i for i, j in enumerate(avg) if j == maxval]  # most promising nodes based on avg logprob of edges with each node as vertex
            node1=np.random.choice(bestnodes)

            if len(v[node1]) > 0:
                #node2=np.random.choice(v[node1]) # old
                
                n2avg=[avg[i] for i in v[node1]]
                maxval=max(n2avg)
                bestnodes=[v[node1][i] for i, j in enumerate(n2avg) if j == maxval]
                node2=np.random.choice(bestnodes)
                
                # print for debugging
                #for i,j in enumerate(avg):
                #    print i, ": ", j
                #print "---"
                #for i,j in enumerate(n2avg):
                #    print v[node1][i], ": ", j
                #print node1, node2
                #raw_input()

                edge=(node1, node2)
                graph=swapEdges(graph,[edge])
                graph_ll, newprobmat=probX(Xs,graph,td,irts=irts,prior=prior,origmat=probmat,changed=[node1,node2])
                if best_ll > graph_ll:
                    record.append(graph_ll)
                    graph=swapEdges(graph,[edge])
                else:
                    record.append(-graph_ll)
                    best_ll = graph_ll
                    probmat = newprobmat
                    numchanges += 1
                    loopcount = 0
                v[node1].remove(node2)   # remove edge from possible choices
                v[node2].remove(node1)
           
                # increment even if graph prob = -np.inf for implicit penalty
                count[node1] += 1
                count[node2] += 1
                if graph_ll != -np.inf:
                    if avg[node1] == -np.inf:
                        avg[node1] = graph_ll
                    else:
                        if fitinfo.followtype=="avg":
                            avg[node1] = avg[node1] * ((count[node1]-1)/count[node1]) + (1.0/count[node1]) * graph_ll
                        elif fitinfo.followtype=="max":
                            avg[node1] = max(avg[node1], graph_ll)
                    if avg[node2] == -np.inf:
                        avg[node2] = graph_ll
                    else:
                        if fitinfo.followtype=="avg":
                            avg[node2] = avg[node2] * ((count[node2]-1)/count[node2]) + (1.0/count[node2]) * graph_ll
                        elif fitinfo.followtype=="max":
                            avg[node2] = max(avg[node2], graph_ll)
            else:                       # no edges on this node left to try!
                avg[node1]=-np.inf      # so we don't try it again...
                finishednodes += 1

        print numchanges, "changes"

        records.append(record)
        return graph, best_ll, probmat, numchanges

    def phases_pool(graph, best_ll, probmat):
        complete=[0,0,0]
        vmaj=0
        vmin=1
        p=mp.Pool(5)
        #while sum(complete) < 3:
        phasenum=complete.index(0)
        if phasenum==0: limit=fitinfo.prune_limit
        if phasenum==1: limit=fitinfo.triangle_limit
        if phasenum==2: limit=fitinfo.other_limit
        if (phasenum==0) and (vmin==1): vmaj += 1
        result_list=[]
        print graph
        #for i in range(5):
        qq=p.apply_async(pivot2, (graph, Xs, td, ), {'irts': irts, 'prior': prior, 'best_ll': best_ll, 'vmaj': vmaj, 'vmin': vmin, 'method': phasenum, 'limit': limit}, callback=result_list.append)
        qq.get()
        p.close()
        p.join()
        print result_list

        return graph
    
    def phases(graph, best_ll, probmat):
        complete=[0,0,0]
        vmaj=0
        vmin=1
        while sum(complete) < 3:
            phasenum=complete.index(0)
            if phasenum==0: limit=fitinfo.prune_limit
            if phasenum==1: limit=fitinfo.triangle_limit
            if phasenum==2: limit=fitinfo.other_limit
            if (phasenum==0) and (vmin==1): vmaj += 1

            graph, best_ll, probmat, numchanges = pivot(graph, best_ll=best_ll, vmaj=vmaj, vmin=vmin, method=phasenum, limit=limit, probmat=probmat)
            if numchanges > 0:
                vmin += 1
            else:
                if (vmin==1): complete[phasenum]=1
                if (phasenum==0) and (vmin>1): complete=[1,0,0]
                if (phasenum==1) and (vmin>1): complete=[0,1,0]
                if (phasenum==2) and (vmin>1): complete=[0,0,1]
                vmin=1

        return graph

    firstedges=[(x[0], x[1]) for x in Xs]
    
    # find a good starting graph using naive RW
    if fitinfo.startGraph=="windowgraph":
        graph=windowGraph(Xs,numnodes)
    elif fitinfo.startGraph=="naiverw":
        graph=noHidden(Xs,numnodes)
  
    best_ll, probmat = probX(Xs,graph,td,irts=irts,prior=prior)   # LL of best graph found
    records=[]
    graph=phases(graph, best_ll, probmat) # JZ change back to phases
    f=open('records.csv','w')
    wr=csv.writer(f)
    for record in records:
        wr.writerow(record)

    return graph, best_ll

def firstEdge(Xs, numnodes):
    a=np.zeros((numnodes,numnodes))
    for x in Xs:
        a[x[0],x[1]]=1
        a[x[1],x[0]]=1 # symmetry
    a=np.array(a.astype(int))
    return a

# first hitting times for each node
def firstHits(walk):
    firsthit=[]
    path=path_from_walk(walk)
    for i in observed_walk(walk):
        firsthit.append(path.index(i))
    return zip(observed_walk(walk),firsthit)

# generate a connected Watts-Strogatz small-world graph
# (n,k,p) = (number of nodes, each node connected to k-nearest neighbors, probability of rewiring)
# k has to be even, tries is number of attempts to make connected graph
def genG(tg, seed=None):
    if tg.graphtype=="wattsstrogatz":
        g=nx.connected_watts_strogatz_graph(tg.numnodes,tg.numlinks,tg.prob_rewire,1000,seed) # networkx graph
    elif tg.graphtype=="random":                               
        g=nx.erdos_renyi_graph(tg.numnodes, tg.prob_rewire,seed)
    elif tg.graphtype=="steyvers":
        g=genSteyvers(tg.numnodes, tg.numlinks)
    
    a=np.array(nx.to_numpy_matrix(g)).astype(int)
    return g, a

# only returns adjacency matrix, not nx graph
def genGfromZ(walk, numnodes):
    a=np.zeros((numnodes,numnodes))
    for i in set(walk):
        a[i[0],i[1]]=1
        a[i[1],i[0]]=1 # symmetry
    a=np.array(a.astype(int))
    return a

# Generate `numgraphs` graphs from data `Xs`, requiring `numnodes` (used in case not all nodes are covered in data)
# Graphs are generated by sequentially adding filler nodes between two adjacent items with p=`theta`
# When theta=0, returns a naive RW graph
def genGraphs(numgraphs, theta, Xs, numnodes):
    Zs=[reduce(operator.add,[genZfromX(x,theta) for x in Xs]) for i in range(numgraphs)]
    As=[genGfromZ(z, numnodes) for z in Zs]
    return As

# generate pdf of small-world metric based on W-S criteria
# n <- # samples (larger n == better fidelity)
# tries <- W-S parameter (number of tries to generate connected graph)
def genPrior(tg, n=10000, bins=100):
    print "generating prior distribution..."
    sw=[]
    for i in range(n):
        if tg.graphtype=="wattsstrogatz":
            g=nx.connected_watts_strogatz_graph(tg.numnodes,tg.numlinks,tg.prob_rewire,tries=1000)
            g=nx.to_numpy_matrix(g)
        if tg.graphtype=="steyvers":
            g=genSteyvers(tg.numnodes, tg.numlinks)

        sw.append(smallworld(g))
    kde=scipy.stats.gaussian_kde(sw)
    binsize=(max(sw)-min(sw))/bins
    print "...done"
    return {'kde': kde, 'binsize': binsize}

def genSteyvers(n,m, tail=1):                          # tail allows m-1 "null" nodes in neighborhood of every node
    a=np.zeros((n,n))                                  # initialize matrix
    for i in range(m):                                 # complete m x m graph
        for j in range(m):
            if i!= j:
                a[i,j]=1
    for i in range(m,n):                               # for the rest of nodes, preferentially attach
        nodeprob=sum(a)/sum(sum(a))                    # choose node to differentiate with this probability distribution
        diffnode=np.random.choice(n,p=nodeprob)        # node to differentiate
        h=list(np.where(a[diffnode])[0]) + [diffnode]  # neighborhood of diffnode
        if tail==1:
            h=h + [-1]*(m-1)
        #hprob=sum(a[:,h])/sum(sum(a[:,h]))                 # attach proportional to node degree?
        #tolink=np.random.choice(h,m,replace=False,p=hprob)
        tolink=np.random.choice(h,m,replace=False)          # or attach randomly
        for j in tolink:
            if j != -1:
                a[i,j]=1
                a[j,i]=1
    return nx.to_networkx_graph(a)

# return simulated data on graph g
# also return number of steps between first hits (to use for IRTs)
def genX(g,td,seed=None):
    rwalk=random_walk(g,td,seed)
    Xs=observed_walk(rwalk)
    
    fh=list(zip(*firstHits(rwalk))[1])
    steps=[fh[i]-fh[i-1] for i in range(1,len(fh))]

    return Xs, steps

# generate random walk that results in observed x
def genZfromX(x, theta):
    x2=x[:]                  # make a local copy
    x2.reverse()
    
    path=[]                  # z to return
    path.append(x2.pop())    # add first two x's to z
    path.append(x2.pop())

    while len(x2) > 0:
        if np.random.random() < theta:     # might want to set random seed for replicability?
            # add random hidden node
            possibles=set(path) # choose equally from previously visited nodes
            possibles.discard(path[-1]) # but exclude last node (node cant link to itself)
            path.append(np.random.choice(list(possibles)))
        else:
            # first hit!
            path.append(x2.pop())
    return walk_from_path(path)

# wrapper returns one graph with theta=0
# aka draw edge between all observed nodes in all lists
def noHidden(Xs, numnodes):
    return genGraphs(1, 0, Xs, numnodes)[0]

# Unique nodes in random walk preserving order
# (aka fake participant data)
# http://www.peterbe.com/plog/uniqifiers-benchmark
def observed_walk(walk):
    seen = {}
    result = []
    for item in path_from_walk(walk):
        if item in seen: continue
        seen[item] = 1
        result.append(item)
    return result

# flat list from tuple walk
def path_from_walk(walk):
    path=list(zip(*walk)[0]) # first element from each tuple
    path.append(walk[-1][1]) # second element from last tuple
    return path

# probability of observing Xs, including irts and prior
#@profile
@nogc
@cython.boundscheck(False) # compiler directive
@cython.wraparound(False) # compiler directive
def probX(list Xs, np.ndarray[long, ndim=2] a, td, irts=Irts({}), prior=0, origmat=None, changed=[]):
    cdef list probs, prob, notinx, deletedlist, x
    cdef int xnum, curpos, numnodes, lenchanged, startindex, absorbingindex, update, i
    cdef np.ndarray[cfloat_t, ndim=2] t
    
    cdef np.ndarray[cfloat_t, ndim=2] Q
    cdef np.ndarray[cfloat_t, ndim=2] I
    cdef np.ndarray[cfloat_t, ndim=2] N
    cdef np.ndarray[cfloat_t, ndim=2] R
    cdef np.ndarray[cfloat_t, ndim=2] B
    cdef np.ndarray[long, ndim=1] notdeleted

    numnodes=len(a)

    #np.random.seed(randomseed)             # bug in nx, random seed needs to be reset    
    probs=[] 

    t=a/sum(a)            # will throw warning if a node is inaccessible

    if (td.jumptype=="stationary") or (td.startX=="stationary"):
        statdist=stationary(t)

    for xnum, x in enumerate(Xs):
        prob=[]
        if td.startX=="stationary":
            prob.append(statdist[x[0]])      # probability of X_1
        elif td.startX=="uniform":
            prob.append(1.0/numnodes)

        # if impossible starting point, return immediately
        if prob[0]==0.0:
            return -np.inf, -np.inf
        
        # optimize? pre-compute and pass to function
        notinx=[i for i in range(numnodes) if i not in x]        # nodes not in trimmed X

        lenchanged=len(changed)
        if (lenchanged > 0) and isinstance(origmat,list):    # if updating prob. matrix based on specific link changes
            update=0            # reset for each list

        for curpos in range(1,len(x)):
            if (lenchanged > 0) and isinstance(origmat,list):
                if update==0:   # first check if probability needs to be updated (only AFTER first changed node has been reached
                    if (Xs[xnum][curpos-1] in changed):
                        update=1
                if update==0:   # if not, take probability from old matrix
                    prob.append(origmat[xnum][curpos])
                    continue
            startindex=x[curpos-1]
            deletedlist=sorted(x[curpos:]+notinx,reverse=True)  # optimize? sort each list once and remove items
            notdeleted=np.array([i for i in range(numnodes) if i not in deletedlist])
           
            Q=t[notdeleted[:, None],notdeleted]

            if (len(irts.data) > 0) and (irts.irt_weight < 1): # use this method only when passing IRTs with weight < 1
                startindex = startindex-sum([startindex > i for i in deletedlist])
                # same as startindex=sorted(x[:curpos]).index(x[curpos-1])... less readable, maybe more efficient?
                
                numcols=len(Q)
                flist=[]
                newQ=np.zeros(numcols)  # init to Q^0, for when r=1 (using only one: row for efficiency)
                newQ[startindex]=1.0

                irt=irts.data[xnum][curpos-1]

                # precompute for small speedup
                if irts.irttype=="gamma":
                    logbeta=math.log(irts.beta)
                    logirt=math.log(irt)

                for r in range(1,irts.rcutoff):
                    innersum=0
                    for k in range(numcols):
                        num1=newQ[k]                        # probability of being at node k in r-1 steps
                        num2=t[x[curpos],notdeleted[k]]     # probability transitioning from k to absorbing node    
                        innersum=innersum+(num1*num2)

                    # much faster than using scipy.stats.gamma.pdf
                    if irts.irttype=="gamma":
                        log_dist=r*logbeta-math.lgamma(r)+(r-1)*logirt-irts.beta*irt # r=alpha. probability of observing irt at r steps
                    if irts.irttype=="exgauss":
                        log_dist=math.log(irts.lambd/2.0)+(irts.lambd/2.0)*(2.0*r+irts.lambd*(irts.sigma**2)-2*irt)+math.log(math.erfc((r+irts.lambd*(irts.sigma**2)-irt)/(math.sqrt(2)*irts.sigma)))

                    if innersum > 0: # sometimes it's not possible to get to the target node in r steps
                        flist.append(log_dist*(1-irts.irt_weight)+irts.irt_weight*math.log(innersum))
                    newQ=np.inner(newQ,Q)     # raise power by one

                f=sum([math.e**i for i in flist])
                prob.append(f)           # probability of x_(t-1) to X_t
            else:                        # if no IRTs, use standard INVITE
                I=np.identity(len(Q))    # optimize? how?/c.inde
                reg=(1+1e-10)            # nuisance parameter to prevent errors; can also use pinv, but that's much slower
                N=inv(I*reg-Q)
               
                r=sorted(x[curpos:])
                c=sorted(x[:curpos])
                R=t[np.array(r)[:,None],c]

                B = np.dot(R,N)
                startindex = c.index(x[curpos-1])
                absorbingindex = r.index(x[curpos])
                prob.append(B[absorbingindex,startindex])

            # if there's an impossible transition and no jumping, return immediately
            if (prob[curpos]==0.0) and (td.jump == 0.0):
                return -np.inf, -np.inf

        probs.append(prob)

    # adjust for jumping probability
    if td.jump > 0.0:
        if td.jumptype=="uniform":
            probs=addJumps(probs, td, numnodes=numnodes)
        elif td.jumptype=="stationary":
            probs=addJumps(probs, td, statdist=statdist, Xs=Xs)
        if probs==-np.inf:
            return -np.inf, -np.inf

    # total ll of graph
    ll=sum([sum([math.log(j) for j in probs[i]]) for i in range(len(probs))])

    # inclue prior?
    if prior:
        sw=smallworld(a)
        priorprob = evalPrior(sw,prior)
        if priorprob == 0.0:
            return -np.inf, -np.inf
        else:
            ll=ll + math.log(priorprob)

    return ll, probs

# given an adjacency matrix, take a random walk that hits every node; returns a list of tuples
def random_walk(g,td,seed=None):

    if (td.startX=="stationary") or (td.jumptype=="stationary"):
        a=np.array(nx.to_numpy_matrix(g))
        t=a/sum(a).astype(float)
        statdist=stationary(t)
        statdist=scipy.stats.rv_discrete(values=(range(len(t)),statdist))
    
    if td.startX=="stationary":
        start=statdist.rvs(random_state=seed)      # choose starting point from stationary distribution
    elif td.startX=="uniform":
        start=np.random.choice(nx.nodes(g))      # choose starting point uniformly

    walk=[]
    unused_nodes=set(nx.nodes(g))
    unused_nodes.remove(start)
    first=start
    while len(unused_nodes) > 0:
        if np.random.random() > td.jump:
            second=np.random.choice([x for x in nx.all_neighbors(g,first)]) # follow random edge
        else:
            if td.jumptype=="stationary":
                second=statdist.rvs(random_state=seed)       # jump based on statdist
            elif td.jumptype=="uniform":
                second=np.random.choice(nx.nodes(g))          # jump uniformly
        walk.append((first,second))
        if second in unused_nodes:
            unused_nodes.remove(second)
        first=second
    return walk

# return small world statistic of a graph
# returns metric of largest component if disconnected
def smallworld(a):
    if isinstance(a,np.array):
        g_sm=nx.from_numpy_matrix(a)    # if matrix is passed, convert to networkx
    else:
        g_sm = a                        # else assume networkx graph was passed
    g_sm=max(nx.connected_component_subgraphs(g_sm),key=len)   # largest component
    numnodes=g_sm.number_of_nodes()
    numedges=g_sm.number_of_edges()
    nodedegree=(numedges*2.0)/numnodes
    
    c_sm=nx.average_clustering(g_sm)        # c^ws in H&G (2006)
    #c_sm=sum(nx.triangles(usfg).values())/(# of paths of length 2) # c^tri
    l_sm=nx.average_shortest_path_length(g_sm)
    
    # c_rand same as edge density for a random graph? not sure if "-1" belongs in denominator, double check
    #c_rand= (numedges*2.0)/(numnodes*(numnodes-1))   # c^ws_rand?  
    c_rand= float(nodedegree)/numnodes                  # c^tri_rand?
    l_rand= math.log(numnodes)/math.log(nodedegree)    # approximation, see humphries & gurney (2008) eq 11
    #l_rand= (math.log(numnodes)-0.5772)/(math.log(nodedegree)) + .5 # alternative ASPL from fronczak, fronczak & holyst (2004)
    s=(c_sm/c_rand)/(l_sm/l_rand)
    return s

def stationary(t,method="unweighted"):
    if method=="unweighted":                 # only works for unweighted matrices!
        return sum(t>0)/float(sum(sum(t>0)))   
    elif method=="power":                       # slow?
        return np.linalg.matrix_power(t,500)[:,0]
    else:                                       # buggy
        eigen=np.linalg.eig(t)[1][:,0]
        return np.real(eigen/sum(eigen))

# generates fake IRTs from # of steps in a random walk, using gamma distribution
def stepsToIRT(irts, seed=None):
    np.random.RandomState(seed)        # to generate the same IRTs each time
    new_irts=[]
    for irtlist in irts.data:
        if irts.irttype=="gamma":
            newlist=[np.random.gamma(irt, (1.0/irts.beta)) for irt in irtlist]  # beta is rate, but random.gamma uses scale (1/rate)
        if irts.irttype=="exgauss":
            newlist=[rand_exg(irt, irts.sigma, irts.lambd) for irt in irtlist] 
        new_irts.append(newlist)
    return new_irts

# runs a batch of toy graphs. logging code needs to be cleaned up significantly
def toyBatch(tg, td, outfile, irts=Irts({}), fitinfo=Fitinfo({}), start_seed=0,
             methods=['rw','fe','uinvite','uinvite_irt','uinvite_prior','uinvite_irt_prior'],header=1,debug="F"):
    np.random.seed(start_seed)

    # break out of function if using unknown method
    for method in methods:
        if method not in ['rw','fe','uinvite','uinvite_irt','uinvite_prior','uinvite_irt_prior']:
            raise ValueError('ERROR: Trying to fit graph with unknown method: ', method)

    # flag if using a prior method
    if ('uinvite_prior' in methods) or ('uinvite_irt_prior' in methods): use_prior=1
    else: use_prior=0

    # flag if using an IRT method
    if ('uinvite_irt' in methods) or ('uinvite_irt_prior' in methods): use_irt=1
    else: use_irt=0

    if use_prior:
        prior=genPrior(tg)

    # stuff to write to file
    # more vals to write to file
    globalvals=['numedges','graph_seed','x_seed','truegraph','ll_tg', 'll_tg_prior','ll_tg_irt',
                'll_tg_irt_prior']  # same across all methods
    methodvals=['cost','time','bestgraph','hit','miss','fa','cr','ll']     # differ per method

    f=open(outfile,'a', 0)                # write/append to file with no buffering
    if header==1:
        irtvals=irts.keys()
        irtvals.remove('data')          # don't write IRT data
        f.write(','.join(tg.keys())+',')
        f.write(','.join(td.keys())+',')
        f.write(','.join(irtvals)+',')
        f.write(','.join(globalvals)+',')
        for methodnum, method in enumerate(methods):
            towrite=[i+'_'+method for i in methodvals]
            f.write(','.join(towrite))
            if methodnum != (len(methods)-1):   # if not the last method, add a comma
                f.write(',')
        f.write('\n')
 
    # store all data in dict to write to file later
    data={}
    for method in methods:
        data[method]={}
        for val in methodvals:
            data[method][val]=[]

    # how many graphs to run?
    seed_param=start_seed
    last_seed=start_seed+tg.numgraphs

    while seed_param < last_seed:

        # generate toy graph and data
        # give up if it's hard to generate data that cover full graph

        # generate toy data
        graph_seed=seed_param
        g,a=genG(tg,seed=graph_seed)

        tries=0
        while True:
            x_seed=seed_param

            [Xs,irts.data]=zip(*[genX(g, td, seed=x_seed+i) for i in range(td.numx)])
            Xs=list(Xs)
            irts.data=list(irts.data)
            [Xs,irts.data,alter_graph]=trimX(td.trim,Xs,irts.data)      # trim data when necessary

            # generate IRTs if using IRT model
            if use_irt: irts.data=stepsToIRT(irts, seed=x_seed)

            if alter_graph==0:                      # only use data that covers the entire graph
                break
            else:
                tries=tries+1
                seed_param = seed_param + 1
                last_seed = last_seed + 1    # if data is unusable (doesn't cover whole graph), add another seed
                if tries >= 1000:
                    raise ValueError("Data doesn't cover full graph... Increase 'trim' or 'numx' (or change graph)")

        numedges=nx.number_of_edges(g)
        truegraph=nx.generate_sparse6(g)  # to write to file
        # true graph LL
        ll_tg=probX(Xs, a, td)[0]
        if use_prior: ll_tg_prior=probX(Xs, a, td, prior=prior)[0]
        else: ll_tg_prior=""
        if use_irt: ll_tg_irt=probX(Xs, a, td, irts=irts)[0]
        else: ll_tg_irt=""
        if use_prior and use_irt: ll_tg_irt_prior=probX(Xs, a, td, prior=prior, irts=irts)[0]
        else: ll_tg_irt_prior=""

        for method in methods:
            if debug=="T": print "SEED: ", seed_param, "method: ", method
            
            # Find best graph! (and log time)
            starttime=datetime.now()
            if method == 'uinvite': 
                bestgraph, ll=findBestGraph(Xs, td, tg.numnodes, debug=debug, fitinfo=fitinfo)
            if method == 'uinvite_prior':
                bestgraph, ll=findBestGraph(Xs, td, tg.numnodes, prior=prior, debug=debug, fitinfo=fitinfo)
            if method == 'uinvite_irt':
                bestgraph, ll=findBestGraph(Xs, td, tg.numnodes, irts=irts, debug=debug, fitinfo=fitinfo)
            if method == 'uinvite_irt_prior':
                bestgraph, ll=findBestGraph(Xs, td, tg.numnodes, irts=irts, prior=prior, debug=debug, fitinfo=fitinfo)
            if method == 'rw':
                bestgraph=noHidden(Xs,tg.numnodes)
                ll=probX(Xs, bestgraph, td)[0]
            if method == 'fe':
                bestgraph=firstEdge(Xs,tg.numnodes)
                ll=probX(Xs, bestgraph, td)[0]
            elapsedtime=str(datetime.now()-starttime)
            if debug=="T": 
                print elapsedtime
                print "COST: ", cost(bestgraph,a)
                print nx.generate_sparse6(nx.to_networkx_graph(bestgraph))
    
            # compute SDT
            hit, miss, fa, cr = costSDT(bestgraph,a)

            # Record cost, time elapsed, LL of best graph, hash of best graph, and SDT measures
            data[method]['cost'].append(cost(bestgraph,a))
            data[method]['time'].append(elapsedtime)
            data[method]['ll'].append(ll)
            data[method]['bestgraph'].append(nx.generate_sparse6(nx.to_networkx_graph(bestgraph)))
            data[method]['hit'].append(hit)
            data[method]['miss'].append(miss)
            data[method]['fa'].append(fa)
            data[method]['cr'].append(cr)

        # log stuff here
        towrite=[str(tg[i]) for i in tg.keys()]
        towrite=towrite+[str(td[i]) for i in td.keys()]
        towrite=towrite+[str(irts[i]) for i in irts.keys() if i != 'data']  # don't write IRT data
        towrite=towrite+[str(eval(i)) for i in globalvals]
        f.write(','.join(towrite))
        for method in methods:
            for val in methodvals:
                f.write(','+str(data[method][val][-1]))
        f.write('\n')

        seed_param = seed_param + 1
    f.close()

# trim Xs to proportion of graph size, the trim graph to remove any nodes that weren't hit
# used to simulate human data that doesn't cover the whole graph every time
def trimX(trimprop, Xs, steps):
    numnodes=len(Xs[0])             # since Xs haven't been trimmed, we know list covers full graph
    alter_graph_size=0              # report if graph size changes-- may result in disconnected graph!
    numtrim=int(round(numnodes*trimprop))
    Xs=[i[0:numtrim] for i in Xs]
    steps=[i[0:(numtrim-1)] for i in steps]
    for i in range(numnodes):
        if i not in set(flatten_list(Xs)):
            alter_graph_size=1
    return Xs, steps, alter_graph_size

# tuple walk from flat list
def walk_from_path(path):
    walk=[]
    for i in range(len(path)-1):
        walk.append((path[i],path[i+1])) 
    return walk

# incomplete
def windowGraph(Xs, numnodes, windowsize=2):
    if windowsize < 1:
        print "Error in windowGraph(): windowsize must be >= 1"
        return
    graph = genGraphs(1, 0, Xs, numnodes)[0]    # start with naive RW
    if windowsize==1:
        return graph                            # same as naive RW if windowsize=1
    for x in Xs:                                # for each list
        for pos in range(len(x)):               # for each item in list
            for i in range(2,windowsize+1):     # for each window size
                if pos+i < len(x):
                    graph[x[pos],x[pos+i]]=1
                    graph[x[pos+i],x[pos]]=1
                if pos-i >= 0:
                    graph[x[pos],x[pos-i]]=1
                    graph[x[pos-i],x[pos]]=1
    return graph
