from __future__ import division

import pickle
import networkx as nx
import numpy as np
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

# TODO: when doing same phase twice in a row, don't re-try same failures
    # (pass dict of failures, don't try if numchanges==0)
# TODO: in hierarchical model, only redo subjects prior to last change


# mix U-INVITE with random jumping model
def addJumps(probs, td, numnodes=None, statdist=None, Xs=None):
    if (td.jumptype=="uniform") and (numnodes==None):
        raise ValueError("Must specify 'numnodes' when jumptype is uniform [addJumps]")
    if (td.jumptype=="stationary") and ((statdist==None) or (Xs==None)):
        raise ValueError("Must specify 'statdist' and 'Xs' when jumptype is stationary [addJumps]")

    if td.jumptype=="uniform":
        jumpprob=float(td.jump)/numnodes                     # uniform jumping
    
    for l in range(len(probs)):                              # loop through all lists (l)
        for inum, i in enumerate(probs[l][1:]):              # loop through all items (i) excluding first (don't jump to item 1)
            if td.jumptype=="stationary":
                jumpprob=statdist[Xs[l][inum+1]]             # stationary probability jumping
                if np.isnan(jumpprob):                       # if node is disconnected, jumpprob is nan
                    jumpprob=0.0
            probs[l][inum+1]=jumpprob + (1-td.jump)*i        # else normalize existing probability and add jumping probability
    return probs

# mix U-INVITE with priming model
# code is confusing...
def adjustPriming(probs, td, Xs):
    for xnum, x in enumerate(Xs[1:]):         # check all items starting with 2nd list
        for inum, i in enumerate(x[:-1]):     # except last item
            if i in Xs[xnum][:-1]:            # is item in previous list? if so, prime next item
                # follow prime with P td.priming, follow RW with P (1-td.priming)
                idx=Xs[xnum].index(i) # index of item in previous list
                if Xs[xnum][idx+1]==Xs[xnum+1][inum+1]:
                    probs[xnum+1][inum+1] = (probs[xnum+1][inum+1] * (1-td.priming)) + td.priming
                else:
                    probs[xnum+1][inum+1] = (probs[xnum+1][inum+1] * (1-td.priming))
    return probs

def blockModel(Xs, td, numnodes, fitinfo=Fitinfo({}), prior=None, debug=True, seed=None):
    nplocal=np.random.RandomState(seed)
    import itertools

    def swapEdges(graph,links):
        for link in links:
            graph[link[0],link[1]] = 1 - graph[link[0],link[1]]
            graph[link[1],link[0]] = 1 - graph[link[1],link[0]]
        return graph

    # starting categories
    categories = flatten_list([walk_from_path(x) for x in Xs])
    categories = list({frozenset(i) for i in categories})
    categories = [set(i) for i in categories]
    nplocal.shuffle(categories)
 
    # since all categories are tuples of adjacent pairs, starting graph is equivalent to naive random walk 
    a = noHidden(Xs, numnodes)
    best_ll, probs = probX(Xs, a, td)

    for cat in categories:
        for catpair in categories:
            if cat != catpair:
                newset=set.union(cat,catpair)
                edgeschanged = []
                for i in itertools.combinations(newset, 2):
                    if (a[i[0],i[1]] == 0) and (i[0] != i[1]):
                        a = swapEdges(a,[i])
                        edgeschanged.append(i)
                nodeschanged = list({j for i in edgeschanged for j in i})
                if len(nodeschanged) > 0:
                    graph_ll, probs = probX(Xs, a, td, changed=nodeschanged)
                    if graph_ll > best_ll:                                      # if merged categories are better, keep them merged
                        print "GOOD"
                        best_ll = graph_ll
                        #categories.remove(cat)
                        #categories.remove(catpair)
                        #categories.append(newset)
                    else:                                                       # else keep categories separate
                        a = swapEdges(a,edgeschanged)
                else:                          
                    pass
                    # categories are already inter-connected, merge them
                    #categories.remove(cat)
                    #categories.append(newset)
                    
    return a

# objective graph cost
# returns the number of links that need to be added or removed to reach the true graph
def cost(graph,a):
    return sum(sum(np.array(abs(graph-a))))/2

# graph=estimated graph, a=target/comparison graph
def costSDT(graph, a):
    hit=0; miss=0; fa=0; cr=0
    check=(graph==a)
    for rnum, r in enumerate(a):
        for cnum, c in enumerate(r[:rnum]):
            if check[rnum,cnum]==True:
                if a[rnum,cnum]==1:
                    hit += 1
                else:
                    cr += 1
            else:
                if a[rnum,cnum]==1:
                    miss += 1
                else:
                    fa += 1
    return [hit, miss, fa, cr]

# calculate P(SW_graph|graph type) using pdf generated from genSWPrior
def evalSWprior(val, prior):
    # unpack dict for convenience
    kde=prior['kde']
    binsize=prior['binsize']

    prob=kde.integrate_box_1d(val-(binsize/2.0),val+(binsize/2.0))
    return prob

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

def fullyConnected(numnodes):
    a=np.ones((numnodes,numnodes))
    for i in range(numnodes):
        a[i,i]=0.0
    return a.astype(int)

# generate a connected Watts-Strogatz small-world graph
# (n,k,p) = (number of nodes, each node connected to k-nearest neighbors, probability of rewiring)
# k has to be even, tries is number of attempts to make connected graph
def genG(tg, seed=None):
    if tg.graphtype=="wattsstrogatz":
        g=nx.connected_watts_strogatz_graph(tg.numnodes,tg.numlinks,tg.prob_rewire,1000,seed) # networkx graph
    elif tg.graphtype=="random":                               
        g=nx.erdos_renyi_graph(tg.numnodes, tg.prob_rewire,seed)
    elif tg.graphtype=="steyvers":
        g=genSteyvers(tg.numnodes, tg.numlinks, seed=seed)
    
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

# generate starting graph for U-INVITE
def genStartGraph(Xs, numnodes, td, fitinfo):
    if fitinfo.startGraph=="goni_valid":
        graph=goni(Xs, numnodes, td=td, valid=True, fitinfo=fitinfo)
    elif fitinfo.startGraph=="rw":
        graph=noHidden(Xs,numnodes)
    elif fitinfo.startGraph=="fully_connected":
        graph=fullyConnected(numnodes)
    elif fitinfo.startGraph=="empty_graph":
        graph=np.zeros((numnodes,numnodes)).astype(int)           # useless...
    else:
        graph=np.copy(fitinfo.startGraph)                         # assume a graph has been passed as a starting point
    return graph

# generate pdf of small-world metric based on W-S criteria
# n <- # samples (larger n == better fidelity)
# tries <- W-S parameter (number of tries to generate connected graph)
# forcenew <- if 1, don't use cached prior
def genSWPrior(tg, n, bins=100, forcenew=False):

    # filename for prior
    if tg.graphtype=="steyvers":
        filename = "steyvers_" + str(tg.numnodes) + "_" + str(tg.numlinks) + ".prior"
    if tg.graphtype=="wattsstrogatz":
        filename = "wattsstrogatz_" + str(tg.numnodes) + "_" + str(tg.numlinks) + "_" + str(tg.prob_rewire) + ".prior"
    
    def newPrior():
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
        prior={'kde': kde, 'binsize': binsize}
        with open('./priors/' + filename,'wb') as fh:
            pickle.dump(prior,fh)
        return prior

    if not forcenew:                                                 # use cached prior when available
        try:                                                        # check if cached prior exist
            with open('./priors/' + filename,'r') as fh:
                prior=pickle.load(fh)
            print "Retrieving cached prior..."
        except:
            prior=newPrior()
    else:                                                            # don't use cached prior
        prior=newPrior()
    return prior

def genSteyvers(n,m, tail=1, seed=None):               # tail allows m-1 "null" nodes in neighborhood of every node
    nplocal=np.random.RandomState(seed)
    a=np.zeros((n,n))                                  # initialize matrix
    for i in range(m):                                 # complete m x m graph
        for j in range(m):
            if i!= j:
                a[i,j]=1
    for i in range(m,n):                               # for the rest of nodes, preferentially attach
        nodeprob=sum(a)/sum(sum(a))                    # choose node to differentiate with this probability distribution
        diffnode=nplocal.choice(n,p=nodeprob)        # node to differentiate
        h=list(np.where(a[diffnode])[0]) + [diffnode]  # neighborhood of diffnode
        if tail==1:
            h=h + [-1]*(m-1)
        #hprob=sum(a[:,h])/sum(sum(a[:,h]))                 # attach proportional to node degree?
        #tolink=nplocal.choice(h,m,replace=False,p=hprob)
        tolink=nplocal.choice(h,m,replace=False)          # or attach randomly
        for j in tolink:
            if j != -1:
                a[i,j]=1
                a[j,i]=1
    return nx.to_networkx_graph(a)

# return simulated data on graph g
# also return number of steps between first hits (to use for IRTs)
def genX(g, td, seed=None):
    Xs=[]
    steps=[]
    priming_vector=[]

    for xnum in range(td.numx):
        rwalk=random_walk(g, td, priming_vector=priming_vector, seed=seed+xnum)
        x=observed_walk(rwalk)
        fh=list(zip(*firstHits(rwalk))[1])
        step=[fh[i]-fh[i-1] for i in range(1,len(fh))]
        Xs.append(x)
        if td.priming > 0.0:
            priming_vector=x[:]
        steps.append(step)

    alter_graph_size=0
    if td.trim != 1.0:
        numnodes=nx.number_of_nodes(g)
        for i in range(numnodes):
            if i not in set(flatten_list(Xs)):
                alter_graph_size=1

    return Xs, steps, alter_graph_size

# generate random walk that results in observed x
def genZfromX(x, theta, seed=None):
    nplocal=np.random.RandomState(seed)    
    
    x2=x[:]                  # make a local copy
    x2.reverse()
    
    path=[]                  # z to return
    path.append(x2.pop())    # add first two x's to z
    path.append(x2.pop())

    while len(x2) > 0:
        if nplocal.random_sample() < theta:     # might want to set random seed for replicability?
            # add random hidden node
            possibles=set(path) # choose equally from previously visited nodes
            possibles.discard(path[-1]) # but exclude last node (node cant link to itself)
            path.append(nplocal.choice(list(possibles)))
        else:
            # first hit!
            path.append(x2.pop())
    return walk_from_path(path)


def genGraphPrior(graphs, items, fitinfo=Fitinfo({})):
    a=fitinfo.prior_a
    b=fitinfo.prior_b
    priordict={}
    
    # tabulate number of times edge does or doesn't appear in all of the graphs when node pair is present
    for graphnum, graph in enumerate(graphs):   # for each graph
        itemdict=items[graphnum]
        for inum, i in enumerate(graph):     # rows of graph
            for jnum, j in enumerate(i):     # columns of graph
                if inum > jnum:
                    item1 = itemdict[inum]
                    item2 = itemdict[jnum]
                    pair = np.sort((item1,item2))
                    if pair[0] not in priordict.keys():
                        priordict[pair[0]]={}
                    if pair[1] not in priordict[pair[0]].keys():
                        priordict[pair[0]][pair[1]]=[a,b]
                    if j==1:
                        priordict[pair[0]][pair[1]][1] += 1
                    elif j==0:
                        priordict[pair[0]][pair[1]][0] += 1
   
    # use beta distribution to convert to probabilities (of edge being present)
    for item1 in priordict:
        for item2 in priordict[item1]:
            a, b = priordict[item1][item2]      # a=number of participants without link, b=number of participants with link
            priordict[item1][item2] = scipy.stats.beta.cdf(0.5, a, b)
    
    return priordict

def evalGraphPrior(a, prior):
    probs = []
    priordict = prior[0]
    items = prior[1]
    
    for inum, i in enumerate(a):
        for jnum, j in enumerate(i):
            if inum > jnum:
                pair = np.sort((items[inum],items[jnum]))
                try:
                    priorprob = priordict[pair[0]][pair[1]]
                    if j==1:
                        prob = priorprob
                    elif j==0:
                        prob = (1-priorprob)
                except:
                    prob = 0.5  # i.e., scipy.stats.beta.cdf(0.5, 1, 1) -- no information about edge
                probs.append(prob)
    
    probs = [np.log(prob) for prob in probs]      # multiplication probably results in underflow...
    probs = sum(probs)
    return probs

def hierarchicalUinvite(Xs, items, numnodes, td, irts=False, fitinfo=Fitinfo({}), seed=None, debug=True):
    nplocal=np.random.RandomState(seed) 

    # create ids for all subjects
    subs=range(len(Xs))

    if irts:
        for sub in subs:
            [mu, sig, lambd] = mexgauss(flatten_list(irts[sub].data))
            irts[sub].exgauss_sigma = sig
            irts[sub].exgauss_lambda = lambd

    # find starting graphs
    graphs=[]
    for sub in subs:
        td.numx=len(Xs[sub])    # for goni
        graphs.append(genStartGraph(Xs[sub], numnodes[sub], td, fitinfo=fitinfo))

    # cycle though participants
    graphchanges=1
    rnd=1
    while graphchanges > 0:
        if debug: print "Round: ", rnd
        graphchanges = 0
        nplocal.shuffle(subs)
        for sub in subs:
            if debug: print "SS: ", sub

            td.numx = len(Xs[sub])
            fitinfo.startGraph = graphs[sub]

            # generate prior without participant's data, fit graph
            priordict = genGraphPrior(graphs[:sub]+graphs[sub+1:], items[:sub]+items[sub+1:], fitinfo=fitinfo)
            prior = (priordict, items[sub])
            if isinstance(irts, list):
                uinvite_graph, bestval = uinvite(Xs[sub], td, numnodes[sub], fitinfo=fitinfo, prior=prior, irts=irts[sub])
            else:
                uinvite_graph, bestval = uinvite(Xs[sub], td, numnodes[sub], fitinfo=fitinfo, prior=prior)

            if not np.array_equal(uinvite_graph, graphs[sub]):
                graphchanges += 1
                graphs[sub] = uinvite_graph
        rnd += 1

    return graphs, priordict

# construct graph using method using item correlation matrix and planar maximally filtered graph (PMFG)
# see Borodkin, Kenett, Faust, & Mashal (2016) and Kenett, Kenett, Ben-Jacob, & Faust (2011)
# does not work well for small number of lists! many NaN correlations + when two correlations are equal, ordering is arbitrary
def kenett(Xs, numnodes):
    import planarity
    
    # construct matrix of list x item where each cell indicates whether that item is in that list
    list_by_item = np.zeros((numnodes,len(Xs)))
    for node in range(numnodes):
        for x in range(len(Xs)):
            if node in Xs[x]:
                list_by_item[node,x]=1.0
    
    # find pearsonr correlation for all item pairs
    item_by_item = {}
    for item1 in range(numnodes):
        for item2 in range(numnodes):
            if item1 < item2:
                item_by_item[(item1, item2)] = scipy.stats.pearsonr(list_by_item[item1],list_by_item[item2])[0]
    
    corr_vals = sorted(item_by_item, key=item_by_item.get)[::-1]       # keys in correlation dictionary sorted by value (high to low, including NaN first)
    
    # nan correlation occurs when item is in all lists-- exclude from graph (conservative)
    edgelist=[]
    for pair in corr_vals:
        if not np.isnan(item_by_item[pair]):
            edgelist.append(pair)
            if not planarity.is_planar(edgelist):
                edgelist.pop()
    
    g = nx.Graph()
    g.add_edges_from(edgelist)
    a=np.array(nx.to_numpy_matrix(g)).astype(int)
    
    return a

# wrapper returns one graph with theta=0
# aka draw edge between all observed nodes in all lists
def noHidden(Xs, numnodes):
    return genGraphs(1, 0, Xs, numnodes)[0]

# take Xs and convert them from numbers (nodes) to labels
def numToAnimal(Xs, items):
    for lnum, l in enumerate(Xs):
        for inum, i in enumerate(l):
            Xs[lnum][inum]=items[i]
    return Xs

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
#@nogc
def probX(Xs, a, td, irts=Irts({}), prior=None, origmat=None, changed=[], forceCompute=False, pass_link_matrix=True):
    
    numnodes=len(a)
    reg=(1+1e-10)                           # nuisance parameter to prevent errors; can also use pinv, but that's much slower
    identmat=np.identity(numnodes) * reg    # pre-compute for tiny speed-up (only for non-IRT)

    probs=[]

    # generate transition matrix (from: column, to: row) if given link matrix

    if pass_link_matrix:                    # function assumes you pass a link matrix (1s and 0s) unless specified
        t=a/sum(a.astype(float))            # but you can pass a transition matrix if you really want to
    else:                                    
        t=a

    t=np.nan_to_num(t)                        # jumping/priming models can have nan in matrix, need to change to 0

    if (td.jumptype=="stationary") or (td.startX=="stationary"):
        statdist=stationary(t)

    for xnum, x in enumerate(Xs):
        x2=np.array(x)
        t2=t[x2[:,None],x2]                                        # re-arrange transition matrix to be in list order
        prob=[]
        if td.startX=="stationary":
            prob.append(statdist[x[0]])                            # probability of X_1
        elif td.startX=="uniform":
            prob.append(1.0/numnodes)

        # if impossible starting point, return immediately
        if (prob[-1]==0.0) and (not forceCompute):
            return -np.inf, (x[0], x[1])

        if (len(changed) > 0) and isinstance(origmat,list):        # if updating prob. matrix based on specific link changes
            update=0                                               # reset for each list

        for curpos in range(1,len(x)):
            if (len(changed) > 0) and isinstance(origmat,list):
                if update==0:                                      # first check if probability needs to be updated
                    if (Xs[xnum][curpos-1] in changed):            # (only AFTER first changed node has been reached)
                        update=1
                    else:                                          # if not, take probability from old matrix
                        prob.append(origmat[xnum][curpos])
                        continue
            Q=t2[:curpos,:curpos]

            if (len(irts.data) > 0) and (irts.irt_weight < 1):     # use this method only when passing IRTs with weight < 1
                numcols=len(Q)
                flist=[]
                newQ=np.zeros(numcols)                             # init to Q^0, for when r=1
                newQ[curpos-1]=1.0                                 # (using only one: row for efficiency)

                irt=irts.data[xnum][curpos-1]

                # precompute for small speedup
                if irts.irttype=="gamma":
                    logbeta=np.log(irts.beta)
                    logirt=np.log(irt)

                for r in range(1,irts.rcutoff):
                    innersum=0
                    for k in range(numcols):
                        num1=newQ[k]                               # probability of being at node k in r-1 steps
                        num2=t2[curpos,k]                          # probability transitioning from k to absorbing node    
                        innersum=innersum+(num1*num2)

                    # much faster than using scipy.stats.gamma.pdf
                    if irts.irttype=="gamma":
                        log_dist=r*logbeta-math.lgamma(r)+(r-1)*logirt-irts.beta*irt # r=alpha. probability of observing irt at r steps
                    if irts.irttype=="exgauss":
                        log_dist=np.log(irts.exgauss_lambda/2.0)+(irts.exgauss_lambda/2.0)*(2.0*r+irts.exgauss_lambda*(irts.exgauss_sigma**2)-2*irt)+np.log(math.erfc((r+irts.exgauss_lambda*(irts.exgauss_sigma**2)-irt)/(np.sqrt(2)*irts.exgauss_sigma)))

                    if innersum > 0: # sometimes it's not possible to get to the target node in r steps
                        flist.append(log_dist*(1-irts.irt_weight)+irts.irt_weight*np.log(innersum))

                    newQ=np.inner(newQ,Q)                          # raise power by one

                f=sum([np.e**i for i in flist])
                prob.append(f)                                     # probability of x_(t-1) to X_t
            else:                                                  # if no IRTs, use standard U-INVITE
                I=identmat[:len(Q),:len(Q)]
                R=t2[curpos,:curpos]
                N=np.linalg.solve(I-Q,I[-1])
                B=np.dot(R,N)
                if np.isnan(B):
                    B=0.0
                prob.append(B)
                
                # alternative/original using matrix inverse
                #R=t2[curpos:,:curpos]
                #N=inv(I-Q)
                #B=np.dot(R,N)                
                #prob.append(B[0,curpos-1])

            # if there's an impossible transition and no jumping/priming, return immediately
            if (prob[-1]==0.0) and (td.jump == 0.0) and (td.priming == 0.0) and (not forceCompute):
                return -np.inf, (x[curpos-1], x[curpos])

        probs.append(prob)

    uinvite_probs = np.copy(probs)      # store only u-invite transition probabilities (the computationally hard stuff) to avoid recomputing
    
    # adjust for jumping probability
    if td.jump > 0.0:
        if td.jumptype=="uniform":
            probs=addJumps(probs, td, numnodes=numnodes)
        elif td.jumptype=="stationary":
            probs=addJumps(probs, td, statdist=statdist, Xs=Xs)

    if (td.priming > 0.0):
        probs=adjustPriming(probs, td, Xs)

    # check for impossible transitions after priming and jumping
    if not forceCompute:
        for xnum, x in enumerate(probs):
            for inum, i in enumerate(x):
                if (i==0.0) and (inum==0):
                    return -np.inf, (Xs[xnum][inum], Xs[xnum][inum+1])  # link to next item when first item is unreachable
                elif (i==0.0) and (inum > 0):
                    return -np.inf, (Xs[xnum][inum-1], Xs[xnum][inum])  # link to previous item otherwise
                
    # total ll of graph
    try:
        ll=sum([sum([np.log(j) for j in probs[i]]) for i in range(len(probs))])
    except:
        ll=-np.inf

    # inclue prior?
    if prior:
        if isinstance(prior, tuple):    # graph prior
            priorlogprob = evalGraphPrior(a, prior)
            ll = ll + priorlogprob
        else:                           # smallworld prior
            sw=smallworld(a)
            priorprob = evalSWprior(sw, prior)
            if priorprob == 0.0:
                return -np.inf, "prior"
            else:
                ll = ll + np.log(priorprob)
    return ll, uinvite_probs

# given an adjacency matrix, take a random walk that hits every node; returns a list of tuples
def random_walk(g, td, priming_vector=[], seed=None):
    nplocal=np.random.RandomState(seed)    

    def jump():
        if td.jumptype=="stationary":
            second=statdist.rvs(random_state=seed)       # jump based on statdist
        elif td.jumptype=="uniform":
            second=nplocal.choice(nx.nodes(g))         # jump uniformly
        return second

    if (td.startX=="stationary") or (td.jumptype=="stationary"):
        a=np.array(nx.to_numpy_matrix(g))
        t=a/sum(a).astype(float)
        statdist=stationary(t)
        statdist=scipy.stats.rv_discrete(values=(range(len(t)),statdist))
    
    if td.startX=="stationary":
        start=statdist.rvs(random_state=seed)      # choose starting point from stationary distribution
    elif td.startX=="uniform":
        start=nplocal.choice(nx.nodes(g))        # choose starting point uniformly
    elif td.startX[0]=="specific":
        start=td.startX[1]

    walk=[]
    unused_nodes=set(nx.nodes(g))
    unused_nodes.remove(start)
    first=start
    
    numnodes=nx.number_of_nodes(g)
    if td.trim <= 1:
        numtrim=int(round(numnodes*td.trim))       # if <=1, paramater is proportion of a list
    else:
        numtrim=td.trim                            # else, parameter is length of a list
    num_unused = numnodes - numtrim

    censoredcount=0                                # keep track of censored nodes and jump after td.jumponcensored censored nodes

    while len(unused_nodes) > num_unused:

        # jump after n censored nodes or with random probability (depending on parameters)
        if (censoredcount == td.jumponcensored) or (nplocal.random_sample() < td.jump):
            second=jump()
        else:                                           # no jumping!
            second=nplocal.choice([x for x in nx.all_neighbors(g,first)]) # follow random edge (actual random walk!)
            if (td.priming > 0.0) and (len(priming_vector) > 0):
                if (first in priming_vector[:-1]) & (nplocal.random_sample() < td.priming):      
                    idx=priming_vector.index(first)
                    second=priming_vector[idx+1]          # overwrite RW... kinda janky
        walk.append((first,second))
        if second in unused_nodes:
            unused_nodes.remove(second)
            censoredcount=0
        else:
            censoredcount += 1
        first=second
    return walk

# return small world statistic of a graph
# returns metric of largest component if disconnected
def smallworld(a):
    if isinstance(a,np.ndarray):
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
    l_rand= np.log(numnodes)/np.log(nodedegree)    # approximation, see humphries & gurney (2008) eq 11
    #l_rand= (np.log(numnodes)-0.5772)/(np.log(nodedegree)) + .5 # alternative ASPL from fronczak, fronczak & holyst (2004)
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
    nplocal=np.random.RandomState(seed)        # to generate the same IRTs each time
    new_irts=[]
    for irtlist in irts.data:
        if irts.irttype=="gamma":
            newlist=[nplocal.gamma(irt, (1.0/irts.beta)) for irt in irtlist]  # beta is rate, but random.gamma uses scale (1/rate)
        if irts.irttype=="exgauss":
            newlist=[rand_exg(irt, irts.exgauss_sigma, irts.exgauss_lambda) for irt in irtlist] 
        new_irts.append(newlist)
    return new_irts

# ** this function is not really needed anymore since moving functionality to genX, 
# ** but there may be some niche cases where needed...
# trim Xs to proportion of graph size, the trim graph to remove any nodes that weren't hit
# used to simulate human data that doesn't cover the whole graph every time
def trimX(trimprop, Xs, steps):
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

#@profile
def uinvite(Xs, td, numnodes, irts=Irts({}), fitinfo=Fitinfo({}), prior=None, debug=True, recordname="records.csv", seed=None):
    nplocal=np.random.RandomState(seed) 

    # return list of neighbors of neighbors of i, that aren't themselves neighbors of i
    # i.e., an edge between i and any item in nn forms a triangle
    #@profile
    def neighborsofneighbors(i, nxg):
        nn=[]                                       # neighbors of neighbors (nn)
        n=list(nx.all_neighbors(nxg,i))
        for j in n:
            nn=nn+list(nx.all_neighbors(nxg,j))
        nn=list(set(nn))
        for k in n:                                 # remove neighbors
            if k in nn:
                nn.remove(k)
        if i in nn:
            nn.remove(i)                            # remove self
        return nn
        
    # toggle links back, should be faster than making graph copy
    #@profile
    def swapEdges(graph,links):
        for link in links:
            graph[link[0],link[1]] = 1 - graph[link[0],link[1]]
            if not fitinfo.directed:
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
            if debug:
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
            if debug:
                print "Adding triangles", str(vmaj) + "." + str(vmin), "... ", # (len(edges)/2), "possible:",
            sys.stdout.flush()
            nn=dict()
            for i in range(len(graph)):
                nn[i]=neighborsofneighbors(i, nxg)
            v=nn
        
        # generate dict where v[i] is a list of nodes where (i, v[i]) is NOT an existing an edge and does NOT form a triangle
        if (method=="nonneighbors") or (method==2):
            # list of a node's non-neighbors (non-edges) that don't form triangles
            if debug:
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
            node1=nplocal.choice(bestnodes)

            if len(v[node1]) > 0:
                #node2=nplocal.choice(v[node1]) # old
                
                n2avg=[avg[i] for i in v[node1]]
                maxval=max(n2avg)
                bestnodes=[v[node1][i] for i, j in enumerate(n2avg) if j == maxval]
                node2=nplocal.choice(bestnodes)

                edge=(node1, node2)
                graph=swapEdges(graph,[edge])

                graph_ll, newprobmat=probX(Xs,graph,td,irts=irts,prior=prior,origmat=probmat,changed=[node1,node2])

                if best_ll >= graph_ll:
                    record.append(graph_ll)
                    graph=swapEdges(graph,[edge])
                else:
                    record.append(-graph_ll)
                    best_ll = graph_ll
                    probmat = newprobmat
                    numchanges += 1
                    loopcount = 0
                v[node1].remove(node2)   # remove edge from possible choices
                if not fitinfo.directed:
                    v[node2].remove(node1)
           
                # increment even if graph prob = -np.inf for implicit penalty
                count[node1] += 1
                count[node2] += 1
                if (graph_ll != -np.inf) and (fitinfo.followtype != "random"):
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

        if debug:
            print numchanges, "changes"

        records.append(record)
        return graph, best_ll, probmat, numchanges

    #    return graph
    
    def phases(graph, best_ll, probmat):
        complete=[0,0,0]         # marks which phases are complete
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
        return graph, best_ll

    firstedges=[(x[0], x[1]) for x in Xs]
    
    # find a good starting graph using naive RW
    graph = genStartGraph(Xs, numnodes, td, fitinfo)

    best_ll, probmat = probX(Xs,graph,td,irts=irts,prior=prior)   # LL of starting graph
    records=[]
    graph, best_ll = phases(graph, best_ll, probmat)
    if fitinfo.record:
        f=open(fitinfo.recorddir+recordname,'w')
        wr=csv.writer(f)
        for record in records:
            wr.writerow(record)

    return graph, best_ll

# tuple walk from flat list
def walk_from_path(path):
    walk=[]
    for i in range(len(path)-1):
        walk.append((path[i],path[i+1])) 
    return walk

# w = window size; two items appear within +/- w steps of each other (where w=1 means adjacent items)
# f = filter frequency; if two items don't fall within the same window more than f times, then no edge is inferred
# c = confidence interval; retain the edge if there is a <= c probability that two items occur within the same window n times by chance alone
# valid (t/f) ensures that graph can produce data using censored RW.
def goni(Xs, numnodes, fitinfo=Fitinfo({}), c=0.05, valid=False, td=0):
    w=fitinfo.goni_size
    f=fitinfo.goni_threshold
    
    if f<1:                 # if <1 treat as proportion of total lists; if >1 treat as absolute # of lists
        f=int(round(len(Xs)*f))

    if valid and td==0:
        raise ValueError('Need to pass Toydata when generating \'valid\' goni()')

    if c<1:
        from statsmodels.stats.proportion import proportion_confint as pci

    if w < 1:
        print "Error in goni(): w must be >= 1"
        return

    graph=np.zeros((numnodes, numnodes)).astype(int)         # empty graph

    # frequency of co-occurrences within window (w)
    for x in Xs:                                             # for each list
        for pos in range(len(x)):                            # for each item in list
            for i in range(1, w+1):                          # for each window size
                if pos+i<len(x):
                    graph[x[pos],x[pos+i]] += 1
                    graph[x[pos+i],x[pos]] += 1

    # exclude edges with co-occurrences less than frequency (f) and binarize
    # but first save co-occurence frequencies
    cooccur = np.copy(graph)
    for i in range(len(graph)):
        for j in range(len(graph)):
            if graph[i, j] < f:
                graph[i, j] = 0
            else:
                graph[i, j] = 1

    # check if co-occurrences are due to chance
    if c<1:
        setXs=[list(set(x)) for x in Xs]                              # unique nodes in each list
        flatX=flatten_list(setXs)                                     # flattened
        xfreq=[flatX.count(i) for i in range(numnodes)]               # number of lists each item appears in (at least once)
        listofedges=zip(*np.nonzero(graph))                           # list of edges in graph to check
        numlists=float(len(Xs))
        meanlistlength=np.mean([len(x) for x in Xs])
    
        # Goni et al. (2011), eq. 10
        p_adj = (2.0/(meanlistlength*(meanlistlength-1))) * ((w*meanlistlength) - ((w*(w+1))/2.0))
        for i,j in listofedges:
            p_linked = (xfreq[i]/numlists) * (xfreq[j]/numlists) * p_adj
            ci=pci(cooccur[i,j],numlists,alpha=c,method="beta")[0]    # lower bound of Clopper-Pearson binomial CI
            if p_linked >= ci:                                        # if co-occurrence could be due to chance, remove edge
                graph[i,j]=0
                graph[j,i]=0

    if valid:
        # add direct edges when transition is impossible
        check=probX(Xs, graph, td)
        while check[0] == -np.inf:
            if isinstance(check[1],tuple):
                graph[check[1][0], check[1][1]] = 1
                graph[check[1][1], check[1][0]] = 1
            elif check[1] == "prior":
                raise ValueError('Starting graph has prior probability of 0.0')
            else:
                raise ValueError('Unexpected error from goni()')
            check=probX(Xs, graph, td)

    # make sure there are no self-transitions
    for i in range(len(graph)):
        graph[i,i]=0.0

    return graph
