#!/usr/bin/python

import networkx as nx
import numpy as np
import random
import operator
import math
import matplotlib.pyplot as plt
import scipy
from numpy.linalg import inv
import textwrap
from itertools import *
from datetime import datetime
from ExGUtils.stats import *
from ExGUtils.exgauss import *

# TODO: unit tests
# TODO: reset random seed after each method
# TODO: print num changes, cost for each toy network

# Set the random seed to allow search process to be replicable. Untested.
# Currently needed because search procedure is inconsistent
randomseed=1

# objective graph cost
# returns the number of links that need to be added or removed to reach the true graph
def cost(graph,a):
    return sum(sum(np.array(abs(graph-a))))/2

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


# write graph to GraphViz file (.dot)
def drawDot(g, filename, labels={}):
    if type(g) == np.ndarray:
        g=nx.to_networkx_graph(g)
    if labels != {}:
        nx.relabel_nodes(g, labels, copy=False)
    nx.drawing.write_dot(g, filename)

# draw graph
def drawG(g,Xs=[],labels={},save=False,display=True):
    if type(g) == np.ndarray:
        g=nx.to_networkx_graph(g)
    nx.relabel_nodes(g, labels, copy=False)
    #pos=nx.spring_layout(g, scale=5.0)
    pos = nx.graphviz_layout(g, prog="fdp")
    nx.draw_networkx(g,pos,node_size=1000)
#    for node in range(numnodes):                    # if the above doesn't work
#        plt.annotate(str(node), xy=pos[node])       # here's a workaround
    if Xs != []:
        plt.title(Xs)
    plt.axis('off')
    if save==True:
        plt.savefig('temp.png')                      # need to parameterize
    if display==True:
        plt.show()

# returns a vector of how many hidden nodes to expect between each Xi for each X in Xs
def expectedHidden(Xs, a, numnodes):
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

def findBestGraph(Xs, irts=[], jeff=0.5, beta=1.0, numnodes=0, tolerance=1500, kde=0):
    random.seed(randomseed)     # for replicability
   
    # free parameters
    prob_overlap=.8     # probability a link connecting nodes in multiple graphs
    prob_multi=.8       # probability of selecting an additional link

    if numnodes==0:         # unless specified (because Xs are trimmed and dont cover all nodes)
        numnodes=len(set(flatten_list(Xs)))

    converge = 0        # when converge >= tolerance, declare the graph converged.
    itern=0 # tmp variable for debugging

    # find a good starting graph using naive RW
    #graph=noHidden(Xs,numnodes)
    graph=genGraphs(1, .5, Xs, numnodes)[0]

    best_ll=probX(Xs,graph,numnodes,irts,jeff,beta)   # LL of best graph found
    #JZ
    sw=smallworld(graph)
    best_ll=best_ll + math.log(kde(sw)[0])
    #ZJ

    # items in at least 2 lists. links between these nodes are more likely to affect P(G)
    # http://stackoverflow.com/q/2116286/353278
    overlap=reduce(set.union, (starmap(set.intersection, combinations(map(set, Xs), 2))))
    overlap=list(overlap)
    combos=list(combinations(overlap,2))    # all possible links btw overlapping nodes
    firstedges=[(x[0], x[1]) for x in Xs]

    while converge < tolerance:
        
        links=[]        # links to toggle in candidate graph
        while True:     # emulates do-while loop (ugly)
            if random.random() <= prob_overlap:      # choose link from combos most of the time
                link=random.choice(combos)
            else:                                    # sometimes choose a link at random
                link=(0,0)
                # avoid self-transition, avoid first edges (for efficiency only, since FE are required to produce data)
                while (link[0]==link[1]) or (link in firstedges) or (link[::-1] in firstedges): 
                    link=(int(math.floor(random.random()*numnodes)),int(math.floor(random.random()*numnodes)))
            links.append(link)
            if random.random() <= prob_multi:
                break

        # toggle links
        for link in links:
            graph[link[0],link[1]] = 1 - graph[link[0],link[1]] 
            graph[link[1],link[0]] = 1 - graph[link[1],link[0]]

        graph_ll=probX(Xs,graph,numnodes,irts,jeff,beta)
        #JZ
        sw=smallworld(graph)
        graph_ll=graph_ll + math.log(kde(sw)[0])
        #ZJ
        
        print "ORIG: ", best_ll, " NEW: ", graph_ll

        # for debugging... make sure its testing lots of graphs
        #itern=itern+1
        #if itern % 100 == 0:
            #print itern

        # if graph is better than current graph, accept it
        if (graph_ll > best_ll):
            # for debugging
            #print "GRAPH: ", graph_ll, "BEST: ", best_ll
            best_ll = graph_ll
            converge = 0          # reset convergence criterion only if new graph is better than BEST graph
        else:
            converge += 1
            # toggle links back, should be faster than making graph copy
            for link in links:
                graph[link[0],link[1]] = 1 - graph[link[0],link[1]]
                graph[link[1],link[0]] = 1 - graph[link[1],link[0]] 
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

# helper function generate flast lists from nested lists
def flatten_list(l):
    return [item for sublist in l for item in sublist]

# generate a connected Watts-Strogatz small-world graph
# (n,k,p) = (number of nodes, each node connected to k-nearest neighbors, probability of rewiring)
# k has to be even, tries is number of attempts to make connected graph
def genG(n,k,p,tries=1000, seed=None, graphtype="sw"):
    if graphtype=="sw":
        g=nx.connected_watts_strogatz_graph(n,k,p,tries,seed) # networkx graph
        random.seed(randomseed)                               # bug in nx, random seed needs to be reset    
        a=np.array(nx.adjacency_matrix(g).todense())          # adjacency matrix
    else: # erdos-renyi... TODO fix up
        g=nx.erdos_renyi_graph(n,p)
    return g, np.array(a, dtype=np.int32)

# only returns adjacency matrix, not nx graph
def genGfromZ(walk, numnodes):
    a=np.zeros((numnodes,numnodes))
    for i in set(walk):
        a[i[0],i[1]]=1
        a[i[1],i[0]]=1 # symmetry
    a=np.array(a.astype(int))
    return a

def genGraphs(numgraphs, theta, Xs, numnodes):
    Zs=[reduce(operator.add,[genZfromX(x,theta) for x in Xs]) for i in range(numgraphs)]
    As=[genGfromZ(z, numnodes) for z in Zs]
    return As

# return simulated data on graph g
# if usr_irts==1, return irts (as steps)
def genX(g,s=None,use_irts=0,seed=None):
    rwalk=random_walk(g,s,seed)
    Xs=observed_walk(rwalk)
    
    if use_irts==0:
        return Xs
    else:
        fh=list(zip(*firstHits(rwalk))[1])
        irts=[fh[i]-fh[i-1] for i in range(1,len(fh))]
        return Xs, irts

# generate random walk that results in observed x
def genZfromX(x, theta):
    x2=x[:]                  # make a local copy
    x2.reverse()
    
    path=[]                  # z to return
    path.append(x2.pop())    # add first two x's to z
    path.append(x2.pop())

    while len(x2) > 0:
        if random.random() < theta:     # might want to set random seed for replicability?
            # add random hidden node
            possibles=set(path) # choose equally from previously visited nodes
            possibles.discard(path[-1]) # but exclude last node (node cant link to itself)
            path.append(random.choice(list(possibles)))
        else:
            # first hit!
            path.append(x2.pop())
    return walk_from_path(path)

# DEPRECATED
# search for best graph by hill climbing with stochastic search
# returns numkeep graphs with the best graph at index 0
def graphSearch(graphs,numkeep,Xs,numnodes,jeff=0.5,irts=[],prior=0,beta=1,maxlen=20):
    loglikelihood=[]
    
    for it, graph in enumerate(graphs):
        tmp=probX(Xs,graph,numnodes,irts,jeff,beta)
        if prior:
            priordistribution=scipy.stats.norm(loc=4.937072,scale=0.5652063)
            sw=smallworld(graph)
            tmp=tmp + math.log(priordistribution.pdf(sw))

        loglikelihood.append(tmp)
    
    maxvals=maxn(loglikelihood,numkeep)
    maxpos=[loglikelihood.index(i) for i in maxvals]
    maxgraphs=[graphs[i] for i in maxpos]
    print "MAX: ", max(loglikelihood)
    return maxgraphs, max(loglikelihood)

# helper function converts binary adjacency matrix to base 36 string for easy storage in CSV
# binary -> int -> base36
def graphToHash(a,numnodes):
    def baseN(num,b,numerals="0123456789abcdefghijklmnopqrstuvwxyz"):
        return ((num == 0) and numerals[0]) or (baseN(num // b, b, numerals).lstrip(numerals[0]) + numerals[num % b])
    return str(numnodes) + '!' + baseN(int(''.join([str(i) for i in flatten_list(a)]),2), 36)

def hashToGraph(graphhash):
    numnodes, graphhash = graphhash.split('!')
    graphstring=bin(int(graphhash, 36))[2:]
    zeropad=numnodes**2-len(graphstring)
    graphstring=''.join(['0' for i in range(zeropad)]) + graphstring
    arrs=textwrap.wrap(graphstring, numnodes)
    mat=np.array([map(int, s) for s in arrs])
    return mat

# log trick given list of log-likelihoods **UNUSED
def logTrick(loglist):
    logmax=max(loglist)
    loglist=[i-logmax for i in loglist]                     # log trick: subtract off the max
    p=math.log(sum([math.e**i for i in loglist])) + logmax  # add it back on
    return p

# helper function grabs highest n items from list items
# http://stackoverflow.com/questions/350519/getting-the-lesser-n-elements-of-a-list-in-python
def maxn(items,n):
    maxs = items[:n]
    maxs.sort(reverse=True)
    for i in items[n:]:
        if i > maxs[-1]: 
            maxs.append(i)
            maxs.sort(reverse=True)
            maxs= maxs[:n]
    return maxs

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

# probability of observing Xs, including irts
def probX(Xs, a, numnodes, irts=[], jeff=0.5, beta=1, maxlen=20, irtmethod="gamma",mattype="link"):
    #random.seed(randomseed)             # bug in nx, random seed needs to be reset    
    probs=[] 

    # generate transition matrix (from: column, to: row) if given link matrix
    if mattype=="link":
        t=a/sum(a.astype(float))            # will throw warning if a node is inaccessible
    else:
        t=a

    statdist=stationary(t)
    
    for xnum, x in enumerate(Xs):
        prob=[]
        prob.append(statdist[x[0]])      # probability of X_1

        # if impossible starting point, return immediately
        # TODO: check/ shouldn't this be under for curpos loop? does it make a difference (in terms of speed)?
        if prob[-1]==0.0:
            return -np.inf
        
        notinx=[i for i in range(numnodes) if i not in x]        # nodes not in trimmed X

        for curpos in range(1,len(x)):
            startindex=x[curpos-1]
            deletedlist=sorted(x[curpos:]+notinx,reverse=True)
            notdeleted=np.array([i for i in range(numnodes) if i not in deletedlist])
           
            Q=t[notdeleted[:, None],notdeleted]
                
            if (len(irts) > 0) and (jeff < 1): # use this method only when passing IRTs with weight < 1
                startindex = startindex-sum([startindex > i for i in deletedlist])
                # same as startindex==sorted(x[:curpos]).index(x[curpos-1])... less readable, maybe more efficient?
                
                numcols=len(Q)
                flist=[]
                newQ=np.zeros(numcols)  # init to Q^0, for when r=1 (using only one row for efficiency)
                newQ[startindex]=1.0

                irt=irts[xnum][curpos-1]

                # precompute for small speedup
                if irtmethod=="gamma":
                    logbeta=math.log(beta)
                    logirt=math.log(irt)

                for r in range(1,maxlen):
                    innersum=0
                    for k in range(numcols):
                        num1=newQ[k]                        # probability of being at node k in r-1 steps
                        num2=t[x[curpos],notdeleted[k]]     # probability transitioning from k to absorbing node    
                        innersum=innersum+(num1*num2)

                    # much faster than using scipy.stats.gamma.pdf

                    if irtmethod=="gamma":
                        log_dist=r*logbeta-math.lgamma(r)+(r-1)*logirt-beta*irt # r=alpha. probability of observing irt at r steps
                    if irtmethod=="exgauss":
                        tau=.5
                        sig=.5
                        # same as math.log(exgauss(irt,r,sig,tau)) but avoids underflow
                        log_dist=math.log(tau/2.0)+(tau/2.0)*(2.0*r+tau*(sig**2)-2*irt)+math.log(math.erfc((r+tau*(sig**2)-irt)/(math.sqrt(2)*sig)))

                    if innersum > 0: # sometimes it's not possible to get to the target node in r steps
                        flist.append(log_dist*(1-jeff)+jeff*math.log(innersum))
                    newQ=np.inner(newQ,Q)     # raise power by one

                f=sum([math.e**i for i in flist])
                prob.append(f)           # probability of x_(t-1) to X_t
            else:                        # if no IRTs, use standard INVITE
                I=np.identity(len(Q))
                reg=(1+1e-10)             # nuisance parameter to prevent errors
                N=inv(I*reg-Q)
                
                r=np.array(sorted(x[curpos:]))
                c=sorted(x[:curpos])
                R=t[r[:,None],c]

                B=np.dot(R,N)
                startindex=sorted(x[:curpos]).index(x[curpos-1])
                absorbingindex=sorted(x[curpos:]).index(x[curpos])
                prob.append(B[absorbingindex,startindex])
            
            # if there's an impossible transition, return immediately
            if prob[-1]==0.0:
                return -np.inf

        probs.append(prob)
    for i in range(len(probs)):
        probs[i]=sum([math.log(j) for j in probs[i]])
    probs=sum(probs)
    return probs

# read Xs in from user files
def readX(subj,category,filepath):
    if type(subj) == str:
        subj=[subj]
    game=-1
    cursubj=-1
    Xs=[]
    irts=[]
    items={}
    idx=0
    with open(filepath) as f:
        for line in f:
            row=line.split(',')
            if (row[0] in subj) & (row[2] == category):
                if (row[1] != game) or (row[0] != cursubj):
                    Xs.append([])
                    irts.append([])
                    game=row[1]
                    cursubj=row[0]
                item=row[3]
                irt=row[4]
                if item not in items.values():
                    items[idx]=item
                    idx += 1
                itemval=items.values().index(item)
                if itemval not in Xs[-1]:   # ignore any duplicates in same list resulting from spelling corrections
                    Xs[-1].append(itemval)
                    irts[-1].append(int(irt)/1000.0)
    numnodes = len(items)
    return Xs, items, irts, numnodes

# given an adjacency matrix, take a random walk that hits every node; returns a list of tuples
def random_walk(g,start=None,seed=None):
    myrandom=random.Random(seed)
    if start is None:
        start=myrandom.choice(nx.nodes(g))      # choose starting point uniformly
    elif isinstance(start,scipy.stats._distn_infrastructure.rv_discrete):
        start=start.rvs(random_state=seed)      # choose starting point from stationary distribution

    walk=[]
    unused_nodes=set(nx.nodes(g))
    unused_nodes.remove(start)
    while len(unused_nodes) > 0:
        p=start
        start=myrandom.choice([x for x in nx.all_neighbors(g,start)]) # follow random edge
        walk.append((p,start))
        if start in unused_nodes:
            unused_nodes.remove(start)
    return walk


# return small world statistic of a graph
# returns metric of largest component if disconnected
def smallworld(a):
    g_sm=nx.from_numpy_matrix(a)
    g_sm=max(nx.connected_component_subgraphs(g_sm),key=len)   # largest component
    numnodes=g_sm.number_of_nodes()
    numedges=g_sm.number_of_edges()
    numlinks=numedges/(numnodes*2.0)
    
    c_sm=nx.average_clustering(g_sm)
    l_sm=nx.average_shortest_path_length(g_sm)
    
    # c_rand same as edge density for a random graph? not sure if "-1" belongs in denominator, double check
    c_rand= (numedges*2.0)/(numnodes*(numnodes-1))     
    l_rand= math.log(numnodes)/math.log(2*numlinks)    # see humphries & gurney (2006) eq 11
    #l_rand= (math.log(numnodes)-0.5772)/(math.log(2*numlinks)) + .5 # alternative APL from fronczak, fronczak & holyst (2004)
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
def stepsToIRT(irts, beta=1.0, method="gamma", seed=None):
    myrandom=np.random.RandomState(seed)        # to generate the same IRTs each time
    new_irts=[]
    for irtlist in irts:
        if method=="gamma":
            newlist=[myrandom.gamma(irt, (1.0/beta)) for irt in irtlist]  # beta is rate, but random.gamma uses scale (1/rate)
        if method=="exgauss":
            sig=0.5
            tau=0.5
            newlist=[rand_exg(irt, 0.5, (1/0.5)) for irt in irtlist]  # ex-gaussian
        new_irts.append(newlist)
    return new_irts

# runs a batch of toy graphs. logging code needs to be cleaned up significantly
def toyBatch(numgraphs, numnodes, numlinks, probRewire, numx, trim, jeff, beta, outfile, start_seed=0, 
             methods=['rw','invite','inviteirt','fe'],header=1):

    # PRIOR - move when ready, lines marked JZ/ZJ
    #JZ
    print "generating prior distribution..."
    import scipy
    sw=[]
    for i in range(10000):
        g=nx.connected_watts_strogatz_graph(50,4,p=.3,tries=1000)
        g=nx.to_numpy_matrix(g)
        sw.append(smallworld(g))
    kde=scipy.stats.gaussian_kde(sw)
    print "...done"
    #ZJ

    # break out of function if using unknown method
    for method in methods:
        if method not in ['rw','invite','inviteirt','fe']:
            print "ERROR: Trying to fit graph with unknown method: ", method
            raise

    # stuff to write to file
    globalvals=['jeff','beta','numnodes','numlinks','probRewire',
                'numedges','graph_seed','numx','trim','x_seed']               # same across all methods
    methodvals=['cost','time','bestgraph','hit','miss','fa','cr','bestval']   # differ per method

    f=open(outfile,'a', 0)                # write/append to file with no buffering
    if header==1:
        f.write(','.join(globalvals))
        f.write(',toynetwork,')
        for method in methods:
            towrite=[i+'_'+method for i in methodvals]
            f.write(','.join(towrite))
            f.write(',')
        f.write('\n')
 
    # store all data in dict to write to file later
    data={}
    for method in methods:
        data[method]={}
        for val in methodvals:
            data[method][val]=[]

    numedges=numnodes*(numlinks/2)        # number of edges in graph    

    # how many graphs to run?
    seed_param=start_seed
    last_seed=start_seed+numgraphs

    while seed_param < last_seed:
        graph_seed=seed_param
        x_seed=seed_param

        # generate toy data
        g,a=genG(numnodes,numlinks,probRewire,seed=graph_seed)
        t=a/sum(a).astype(float)
        statdist=stationary(t)     # TODO: put this stuff in genX or somewhere else so user doesn't have to deal with it
        randstart=scipy.stats.rv_discrete(values=(range(len(t)),statdist))
        
        [Xs,steps]=zip(*[genX(g, s=randstart,seed=x_seed+i,use_irts=1) for i in range(numx)])
        Xs=list(Xs)
        steps=list(steps)

        # generate IRTs if using IRT model
        irts=[]
        if 'inviteirt' in methods:
            irts=stepsToIRT(steps, beta, seed=x_seed)

        # trim data when necessary
        [Xs,irts,alter_graph]=trimX(trim,Xs,irts,g)

        if alter_graph==0:      # only use data that covers entire graph (only a problem when trimming data)
            for method in methods:
                print "SEED: ", seed_param, "method: ", method
                
                # Find best graph! (and log time)
                starttime=datetime.now()
                if method == 'invite':
                    bestgraph, bestval=findBestGraph(Xs, numnodes=numnodes, kde=kde) # JZ
                    print graphToHash(bestgraph,numnodes)
                if method == 'inviteirt':
                    bestgraph, bestval=findBestGraph(Xs, irts, jeff, beta, numnodes)
                    print graphToHash(bestgraph,numnodes)
                if method == 'rw':
                    bestgraph=noHidden(Xs,numnodes)
                    bestval=probX(Xs, bestgraph, numnodes)
                if method == 'fe':
                    bestgraph=firstEdge(Xs,numnodes)
                    bestval=probX(Xs, bestgraph, numnodes)
                elapsedtime=str(datetime.now()-starttime)
                print elapsedtime
        
                # compute SDT
                hit, miss, fa, cr = costSDT(bestgraph,a)

                # Record cost, time elapsed, LL of best graph, hash of best graph, and SDT measures
                data[method]['cost'].append(cost(bestgraph,a))
                data[method]['time'].append(elapsedtime)
                data[method]['bestval'].append(bestval)
                data[method]['bestgraph'].append(graphToHash(bestgraph,numnodes))
                data[method]['hit'].append(hit)
                data[method]['miss'].append(miss)
                data[method]['fa'].append(fa)
                data[method]['cr'].append(cr)
    
            # log stuff here
            if outfile != '':
                towrite=[str(eval(i)) for i in globalvals] # EVAL!!!
                f.write(','.join(towrite))
                f.write(','+graphToHash(a,numnodes))
                for method in methods:
                    for val in methodvals:
                        f.write(','+str(data[method][val][-1]))
                f.write('\n')
        else:
            last_seed = last_seed + 1       # if data is unusable (doesn't cover whole graph), add another seed
        seed_param = seed_param + 1
    f.close()

# trim Xs to proportion of graph size, the trim graph to remove any nodes that weren't hit
# used to simulate human data that doesn't cover the whole graph every time
def trimX(prop, Xs, steps, g):
    numnodes=g.number_of_nodes()
    alter_graph_size=0              # report if graph size changes-- may result in disconnected graph!
    numtrim=int(round(numnodes*prop))
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

def write_csv(gs, fh, subj="NA"):
    fh=open(fh,'w',0)
    if isinstance(gs,nx.classes.graph.Graph):
        edges=set(flatten_list([gs.edges()]))
        for edge in edges:
            fh.write(subj    + "," +
                    edge[0]  + "," +
                    edge[1]  + "\n")
    else:
        onezero={True: '1', False: '0'}        
        edges=set(flatten_list([gs[i].edges() for i in range(len(gs))]))
        for edge in edges:
            edgelist=""
            for g in gs:
                edgelist=edgelist+","+onezero[g.has_edge(edge[0],edge[1])]
            fh.write(subj    + "," +
                    edge[0]  + "," +
                    edge[1]  + 
                    edgelist + "\n")
    return
