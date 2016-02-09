#!/usr/bin/python

import networkx as nx
import numpy as np
import random
import operator
import math
import matplotlib.pyplot as plt
import time
import scipy
from scipy import stats
from numpy.linalg import inv
from scipy.optimize import fmin
import sys
import textwrap
from itertools import *
from datetime import datetime

# create toy graph object. currently only small-world
#class ToyGraph:
#    def __init__(self, numnodes, numlinks, probRewire, graph_seed=None):
#
#        self.numnodes = numnodes                # number of nodes in graph
#        self.numlinks = numlinks                # initial number of edges per node (must be even)
#        self.probRewire = probRewire            # probability of re-wiring an edge
#        self.numedges = numnodes*(numlinks/2)   # number of edges in smallworld-graph
#        
#        self.g, self.a = genG(numnodes,numlinks,probRewire,seed=graph_seed)

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

def findBestGraph(Xs, irts=[], jeff=0.5, beta=1.0, numnodes=0, tolerance=1500):
    # free parameters
    prob_overlap=.8     # probability a link connecting nodes in multiple graphs
    prob_multi=.8       # probability of selecting an additional link

    if numnodes==0:         # unless specified (because Xs are trimmed and dont cover all nodes)
        numnodes=len(set(flatten_list(Xs)))

    converge = 0        # when converge >= tolerance, declare the graph converged.
    itern=0 # tmp variable for debugging

    # find a good starting graph using naive RW
    graph=noHidden(Xs,numnodes)
  
    bestgraph=np.copy(graph)       # store copy of best graph
    cur_graph=np.copy(graph)       # candidate graph for comparison

    best_ll=probX(Xs,bestgraph,numnodes,irts,jeff,beta)   # LL of best graph found
    cur_ll=best_ll                                        # LL of current graph

    # items in at least 2 lists. links between these nodes are more likely to affect P(G)
    # http://stackoverflow.com/q/2116286/353278
    overlap=reduce(set.union, (starmap(set.intersection, combinations(map(set, Xs), 2))))
    overlap=list(overlap)
    combos=list(combinations(overlap,2))    # all possible links btw overlapping nodes

    while converge < tolerance:
        
        links=[]        # links to toggle in candidate graph
        while True:     # emulates do-while loop (ugly)
            if random.random() <= prob_overlap:      # choose link from combos most of the time
                link=random.choice(combos)
            else:                                    # sometimes choose a link at random
                link=(0,0)
                while link[0]==link[1]:              # avoid self-transitions
                    link=(int(math.floor(random.random()*numnodes)),int(math.floor(random.random()*numnodes)))
            links.append(link)
            if random.random() <= prob_multi:
                break

        # toggle links
        for link in links:
            cur_graph[link[0],link[1]] = 1 - cur_graph[link[0],link[1]] 
            cur_graph[link[1],link[0]] = 1 - cur_graph[link[1],link[0]]

        graph_ll=probX(Xs,cur_graph,numnodes,irts,jeff,beta)

        # debugging... make sure its testing lots of graphs
        itern=itern+1
        if itern % 100 == 0:
            print itern

        # if graph is better than current graph, accept it
        # if graph is worse, accept with some probability
        if (graph_ll > cur_ll): # or (random.random() <= (math.exp(graph_ll)/math.exp(cur_ll))):
            print "GRAPH: ", graph_ll, "CUR: ", cur_ll, "BEST: ", best_ll
            graph=np.copy(cur_graph)
            cur_ll = graph_ll
            if cur_ll > best_ll:
                converge = 0          # reset convergence criterion only if new graph is better than BEST graph
                best_ll=cur_ll
                bestgraph = np.copy(cur_graph)
            else:
                converge += 1
        else:
            converge += 1
            # toggle links back. my guess is this is faster than making graph copies but i dont know
            for link in links:
                cur_graph[link[0],link[1]] = 1 - cur_graph[link[0],link[1]]
                cur_graph[link[1],link[0]] = 1 - cur_graph[link[1],link[0]] 
    return bestgraph, best_ll

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
def genG(n,k,p,tries=1000, seed=None):
    g=nx.connected_watts_strogatz_graph(n,k,p,tries,seed) # networkx graph
    a=np.array(nx.adjacency_matrix(g).todense())     # adjacency matrix
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
        if random.random() < theta:
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
def probX(Xs, a, numnodes, irts=[], jeff=0.5, beta=1, maxlen=20):
    probs=[]
    t=a/sum(a.astype(float))            # transition matrix (from: column, to: row)
    
    for xnum, x in enumerate(Xs):
        prob=[]
        
        notinx=[]       # nodes not in trimmed X
        for i in range(numnodes):
            if i not in x:
                notinx.append(i)
        
        for curpos in range(1,len(x)):
            startindex=x[curpos-1]
            deletedlist=sorted(x[curpos:]+notinx,reverse=True)
            notdeleted=[i for i in range(numnodes) if i not in deletedlist]
            Q=np.delete(t,deletedlist,0) # make copy of t and delete rows
            Q=np.delete(Q,deletedlist,1) # delete columns
                
            if (len(irts) > 0) and (jeff < 1): # use this method only when passing IRTs with weight < 1
                startindex = startindex-sum([startindex > i for i in deletedlist])
                numcols=np.shape(Q)[1]
                flist=[]
                oldQ=np.copy(Q)
                Q=np.identity(len(oldQ)) # init to Q^0, for when r=1
                irt=irts[xnum][curpos-1]

                for r in range(1,maxlen):
                    sumlist=[]
                    for k in range(numcols):
                        num1=Q[k,startindex]                # probability of being at node k in r-1 steps
                        num2=t[x[curpos],notdeleted[k]]     # probability transitioning from k to absorbing node    
                        sumlist.append(num1*num2)
                    innersum=sum(sumlist)                   # sum over all possible paths
                    
                    # much faster than using scipy.stats.gamma.pdf
                    log_gamma=r*math.log(beta)-math.lgamma(r)+(r-1)*math.log(irt)-beta*irt # r=alpha. probability of observing irt at r steps
                    
                    if innersum > 0: # sometimes it's not possible to get to the target node in r steps
                        flist.append(log_gamma*(1-jeff)+jeff*math.log(innersum))
                    Q=np.dot(Q,oldQ)    # raise the power by one
                
                f=sum([math.e**i for i in flist])
                prob.append(f)           # probability of x_(t-1) to X_t
            else:                        # if no IRTs, use standard INVITE
                I=np.identity(len(Q))
                reg=(1+1e-5)             # nuisance parameter to prevent errors
                N=inv(I*reg-Q)
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
            #print "Warning: Zero-probability transition; graph cannot produce X"
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
    random.seed(seed)
    if start is None:
        start=random.choice(nx.nodes(g))
    walk=[]
    unused_nodes=set(nx.nodes(g))
    unused_nodes.remove(start)
    while len(unused_nodes) > 0:
        p=start
        start=random.choice([x for x in nx.all_neighbors(g,start)]) # follow random edge
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

# generates fake IRTs from # of steps in a random walk, using gamma distribution
def stepsToIRT(irts, beta=1.0, seed=None):
    np.random.seed(seed)        # to generate the same IRTs each time
    new_irts=[]
    for irtlist in irts:
        newlist=[np.random.gamma(irt, (1.0/beta)) for irt in irtlist]  # beta is rate, but random.gamma uses scale (1/rate)
        new_irts.append(newlist)
    return new_irts

# runs a batch of toy graphs. logging code needs to be cleaned up significantly
def toyBatch(numgraphs, numnodes, numlinks, probRewire, numx, trim, jeff, beta, outfile, start_seed=0, 
             methods=['rw','invite','inviteirt','fe'],header=1):

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
        f.write(',')
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
        [Xs,steps]=zip(*[genX(g, seed=x_seed+i,use_irts=1) for i in range(numx)])
        Xs=list(Xs)
        steps=list(steps)
        [Xs,alter_graph]=trimX(trim,Xs,g)

        if alter_graph==0:      # only use data that covers entire graph
            for method in methods:
                print "SEED: ", seed_param, "method: ", method
 
                # generate IRTs if using IRT model
                irts=[]
                if method=='inviteirt':
                    irts=stepsToIRT(steps, beta, seed=x_seed)
                
                # Find best graph! (and log time)
                starttime=datetime.now()
                if method in ['invite','inviteirt']:
                    bestgraph, bestval=findBestGraph(Xs, irts, jeff, beta, numnodes)
                if method == 'rw':
                    bestgraph=noHidden(Xs,numnodes)
                    bestval=probX(Xs, bestgraph, numnodes)
                if method == 'fe':
                    bestgraph=firstEdge(Xs,numnodes)
                    bestval=probX(Xs, bestgraph, numnodes)
                elapsedtime=str(datetime.now()-starttime)
        
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
def trimX(prop, Xs, g):
    numnodes=g.number_of_nodes()
    alter_graph_size=0              # report if graph size changes-- may result in disconnected graph!
    numtrim=int(round(numnodes*prop))
    Xs=[i[0:numtrim] for i in Xs]
    for i in range(numnodes):
        if i not in set(flatten_list(Xs)):
            alter_graph_size=1
    #        g.remove_node(i)
    #a=np.array(nx.adjacency_matrix(g, range(numnodes)).todense())
    #if 0 not in sum(a):         # ensures that graph is still connected after trimming Xs
    #    alter_graph_size=0
    if alter_graph_size==1:
        print "WARNING: Not all graph nodes encountered after trimming X"
    return Xs, alter_graph_size

# tuple walk from flat list
def walk_from_path(path):
    walk=[]
    for i in range(len(path)-1):
        walk.append((path[i],path[i+1])) 
    return walk
