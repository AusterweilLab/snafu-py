from . import *
from functools import reduce

# TODO: when doing same phase twice in a row, don't re-try same failures
    # (pass dict of failures, don't try if numchanges==0)
# TODO: Implement GOTM/ECN from Goni et al. 2011

# alias for backwards compatibility
def communitynetwork(*args, **kwargs):
    return conceptualNetwork(*args, **kwargs)

# alias for backwards compatibility
def priorToGraph(*args, **kwargs):
    return priorToNetwork(*args, **kwargs) 

# mix U-INVITE with random jumping model
def addJumps(probs, td, numnodes=None, statdist=None, Xs=None):
    if (td.jumptype == "uniform") and (numnodes == None):
        raise ValueError("Must specify 'numnodes' when jumptype is uniform [addJumps]")
    if (td.jumptype == "stationary") and (np.any(statdist == None) or (Xs == None)):
        raise ValueError("Must specify 'statdist' and 'Xs' when jumptype is stationary [addJumps]")

    if td.jumptype == "uniform":
        jumpprob = float(td.jump)/numnodes                     # uniform jumping
    
    for l in range(len(probs)):                                # loop through all lists (l)
        for inum, i in enumerate(probs[l][1:]):                # loop through all items (i) excluding first (don't jump to item 1)
            if td.jumptype=="stationary":
                jumpprob = statdist[Xs[l][inum+1]]             # stationary probability jumping
                if np.isnan(jumpprob):                         # if node is disconnected, jumpprob is nan
                    jumpprob = 0.0
            probs[l][inum+1] = jumpprob + (1-td.jump)*i        # else normalize existing probability and add jumping probability
    return probs

# mix U-INVITE with priming model
# code is confusing...
def adjustPriming(probs, td, Xs):
    for xnum, x in enumerate(Xs[1:]):         # check all items starting with 2nd list
        for inum, i in enumerate(x[:-1]):     # except last item
            if i in Xs[xnum][:-1]:            # is item in previous list? if so, prime next item
                # follow prime with calc_prob_adjacent td.priming, follow RW with calc_prob_adjacent (1-td.priming)
                idx=Xs[xnum].index(i) # index of item in previous list
                if Xs[xnum][idx+1]==Xs[xnum+1][inum+1]:
                    probs[xnum+1][inum+1] = (probs[xnum+1][inum+1] * (1-td.priming)) + td.priming
                else:
                    probs[xnum+1][inum+1] = (probs[xnum+1][inum+1] * (1-td.priming))
    return probs

# Chan, Butters, Paulsen, Salmon, Swenson, & Maloney (1993) jcogneuro (p. 256) + pathfinder (PF; Schvaneveldt)
# + Chan, Butters, Salmon, Johnson, Paulsen, & Swenson (1995) neuropsychology
# Returns only PF(q, r) = PF(n-1, inf) = minimum spanning tree (sparsest possible graph)
# other parameterizations of PF(q, r) not implemented
def pathfinder(Xs, numnodes=None, valid=False, td=None):
    if numnodes == None:
        numnodes = len(set(flatten_list(Xs)))
    
    # From https://github.com/evanmiltenburg/dm-graphs
    def MST_pathfinder(G):
        """The MST-pathfinder algorithm (Quirin et al. 2008) reduces the graph to the
        unions of all minimal spanning trees."""
        NG    = nx.Graph()
        NG.add_nodes_from(list(range(numnodes)))
        edges = sorted( ((G[a][b]['weight'],a,b) for a,b in G.edges()),
                            reverse=False)                                                # smaller distances are more similar
        clusters = {node:i for i,node in enumerate(G.nodes())}
        while not edges == []:
            w1,a,b = edges[0]
            l      = []
            # Select edges to be considered this round:
            for w2,u,v in edges:
                if w1 == w2:
                    l.append((u,v,w2))
                else:
                    break
            # Remaining edges are those not being considered this round:
            edges = edges[len(l):]
            # Only select those edges for which the items are not in the same cluster
            l = [(a,b,c) for a,b,c in l if not clusters[a]==clusters[b]]
            # Add these edges to the graph:
            NG.add_weighted_edges_from(l)
            # Merge the clusters:
            for a,b,w in l:
                cluster_1 = clusters[a]
                cluster_2 = clusters[b]
                clusters = {node:cluster_1 if i==cluster_2 else i
                            for node,i in clusters.items()}
        return NG

    if valid and not td:
        raise ValueError('Need to pass DataModel when generating \'valid\' pathfinder()')
        
    N = float(len(Xs))
    distance_mat = np.zeros((numnodes, numnodes))
    for item1 in range(numnodes):
        for item2 in range(item1+1,numnodes):
            Tij = 0
            dijk = 0
            for x in Xs:
                if (item1 in x) and (item2 in x):
                    Tij += 1
                    dijk = dijk + (abs(x.index(item1) - x.index(item2)) / float(len(x)))
            try:
                dij = dijk * (N / (Tij**2))
            except:
                dij = 0.0      # added constraint for divide-by-zero... this will ensure that no edge will exist between i and j
            distance_mat[item1, item2] = dij
            distance_mat[item2, item1] = dij

    #graph = scipy.sparse.csgraph.minimum_spanning_tree(distance_mat)
    graph = nx.to_numpy_array(MST_pathfinder(nx.Graph(distance_mat)))

    # binarize and make graph symmetric (undirected)... some redundancy but it's cheap
    #graph = np.where(graph.todense(), 1, 0)
    graph = np.array(np.where(graph, 1, 0))
    for rownum, row in enumerate(graph):
        for colnum, val in enumerate(row):
            if val==1:
                graph[rownum,colnum]=1
                graph[colnum,rownum]=1

    if valid:
        graph = makeValid(Xs, graph, td)
        
    #return np.array(graph).astype(int)
    return graph

# objective graph cost
# returns the number of links that need to be added or removed to reach the true graph
def cost(graph,a, undirected=True):
    cost = sum(sum(np.array(abs(graph-a))))
    if undirected:
        return cost/2.0
    else:
        return cost

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

def evalGraphPrior(a, prior, undirected=True):
    probs = []
    priordict = prior[0]
    items = prior[1]
    nullprob = priordict['DEFAULTPRIOR']

    for inum, i in enumerate(a):
        for jnum, j in enumerate(i):
            if (inum > jnum) or ((undirected==False) and (inum != jnum)):
                if undirected:
                    pair = np.sort((items[inum], items[jnum]))
                else:
                    pair = (items[inum], items[jnum])
                try:
                    priorprob = priordict[pair[0]][pair[1]]
                    if j==1:
                        prob = priorprob
                    elif j==0:
                        prob = (1-priorprob)
                except:
                    prob = nullprob  #  no information about edge
                probs.append(prob)
    
    probs = [np.log(prob) for prob in probs]      # multiplication probably results in underflow...
    probs = sum(probs)
    return probs

def firstEdge(Xs, numnodes=None):
    if numnodes == None:
        numnodes = len(set(flatten_list(Xs)))
    a=np.zeros((numnodes,numnodes))
    for x in Xs:
        a[x[0],x[1]]=1
        a[x[1],x[0]]=1 # symmetry
    a=np.array(a.astype(int))
    return a

def fullyConnected(numnodes):
    a=np.ones((numnodes,numnodes))
    for i in range(numnodes):
        a[i,i]=0.0
    return a.astype(int)

# only returns adjacency matrix, not nx graph
def naiveRandomWalk(Xs, numnodes=None, directed=False):
    if numnodes == None:
        numnodes = len(set(flatten_list(Xs)))
    a=np.zeros((numnodes,numnodes))
    for x in Xs:
        walk = edges_from_nodes(x)
        for i in set(walk):
            a[i[0],i[1]]=1
            if directed==False:
                a[i[1],i[0]]=1
    a=np.array(a.astype(int))
    return a

def genGraphPrior(graphs, items, fitinfo=Fitinfo({}), mincount=1, undirected=True, returncounts=False):
    a_start = fitinfo.prior_a
    b_start = fitinfo.prior_b
    method = fitinfo.prior_method
    p = fitinfo.zibb_p
    
    priordict={}
  
    #def betabinomial(a,b):
    #    return (b / (a + b))
    
    def zeroinflatedbetabinomial(a,b,p):
        return (b / ((1-p)*a+b))

    # tabulate number of times edge does or doesn't appear in all of the graphs when node pair is present
    for graphnum, graph in enumerate(graphs):   # for each graph
        itemdict=items[graphnum]
        for inum, i in enumerate(graph):     # rows of graph
            for jnum, j in enumerate(i):     # columns of graph
                if (inum > jnum) or ((undirected==False) and (inum != jnum)):
                    item1 = itemdict[inum]
                    item2 = itemdict[jnum]
                    if undirected:
                        pair = np.sort((item1,item2))
                    else:
                        pair = (item1,item2)
                    if pair[0] not in list(priordict.keys()):
                        priordict[pair[0]]={}
                    if pair[1] not in list(priordict[pair[0]].keys()):
                        priordict[pair[0]][pair[1]] = [a_start, b_start]
                    if j==1:
                        priordict[pair[0]][pair[1]][1] += 1.0 # increment b counts
                    elif j==0:
                        priordict[pair[0]][pair[1]][0] += 1.0 # increment a counts
   
    if not returncounts:
        for item1 in priordict:
            for item2 in priordict[item1]:
                a, b = priordict[item1][item2]      # a=number of participants without link, b=number of participants with link
                if (a+b) >= (mincount + a_start + b_start):
                    #if method == "zeroinflatedbetabinomial":
                    priordict[item1][item2] = zeroinflatedbetabinomial(a,b,p) # zero-inflated beta-binomial
                    #elif method == "betabinomial":
                    #    priordict[item1][item2] = betabinomial(a,b) # beta-binomial
                else:
                    priordict[item1][item2] = 0.0   # if number of observations is less than mincount, make edge prior 0.0
        if 'DEFAULTPRIOR' in list(priordict.keys()):
            raise ValueError('Sorry, you can\'t have a node called DEFAULTPRIOR. \
                              Sure, I should have coded this better, but I really didn\'t think this situation would ever occur.')
        else:
            #if method == "zeroinflatedbetabinomial":
            priordict['DEFAULTPRIOR'] = zeroinflatedbetabinomial(a_start, b_start, p)
            #elif method=="betabinomial":
            #    priordict['DEFAULTPRIOR'] = betabinomial(a_start, b_start)
    
    return priordict

# generate starting graph for U-INVITE
def genStartGraph(Xs, numnodes, td, fitinfo):
    if fitinfo.startGraph=="cn_valid":
        graph = conceptualNetwork(Xs, numnodes, td=td, valid=True, fitinfo=fitinfo)
    elif fitinfo.startGraph=="pf_valid":
        graph = pathfinder(Xs, numnodes, valid=True, td=td)
    elif (fitinfo.startGraph=="rw" or fitinfo.startGraph=="nrw"):
        graph = naiveRandomWalk(Xs,numnodes)
    elif fitinfo.startGraph=="fully_connected":
        graph = fullyConnected(numnodes)
    elif fitinfo.startGraph=="empty_graph":
        graph = np.zeros((numnodes,numnodes)).astype(int)           # useless...
    else:
        graph = np.copy(fitinfo.startGraph)                         # assume a graph has been passed as a starting point
    return graph

# deprecated alias for backwards compatibility
def communitynetwork(*args, **kwargs):
    return conceptualNetwork(*args, **kwargs)

# w = window size; two items appear within +/- w steps of each other (where w=1 means adjacent items)
# f = filter frequency; if two items don't fall within the same window more than f times, then no edge is inferred
# c = confidence interval; retain the edge if there is a <= c probability that two items occur within the same window n times by chance alone
# valid (t/f) ensures that graph can produce data using censored RW.
def conceptualNetwork(Xs, numnodes=None, fitinfo=Fitinfo({}), valid=False, td=None):
    if numnodes == None:
        numnodes = len(set(flatten_list(Xs)))
        
    w = fitinfo.cn_windowsize
    f = fitinfo.cn_threshold
    c = fitinfo.cn_alpha
    
    if f<1:                 # if <1 treat as proportion of total lists; if >1 treat as absolute # of lists
        f=int(round(len(Xs)*f))

    if valid and not td:
        raise ValueError('Need to pass Data when generating \'valid\' goni()')

    #if c<1:
    #    from statsmodels.stats.proportion import proportion_confint as pci

    if w < 1:
        print("Error in goni(): w must be >= 1")
        return

    graph=np.zeros((numnodes, numnodes)).astype(int)         # empty graph

    # frequency of co-occurrences within window (w)
    for x in Xs:                                             # for each list
        cooccur_within_list = []                             # only count one cooccurence per list (for binomial test)
        for pos in range(len(x)):                            # for each item in list
            for i in range(1, w+1):                          # for each window size
                if pos+i<len(x):
                    if (x[pos], x[pos+i]) not in cooccur_within_list:
                        graph[x[pos],x[pos+i]] += 1
                        graph[x[pos+i],x[pos]] += 1
                        cooccur_within_list.append((x[pos], x[pos+i]))
                        cooccur_within_list.append((x[pos+i], x[pos]))

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
        listofedges=list(zip(*np.nonzero(graph)))                           # list of edges in graph to check
        numlists=float(len(Xs))
        meanlistlength=np.mean([len(x) for x in Xs])
    
        # Goni et al. (2011), eq. 10
        p_adj = (2.0/(meanlistlength*(meanlistlength-1))) * ((w*meanlistlength) - ((w*(w+1))/2.0))
        for i,j in listofedges:
            p_linked = (xfreq[i]/numlists) * (xfreq[j]/numlists) * p_adj
            #ci=pci(cooccur[i,j],numlists,alpha=c,method="beta")[0]     # lower bound of Clopper-Pearson binomial CI
            ci = pci_lowerbound(cooccur[i,j], numlists, c)              # lower bound of Clopper-Pearson binomial CI
            if (p_linked >= ci):                                        # if co-occurrence could be due to chance, remove edge
                graph[i,j]=0
                graph[j,i]=0

    if valid:
        graph = makeValid(Xs, graph, td)

    # make sure there are no self-transitions -- is this necessary?
    for i in range(len(graph)):
        graph[i,i]=0.0

    return graph

# enrich graph by finding modules and making them completely interconnected
# using Generalized Topological Overlap Measure (GTOM)
# right now only does n=2 (neighbors and neighbors of neighbors)
# see Goni et al 2010
# TODO unfinished: so far, creates GTOM matrix but doesn't "enrich" network... how to determine # of clusters?
def gtom(graph):

    # modified from uinvite(), copied for convenience (TODO consolidate by moving outside to its own function)
    # return list of neighbors of neighbors of i, that aren't themselves neighbors of i
    # i.e., an edge between i and any item in nn forms a triangle
    def neighborsofneighbors(i, nxg):
        nn=[]                                       # neighbors of neighbors (nn)
        n=list(nx.all_neighbors(nxg,i))
        for j in n:
            nn=nn+list(nx.all_neighbors(nxg,j))
        nn=list(set(nn))
        if i in nn:
            nn.remove(i)                            # remove self
        return nn
    
    nxgraph = nx.to_networkx_graph(graph)
    numnodes = nx.number_of_nodes(nxgraph)
    gtom_mat = np.zeros((numnodes,numnodes))
   
    nn_dict = {}
    for i in range(numnodes):
        nn_dict[i] = neighborsofneighbors(i, nxgraph)
    
    for i in range(numnodes):
        for j in range(i+1,numnodes):
            i_neighbors = nn_dict[i]
            j_neighbors = nn_dict[j]
            min_neighbors = min(len(i_neighbors),len(j_neighbors))
            len_overlap = len(set.intersection(set(i_neighbors),set(j_neighbors)))
            gtom_mat[i, j] = 1 - (float(len_overlap) / min_neighbors)
            gtom_mat[j, i] = gtom_mat[i, j]

    return gtom_mat

 
def hierarchicalUinvite(Xs, items, numnodes=None, td=DataModel({}), irts=False, fitinfo=Fitinfo({}), seed=None, debug=True):

    if numnodes == None:
        numnodes = [len(set(flatten_list(x))) for x in Xs]
    
    nplocal=np.random.RandomState(seed) 
    fitinfoSG = fitinfo.startGraph  # fitinfo is mutable, need to revert at end of function... blah
    # create ids for all subjects
    subs=list(range(len(Xs)))
    graphs=[[]]*len(subs)

    # cycle though participants
    exclude_subs=[]
    graphchanges=1
    rnd=1
    while graphchanges > 0:
        if debug: print("Round: ", rnd)
        graphchanges = 0
        nplocal.shuffle(subs)
        for sub in [i for i in subs if i not in exclude_subs]:
            if debug: print("SS: ", sub)
            
            if graphs[sub] == []:
                fitinfo.startGraph = fitinfoSG      # on first pass for subject, use default fitting method (e.g., NRW, goni, etc)
            else:
                fitinfo.startGraph = graphs[sub]    # on subsequent passes, use ss graph from previous iteration

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
                exclude_subs=[sub]              # if a single change, fit everyone again (except the graph that was just fit)
            else:
                exclude_subs.append(sub)        # if graph didn't change, don't fit them again until another change
        rnd += 1

    ## generate group graph
    priordict = genGraphPrior(graphs, items, fitinfo=fitinfo)
    fitinfo.startGraph = fitinfoSG  # reset fitinfo starting graph to default
    
    return graphs, priordict

def probXhierarchical(Xs, graphs, items, td, priordict=None, irts=Irts({})):
    lls=[]
    for sub in range(len(Xs)):
        if priordict:
            prior = (priordict, items[sub])
        else:
            prior=None
        best_ll, probmat = probX(Xs[sub], graphs[sub], td, irts=irts, prior=prior)   # LL of graph
        lls.append(best_ll)
    ll=sum(lls)
    return ll

# construct graph using method using item correlation matrix and planar maximally filtered graph (PMFG)
# see Borodkin, Kenett, Faust, & Mashal (2016) and Kenett, Kenett, Ben-Jacob, & Faust (2011)
# does not work well for small number of lists! many NaN correlations + when two correlations are equal, ordering is arbitrary
def correlationBasedNetwork(Xs, numnodes=None, minlists=0, valid=False, td=None):
    if valid and not td:
        raise ValueError('Need to pass Data when generating \'valid\' correlationBasedNetwork()')
 
    try:
        import planarity
    except ImportError:
        raise ImportError('Python package planarity is not included by default in SNAFU. Please install it separately from your terminal: pip install planarity')

    
    if numnodes == None:
        numnodes = len(set(flatten_list(Xs)))
 
    # construct matrix of list x item where each cell indicates whether that item is in that list
    list_by_item = np.zeros((numnodes,len(Xs)))
    for node in range(numnodes):
        for x in range(len(Xs)):
            if node in Xs[x]:
                list_by_item[node,x]=1.0
    
    # find pearsonr correlation for all item pairs
    item_by_item = {}
    for item1 in range(numnodes):
        for item2 in range(item1+1,numnodes):
            if (sum(list_by_item[item1]) <= minlists) or (sum(list_by_item[item2]) <= minlists):    # if a word only appears in <= minlists lists, exclude from correlation list (kenett personal communication, to avoid spurious correlations)
                item_by_item[(item1, item2)] = np.nan
            else:
                #item_by_item[(item1, item2)] = scipy.stats.pearsonr(list_by_item[item1],list_by_item[item2])[0]
                item_by_item[(item1, item2)] = pearsonr(list_by_item[item1],list_by_item[item2])
    
    corr_vals = sorted(item_by_item, key=item_by_item.get)[::-1]       # keys in correlation dictionary sorted by value (high to low, including NaN first)

    edgelist=[]
    for pair in corr_vals:
        if not np.isnan(item_by_item[pair]):    # nan correlation occurs when item is in all lists-- exclude from graph (conservative)
            edgelist.append(pair)
            if not planarity.is_planar(edgelist):
                edgelist.pop()
    
    g = nx.Graph()
    g.add_nodes_from(list(range(numnodes)))
    g.add_edges_from(edgelist)
    a=nx.to_numpy_array(g).astype(int)
   
    if valid:
        a = makeValid(Xs, a, td)
   
    return a

def makeValid(Xs, graph, td, seed=None):
    # add direct edges when transition is impossible
    check=probX(Xs, graph, td)
    while check[0] == -np.inf:
        if isinstance(check[1],tuple):
            graph[check[1][0], check[1][1]] = 1
            graph[check[1][1], check[1][0]] = 1
        # i think these 2 lines are no longer necessary
        #elif check[1] == "prior":
        #    raise ValueError('Starting graph has prior probability of 0.0')
        elif isinstance(check[1],int):
            # when list contains one item and node is unreachable, connect to random node
            nplocal = np.random.RandomState(seed)
            randnode = nplocal.choice(range(len(graph)))
            graph[check[1], randnode] = 1
            graph[randnode, check[1]] = 1
        else:
            raise ValueError('Unexpected error from makeValid()')
        check=probX(Xs, graph, td)
    return graph

# converts priordict to graph if probability of edge is greater than cutoff value
def priorToNetwork(priordict, items, cutoff=0.5, undirected=True):
    numnodes = len(items)
    a = np.zeros((numnodes, numnodes))
    
    for item1 in list(priordict.keys()):
        if item1 != 'DEFAULTPRIOR':
            for item2 in priordict[item1]:
                if priordict[item1][item2] > cutoff:
                    item1_idx = list(items.keys())[list(items.values()).index(item1)]    # syntax is a little convoluted in case dictionary keys are not in sequential order
                    item2_idx = list(items.keys())[list(items.values()).index(item2)]
                    a[item1_idx, item2_idx] = 1.0
                    if undirected:
                        a[item2_idx, item1_idx] = 1.0
    return a

# probability of observing Xs, including irts and prior
#@profile
#@nogc
def probX(Xs, a, td, irts=Irts({}), prior=None, origmat=None, changed=[]):
    try:
        numnodes=len(a)
    except TypeError:
        raise Exception(a)
    reg=(1+1e-10)                           # nuisance parameter to prevent errors; can also use pinv instead of inv, but that's much slower
    identmat=np.identity(numnodes) * reg    # pre-compute for tiny speed-up (only for non-IRT)

    probs=[]

    # generate transition matrix (from: column, to: row) from link matrix
    t=a/sum(a.astype(float))
    t=np.nan_to_num(t)                  # jumping/priming models can have nan in matrix, need to change to 0
    
    if (td.jumptype=="stationary") or (td.start_node=="stationary"):
        statdist=stationary(t)

    # U-INVITE probability excluding jumps, prior, and priming adjustments -- those come later
    for xnum, x in enumerate(Xs):
        x2=np.array(x)
        t2=t[x2[:,None],x2]                                        # re-arrange transition matrix to be in list order
        prob=[]
        if td.start_node=="stationary":
            prob.append(statdist[x[0]])                            # probability of X_1
        elif td.start_node=="uniform":
            prob.append(1.0/numnodes)

        # if impossible starting point, return immediately
        if (prob[-1]==0.0):
            try:
                return -np.inf, (x[0], x[1])
            except:
                return -np.inf, x[0]

        if (len(changed) > 0) and isinstance(origmat,list):        # if updating prob. matrix based on specific link changes
            update=0                                               # reset for each list

        # flag if list contains perseverations
        if len(x) == len(set(x)):
            list_has_perseverations = False
        else:
            list_has_perseverations = True

        for curpos in range(1,len(x)):
            if (len(changed) > 0) and isinstance(origmat,list):
                if update==0:                                      # first check if probability needs to be updated
                    if (Xs[xnum][curpos-1] in changed):            # (only AFTER first changed node has been reached)
                        update=1
                    else:                                          # if not, take probability from old matrix
                        prob.append(origmat[xnum][curpos])
                        continue
            
            if list_has_perseverations:            # a bit slower because matrix is being copied
                x2=np.array([i for i,j in enumerate(x) if (j not in x[:i]) and (i < curpos)]) # column ids for transient states excluding perseverations
                Q=t2[x2[:,None],x2]                # excludes perseverations. could be sped if only performed when Q contains perseverations
                                                   # as opposed to being done for every transition if a perseveration is in the list
            else:                                  
                Q=t2[:curpos,:curpos]              # old way when data does not include perseverations
                
            # td.censor_fault is necessary to model perservations in the data
            if td.censor_fault > 0.0:
                Q=np.multiply(Q, 1.0-td.censor_fault)
            
            if len(irts.data) > 0:     # use this method only when passing IRTs
                numcols=len(Q)
                flist=[]
                newQ=np.zeros(numcols)                             # init to Q^0, for when r=1
                newQ[curpos-1]=1.0                                 # (using only one: row for efficiency)

                irt=irts.data[xnum][curpos-1]

                # precompute for small speedup
                if irts.irttype=="gamma":
                    logbeta=np.log(irts.gamma_beta)
                    logirt=np.log(irt)

                # normalize irt probabilities to avoid irt weighting
                if irts.irttype=="gamma":
                     # r=alpha. probability of observing irt at r steps
                    irtdist=[r*logbeta-math.lgamma(r)+(r-1)*logirt-irts.gamma_beta*irt for r in range(1,irts.rcutoff)]
                if irts.irttype=="exgauss":
                
                    irtdist=[np.log(irts.exgauss_lambda/2.0)+(irts.exgauss_lambda/2.0)*(2.0*r+irts.exgauss_lambda*(irts.exgauss_sigma**2)-2*irt)+np.log(math.erfc((r+irts.exgauss_lambda*(irts.exgauss_sigma**2)-irt)/(np.sqrt(2)*irts.exgauss_sigma))) for r in range(1,irts.rcutoff)]

                for r in range(1,irts.rcutoff):
                    innersum=0
                    for k in range(numcols):
                        num1=newQ[k]                               # probability of being at node k in r-1 steps
                        num2=t2[curpos,k]                          # probability transitioning from k to absorbing node    
                        innersum=innersum+(num1*num2)

                    # compute irt probability given r steps
                    log_dist = irtdist[r-1] / sum(irtdist)

                    if innersum > 0: # sometimes it's not possible to get to the target node in r steps
                        flist.append(log_dist + np.log(innersum))

                    newQ=np.inner(newQ,Q)                          # raise power by one

                f=sum([np.e**i for i in flist])
                prob.append(f)                                     # probability of x_(t-1) to X_t
            else:                                                  # if no IRTs, use standard U-INVITE
                I=identmat[:len(Q),:len(Q)]
                
                # novel items are emitted with probability 1 when encountered. perseverations are emitted with probability td.censor_fault when encountered.
                if list_has_perseverations:              # if list has perseverations. could speed up by only doing this step when a perseveration has been encountered
                    x1=np.array([curpos])   # absorbing node
                    #x2=np.array([i for i,j in enumerate(x) if (j not in x[:i]) and (i < curpos)]) # column ids for transient states excluding perseverations
                    x2=np.array([i for i,j in enumerate(x) if (j not in x[i+1:curpos]) and (i < curpos)]) # column ids for transient states excluding perseverations
                    R=t2[x1[:,None],x2][0]  # why is [0] necessary here but not in the else case?
                    
                    if Xs[xnum][curpos] in Xs[xnum][:curpos]:       # if absorbing state has appeared in list before...
                        R=np.multiply(R,td.censor_fault)
                else:                                       # if not a perseveration
                    R=t2[curpos,:curpos]                    # old way
               
                ### test (when censor_fault=0) to see if absorbing distribution sums to 1... something is broken
                #total = []
                #x2=np.array([j for i,j in enumerate(x) if (i < curpos)]) # column ids for transient states excluding perseverations
                #N=np.linalg.solve(I-Q,I[-1])
                #for i in range(len(t)):
                #    R=t[np.array([i])[:,None],x2]
                #    B=np.dot(R,N)
                #    total.append(B[0])
                #    if B[0] > 1.0:
                #        print("NONONO")
                #print("total ", total)
                #R=t2[curpos,:curpos]                    # old way to reset
                ###
                
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
            if (prob[-1]==0.0) and (td.jump == 0.0) and (td.priming == 0.0):
                return -np.inf, (x[curpos-1], x[curpos])

        probs.append(prob)

    uinvite_probs = copy.deepcopy(probs)      # store only u-invite transition probabilities (the computationally hard stuff) to avoid recomputing
    
    # adjust for jumping probability
    if td.jump > 0.0:
        if td.jumptype=="uniform":
            probs=addJumps(probs, td, numnodes=numnodes)
        elif td.jumptype=="stationary":
            probs=addJumps(probs, td, statdist=statdist, Xs=Xs)

    if (td.priming > 0.0):
        probs=adjustPriming(probs, td, Xs)

    # check for impossible transitions after priming and jumping
    for xnum, x in enumerate(probs):
        for inum, i in enumerate(x):
            if (i==0.0) and (inum==0):
                return -np.inf, (Xs[xnum][inum], Xs[xnum][inum+1])  # link to next item when first item is unreachable
            elif (i==0.0) and (inum > 0):
                return -np.inf, (Xs[xnum][inum-1], Xs[xnum][inum])  # link to previous item otherwise
                
    try:
        ll=sum([sum([np.log(j) for j in probs[i]]) for i in range(len(probs))])
    except:
        ll=-np.inf

    # include prior?
    if prior:
        priorlogprob = evalGraphPrior(a, prior)
        ll = ll + priorlogprob

    return ll, uinvite_probs

#@profile
def uinvite(Xs, td=DataModel({}), numnodes=None, irts=Irts({}), fitinfo=Fitinfo({}), prior=None, debug=True, seed=None):
    nplocal=np.random.RandomState(seed) 

    if numnodes == None:
        numnodes = len(set(flatten_list(Xs)))

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
        numchanges=0     # number of changes in single pivot() call

        if (best_ll == None) or (np.any(probmat == None)):
            best_ll, probmat = probX(Xs,graph,td,irts=irts,prior=prior)   # LL of best graph found
        nxg=nx.to_networkx_graph(graph)

        # generate dict where v[i] is a list of nodes where (i, v[i]) is an existing edge in the graph
        if (method=="prune") or (method==0):
            if debug:
                print("Pruning", str(vmaj) + "." + str(vmin), "... ",) # (len(edges)/2)-len(firstedges), "possible:",
            sys.stdout.flush()
            listofedges=np.where(graph==1)
            v=dict()
            for i in range(numnodes):
                v[i]=[]
            for i in zip(*listofedges):
                if ((i[0], i[1]) not in firstedges) and ((i[1], i[0]) not in firstedges): # don't flip first edges (FE)!
                    if td.jump == 0.0:                                                      # unless jumping is allowed, untested 10/6/17 JCZ
                        v[i[0]].append(i[1])
        
        # generate dict where v[i] is a list of nodes where (i, v[i]) would form a new triangle
        if (method=="triangles") or (method==1):
            if debug:
                print("Adding triangles", str(vmaj) + "." + str(vmin), "... ",) # (len(edges)/2), "possible:",
            sys.stdout.flush()
            nn=dict()
            for i in range(len(graph)):
                nn[i]=neighborsofneighbors(i, nxg)
            v=nn
        
        # generate dict where v[i] is a list of nodes where (i, v[i]) is NOT an existing an edge and does NOT form a triangle
        if (method=="nonneighbors") or (method==2):
            # list of a node's non-neighbors (non-edges) that don't form triangles
            if debug:
                print("Adding other edges", str(vmaj) + "." + str(vmin), "... ",)
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
                    graph=swapEdges(graph,[edge])
                else:
                    best_ll = graph_ll
                    probmat = newprobmat
                    numchanges += 1
                    loopcount = 0
                    # probX under all possible perseveration values JCZ 5/9/2018
                    if fitinfo.estimatePerseveration:
                        old_censor = td.censor_fault
                        best_param = old_censor
                        for censor_param in [i/100.0 for i in range(101)]:
                            td.censor_fault = censor_param
                            graph_ll, newprobmat = probX(Xs,graph,td,irts=irts,prior=prior)   # LL of starting graph
                            if graph_ll > best_ll:
                                best_ll = graph_ll
                                probmat = newprobmat
                                best_param = censor_param
                        td.censor_fault = best_param
                        if debug:
                            print("censor_fault old:", old_censor, " censor_fault new: ", best_param)

                v[node1].remove(node2)   # remove edge from possible choices
                if not fitinfo.directed:
                    v[node2].remove(node1)
           
                # increment even if graph prob = -np.inf for implicit penalty
                count[node1] += 1
                count[node2] += 1
                if (graph_ll != -np.inf) and (fitinfo.followtype != "random"):
                    if avg[node1] == -np.inf:
                        avg[node1] = graph_ll
                    else: # followtype == avg
                            avg[node1] = avg[node1] * ((count[node1]-1)/count[node1]) + (1.0/count[node1]) * graph_ll
                    if avg[node2] == -np.inf:
                        avg[node2] = graph_ll
                    else:  # followtype == avg
                        avg[node2] = avg[node2] * ((count[node2]-1)/count[node2]) + (1.0/count[node2]) * graph_ll
            else:                       # no edges on this node left to try!
                avg[node1]=-np.inf      # so we don't try it again...
                finishednodes += 1

        if debug:
            print(numchanges, "changes")

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

    # check if data has perseverations
    if not [len(set(i)) for i in Xs]==[len(i) for i in Xs]:
        if (td.censor_fault == 0.0) and (not fitinfo.estimatePerseveration):
            raise Exception('''Your data contains perseverations, but \
                            censor_fault = 0.0; Set to some value 0.0 < x <= 1.0 or
                            set estimatePerseveration to True''')

    try:
        firstedges=[(x[0], x[1]) for x in Xs]
    except:
        firstedges=[]
    
    # find a good starting graph
    graph = genStartGraph(Xs, numnodes, td, fitinfo)

    # find best starting perseveration parameter if applicable JCZ 5/9/2018
    if fitinfo.estimatePerseveration:
        best_ll = -np.inf
        best_param = 0.0
        for censor_param in [i/100.0 for i in range(101)]:
            td.censor_fault = censor_param
            graph_ll, probmat = probX(Xs,graph,td,irts=irts,prior=prior)   # LL of starting graph
            if graph_ll > best_ll:
                best_ll = graph_ll
                best_param = censor_param
        td.censor_fault = best_param
        
    best_ll, probmat = probX(Xs,graph,td,irts=irts,prior=prior)   # LL of starting graph
    graph, best_ll = phases(graph, best_ll, probmat)

    return graph, best_ll
