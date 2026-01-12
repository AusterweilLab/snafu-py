from . import *


# return simulated data on graph g
# also return number of steps between first hits (to use for IRTs)
def gen_lists(g, td, seed=None):
    """
    Generate simulated lists by running censored random walks on a graph.

    This function simulates category fluency-style data by performing repeated 
    random walks on a given graph `g`, using parameters defined in `td`. It returns
    the visited item lists, unused step info (placeholder), and a flag indicating
    whether the output used all nodes in the graph.

    Parameters
    ----------
    g : networkx.Graph
        The graph on which the random walks are performed.
    td : DataModel
        Object containing simulation parameters such as list length, jump probability, 
        priming, and censoring behavior.
    seed : int, optional
        Random seed for reproducibility. Default is None.

    Returns
    -------
    list of list
        Simulated fluency data (visited node sequences) for each run.
    list
        Placeholder for step counts between items (currently unused).
    int
        A binary flag indicating if not all nodes were visited (1 = graph altered).
    """    
    Xs=[]
    steps=[]
    priming_vector=[]

    for xnum in range(td.numx):
        if seed == None:
            seedy = None
        else:
            seedy = seed + xnum
        rwalk=random_walk(g, td, priming_vector=priming_vector, seed=seedy)
        x=censored(rwalk, td)
        # fh=list(zip(*firstHits(rwalk)))[1]
        # step=[fh[i]-fh[i-1] for i in range(1,len(fh))]
        Xs.append(x)
        if td.priming > 0.0:
            priming_vector=x[:]
        # steps.append(step)
    td.priming_vector = []      # reset mutable priming vector between participants; JCZ added 9/29, untested

    alter_graph_size=0
    if td.trim != 1.0:
        numnodes=nx.number_of_nodes(g)
        for i in range(numnodes):
            if i not in set(flatten_list(Xs)):
                alter_graph_size=1

    return Xs, steps, alter_graph_size

# given an adjacency matrix, take a random walk that hits every node; returns a list of tuples
def random_walk(g, td, priming_vector=[], seed=None):
    """
    Perform a censored random walk on a graph, optionally with priming and jumping.

    This function simulates a walk on graph `g`, incorporating jump probabilities,
    censoring behavior, and optional priming effects to simulate memory recall or
    fluency search strategies.

    Parameters
    ----------
    g : networkx.Graph
        Graph on which the random walk is performed.
    td : DataModel
        Object defining walk behavior: jump probability, jump type, max steps, 
        starting node policy, censoring, and priming.
    priming_vector : list, optional
        List of previously visited nodes to bias the walk. Default is empty.
    seed : int, optional
        Random seed for reproducibility. Default is None.

    Returns
    -------
    list of tuple
        A list of (from_node, to_node) edges representing the random walk path.
    """

    import scipy.stats
    nplocal=np.random.RandomState(seed)    

    def jump():
        if td.jumptype=="stationary":
            second=statdist.rvs(random_state=seed)     # jump based on statdist
        elif td.jumptype=="uniform": 
            second=nplocal.choice(nx.nodes(g))         # jump uniformly
        return second

    if (td.start_node=="stationary") or (td.jumptype=="stationary"):
        a=nx.to_numpy_array(g)
        t=a/sum(a).astype(float)
        statdist=stationary(t)
        statdist=scipy.stats.rv_discrete(values=(list(range(len(t))),statdist))
    
    if td.start_node=="stationary":
        start=statdist.rvs(random_state=seed)    # choose starting point from stationary distribution
    elif td.start_node=="uniform":
        start=nplocal.choice(nx.nodes(g))        # choose starting point uniformly
    elif td.start_node[0]=="specific":
        start=td.start_node[1]

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

    numsteps = 0
    while (len(unused_nodes) > num_unused) and ((td.maxsteps == None) or (numsteps < td.maxsteps)):       # covers td.trim nodes-- list could be longer if it has perseverations

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
        numsteps += 1
        if second in unused_nodes:
            unused_nodes.remove(second)
            censoredcount=0
        else:
            censoredcount += 1
        first=second
    return walk
