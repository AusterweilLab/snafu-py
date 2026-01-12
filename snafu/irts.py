from . import *

# returns a vector of how many hidden nodes to expect between each Xi for each X in Xs
def expectedHidden(Xs, a):
    """
    Compute expected number of hidden node visits for each position in sequences.

    This function takes a list of index sequences and a transition count matrix,
    and calculates the expected number of hidden transitions (i.e., visits to 
    intermediate nodes) between each pair of observed nodes in the sequence, 
    assuming a Markov process.

    Parameters
    ----------
    Xs : list of list of int
        A list of sequences, where each sequence is a list of node indices.
        Each sequence represents observed states in order.
    
    a : ndarray of shape (n, n)
        A square matrix representing raw transition counts between `n` nodes.
        The matrix will be normalized to form a transition probability matrix.

    Returns
    -------
    expecteds : list of list of float
        A list where each element corresponds to a sequence in `Xs`, and contains 
        the expected number of hidden node visits between each consecutive pair 
        of observed nodes.
    """

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
    """
    Convert step-based inter-response times (IRTs) to simulated time-based IRTs.

    This function takes a set of IRTs (interpreted as mean durations in steps) and 
    simulates corresponding continuous-valued IRTs based on a specified distribution 
    type. It supports gamma and ex-Gaussian distributions.

    Parameters
    ----------
    irts : object
    An object containing:
        - `data` : list of list of float
            Step-based IRTs for each trial or sequence.
        - `irttype` : {'gamma', 'exgauss'}
            Type of distribution used to simulate real-valued IRTs.
        - `gamma_beta` : float, optional
            Inverse scale parameter (rate) for the gamma distribution.
        - `exgauss_sigma` : float, optional
            Standard deviation of the Gaussian component for the ex-Gaussian distribution.
        - `exgauss_lambda` : float, optional
            Rate parameter of the exponential component for the ex-Gaussian distribution.

    seed : int or None, optional
        Random seed for reproducibility.

    Returns
    -------
    new_irts : list of list of float
        Simulated time-based IRTs, one list per sequence in the input.
    """    
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
    """
    Trim sequences to a fixed length or proportion and identify unused nodes.

    This function is primarily used to simulate human data that does not
    fully cover the entire graph. Although its functionality has been mostly
    moved to `genX`, it may still be useful in niche cases.

    Parameters
    ----------
    trimprop : float or int
        If ≤ 1, interpreted as the proportion of each list to retain.
        If > 1, interpreted as the absolute number of items to retain.

    Xs : list of list of int
        Sequences of node indices (e.g., search paths) through a graph.

    steps : list of list
        Sequences of step-related values, aligned with `Xs`.

    Returns
    -------
    Xs_trimmed : list of list of int
        Trimmed node index sequences.

    steps_trimmed : list of list
        Corresponding trimmed step sequences.

    alter_graph_size : int
        1 if any graph nodes are unused after trimming, else 0.
    """

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
