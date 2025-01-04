
from . import *

# http://stackoverflow.com/a/32107024/353278
# use dot notation on dicts for convenience
class dotdict(dict):
    def __init__(self, *args, **kwargs):
        super(dotdict, self).__init__(*args, **kwargs)
        for arg in args:
            if isinstance(arg, dict):
                for k, v in list(arg.items()):
                    self[k] = v

        if kwargs:
            for k, v in list(kwargs.items()):
                self[k] = v

    def __getattr__(self, attr):
        return self.get(attr)

    def __setattr__(self, key, value):
        self.__setitem__(key, value)

    def __setitem__(self, key, value):
        super(dotdict, self).__setitem__(key, value)
        self.__dict__.update({key: value})

    def __delattr__(self, item):
        self.__delitem__(item)

    def __delitem__(self, key):
        super(Map, self).__delitem__(key)  # TODO: no definition of Map
        del self.__dict__[key]

# from http://locallyoptimal.com/blog/2013/01/20/elegant-n-gram-generation-in-python/
# generate list of ngrams
def find_ngrams(input_list, n):
    return list(zip(*[input_list[i:] for i in range(n)]))

# modified from ExGUtils package by Daniel Gamermann <gamermann@gmail.com>
# helper function generate flast lists from nested lists
# modified from http://stackoverflow.com/a/952952/353278
# flattens list of list one level only, preserving non-list items
# flattens type list and type np.ndarray, nothing else (on purpose)
def flatten_list(l, numtimes=1):
    l1 = [item for sublist in l if isinstance(sublist,list) or isinstance(sublist,np.ndarray) for item in sublist]
    l = l1+[item for item in l if not isinstance(item,list) and not isinstance(item,np.ndarray)]
    if numtimes > 1:
        l = flatten_list(l, numtimes-1)
    return l

# log trick given list of log-likelihoods **UNUSED
def logTrick(loglist):
    logmax=max(loglist)
    loglist=[i-logmax for i in loglist]                     # log trick: subtract off the max
    p=np.log(sum([np.e**i for i in loglist])) + logmax  # add it back on
    return p

# helper function grabs highest n items from list items **UNUSED
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

# find best ex-gaussian parameters
# port from R's retimes library, mexgauss function by Davide Massidda <davide.massidda@humandata.it>
# returns [mu, sigma, lambda]
def mexgauss(rts):
    n = len(rts)
    k = [np.nan, np.nan, np.nan]
    start = [np.nan, np.nan, np.nan]
    k[0] = np.mean(rts)
    xdev = [rt - k[0] for rt in rts]
    k[1] = sum([i**2 for i in xdev])/(n - 1.0)
    k[2] = sum([i**3 for i in xdev])/(n - 1.0)
    if (k[2] > 0):
        start[2] = (k[2]/2.0)**(1/3.0)
    else:
        start[2] = 0.8 * np.std(rts)
    start[1] = np.sqrt(abs(k[1] - start[2]**2))
    start[0] = k[0] - start[2]
    start[2] = (1.0/start[2])   # tau to lambda
    return(start)

# decorator; disables garbage collection before a function, enable gc after function completes
# provides some speed-up for functions that have lots of unnecessary garbage collection (e.g., lots of list appends)
def nogc(fun):
    import gc
    def gcwrapper(*args, **kwargs):
        gc.disable()
        returnval = fun(*args, **kwargs)
        gc.enable()
        return returnval
    return gcwrapper

# take list of lists in number/node and translate back to items using dictionary (e.g., 1->dog, 2->cat)
def numToItemLabel(data, items):
    new_data=[]
    for l in data:
        new_data.append([])
        for i in l:
            new_data[-1].append(items[i])
    return new_data

# modified from ExGUtils package by Daniel Gamermann <gamermann@gmail.com>
def rand_exg(irt, sigma, lambd):
    tau=(1.0/lambd)
    nexp = -tau*np.log(1.-np.random.random())
    ngau = np.random.normal(irt, sigma)
    return nexp + ngau

#def renumber(Xs,numsubs,numper):
#    start=0
#    end=numper
#    ssnumnodes=[]
#    itemsb=[]
#    datab=[]
#    for sub in range(len(subs)):
#        subXs = Xs[start:end]
#        itemset = set(snafu.flatten_list(subXs))
#        ssnumnodes.append(len(itemset))
#                                                    
#        ss_items = {}
#        convertX = {}
#        for itemnum, item in enumerate(itemset):
#            ss_items[itemnum] = items[item]
#            convertX[item] = itemnum
#                                                    
#        itemsb.append(ss_items)
#                                                    
#        subXs = [[convertX[i] for i in x] for x in subXs]
#        datab.append(subXs)
#        start += 3
#        end += 3

# decorator; prints elapsed time for function call
def timer(fun):
    from datetime import datetime
    def timerwrapper(*args, **kwargs):
        starttime=datetime.now()
        returnval = fun(*args, **kwargs)
        elapsedtime=str(datetime.now()-starttime)
        print(elapsedtime)
        return returnval
    return timerwrapper

def reverseDict(items):
    newitems=dict()
    for itemnum in items:
        itemlabel = items[itemnum]
        newitems[itemlabel] = itemnum
    return newitems

# remove perseverations -- keep only first occurrence in place
# https://www.peterbe.com/plog/uniqifiers-benchmark
def no_persev(x):
    seen = set()
    seen_add = seen.add
    return [i for i in x if not (i in seen or seen_add(i))]

# this function is copied from scipy to avoid shipping that whole library with snafu
# unlike scipy version, this one doesn't return p-value (requires C code from scipy)
def pearsonr(x, y):
    
    def _sum_of_squares(a, axis=0):
        a, axis = _chk_asarray(a, axis)
        return np.sum(a*a, axis)

    def _chk_asarray(a, axis):
        if axis is None:
            a = np.ravel(a)
            outaxis = 0
        else:
            a = np.asarray(a)
            outaxis = axis

        if a.ndim == 0:
            a = np.atleast_1d(a)

        return a, outaxis
    
    
    # x and y should have same length.
    x = np.asarray(x)
    y = np.asarray(y)
    n = len(x)
    mx = x.mean()
    my = y.mean()
    xm, ym = x - mx, y - my
    r_num = np.add.reduce(xm * ym)
    r_den = np.sqrt(_sum_of_squares(xm) * _sum_of_squares(ym))
    r = r_num / r_den

    return r

# takes an individual's data in group space and translates it into local space
def groupToIndividual(Xs, group_dict):
    itemset = set(flatten_list(Xs))
    
    ss_items = {}
    convertX = {}
    for itemnum, item in enumerate(itemset):
        ss_items[itemnum] = group_dict[item]
        convertX[item] = itemnum
    
    Xs = [[convertX[i] for i in x] for x in Xs]
    
    return Xs, ss_items

# take Xs and convert them from numbers (nodes) to labels
def numToLabel(Xs, items):
    for lnum, l in enumerate(Xs):
        for inum, i in enumerate(l):
            Xs[lnum][inum]=items[i]
    return Xs

# flat list from tuple walk
def nodes_from_edges(walk):
    path=list(list(zip(*walk))[0]) # first element from each tuple
    path.append(walk[-1][1]) # second element from last tuple
    return path

# tuple walk from flat list
def edges_from_nodes(path):
    walk=[]
    for i in range(len(path)-1):
        walk.append((path[i],path[i+1])) 
    return walk

def stationary(t, method="unweighted"):
    if method=="unweighted":                 # only works for unweighted matrices!
        return sum(t>0)/float(sum(sum(t>0)))
    else:                                       # buggy
        eigen=np.linalg.eig(t)[1][:,0]
        return np.real(eigen/sum(eigen))


# Unique nodes in random walk preserving order
# (aka fake participant data)
# http://www.peterbe.com/plog/uniqifiers-benchmark
def censored(walk, td=None, seed=None):
    def addItem(item):
        seen[item] = 1
        result.append(item)

    nplocal = np.random.RandomState(seed)
    seen = {}
    result = []
    for item in nodes_from_edges(walk):
        if item in seen:
            try:
                if nplocal.rand() <= td.censor_fault:
                    addItem(item)
            except:
                continue
        else:
            try:
                if nplocal.rand() <= td.emission_fault:
                    continue
                else:
                    addItem(item)
            except:
                addItem(item)
    return result

# first hitting times for each node
# TODO: Doesn't work with faulty censoring!!!
def firstHits(walk):
    firsthit=[]
    path=edges_from_nodes(walk)
    for i in censored(walk):
        firsthit.append(path.index(i))
    return list(zip(censored(walk),firsthit))
