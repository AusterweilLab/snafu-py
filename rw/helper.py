# http://stackoverflow.com/a/32107024/353278
# use dot notation on dicts for convenience
class dotdict(dict):
    """
    Example:
    m = dotdict({'first_name': 'Eduardo'}, last_name='Pool', age=24, sports=['Soccer'])
    """
    def __init__(self, *args, **kwargs):
        super(dotdict, self).__init__(*args, **kwargs)
        for arg in args:
            if isinstance(arg, dict):
                for k, v in arg.iteritems():
                    self[k] = v

        if kwargs:
            for k, v in kwargs.iteritems():
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
        super(Map, self).__delitem__(key)
        del self.__dict__[key]

# helper function generate flast lists from nested lists
# modified from http://stackoverflow.com/a/952952/353278
# flattens list of list one level only, preserving non-list items
# flattens type list and type np.ndarray, nothing else (on purpose)
def flatten_list(l):
    import numpy as np
    l1=[item for sublist in l if isinstance(sublist,list) or isinstance(sublist,np.ndarray) for item in sublist]
    l=l1+[item for item in l if not isinstance(item,list) and not isinstance(item,np.ndarray)]
    return l

# log trick given list of log-likelihoods **UNUSED
def logTrick(loglist):
    logmax=max(loglist)
    loglist=[i-logmax for i in loglist]                     # log trick: subtract off the max
    p=math.log(sum([math.e**i for i in loglist])) + logmax  # add it back on
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

# modified from ExGUtils package by Daniel Gamermann <gamermann@gmail.com>
def rand_exg(irt, sigma, lambd):
    import math
    import numpy as np
    tau=(1.0/lambd)
    nexp = -tau*math.log(1.-np.random.random())
    ngau = np.random.normal(irt, sigma)
    return nexp + ngau

# decorator; prints elapsed time for function call
def timer(fun):
    from datetime import datetime
    def timerwrapper(*args, **kwargs):
        starttime=datetime.now()
        returnval = fun(*args, **kwargs)
        elapsedtime=str(datetime.now()-starttime)
        print elapsedtime
        return returnval
    return timerwrapper
    
