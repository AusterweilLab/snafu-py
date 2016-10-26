from helper import *

# frequency of each item in the data
def freq(Xs, perlist=0):
    if perlist==1:
        Xs=[list(set(x)) for x in Xs]   # only count each item once per list
    Xflat=flatten_list(Xs)
    counts=[Xflat.count(i) for i in set(Xflat)]
    return dict(zip(set(Xflat),counts))

# distribution of frequencies in the data (e.g., X items appeared once, Y items appeared twice, etc.)
def freq_stat(Xs):
    freqdist=freq(Xs).values()
    counts=[freqdist.count(i) for i in set(freqdist)]
    return dict(zip(set(freqdist),counts))


