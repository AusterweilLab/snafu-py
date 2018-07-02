import numpy as np
import networkx as nx

from .helper import *

# frequency of each item in the data
# count also use collections.Counter ?
def freq(Xs, perlist=0):
    if perlist==1:
        Xs=[list(set(x)) for x in Xs]   # only count each item once per list
    Xflat=flatten_list(Xs)
    counts=[Xflat.count(i) for i in set(Xflat)]
    return dict(list(zip(set(Xflat),counts)))

# distribution of frequencies in the data (e.g., X items appeared once, Y items appeared twice, etc.)
def freq_stat(Xs):
    freqdist=list(freq(Xs).values())
    counts=[freqdist.count(i) for i in set(freqdist)]
    return dict(list(zip(set(freqdist),counts)))

# ** UNTESTED // modified from usf_cogsci.py
# num unique animals named across lists
def numNgrams(data, ngram=1):
    animallists=[find_ngrams(data[i],ngram) for i in range(len(data))]
    animalsNamed=len(set(flatten_list(animallists)))
    return animalsNamed
