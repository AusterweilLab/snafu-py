import rw
import networkx as nx
import numpy as np
import pickle
import sys
import numpy as np
import scipy.stats
import random

subs=["S"+str(i) for i in range(101,151)]
filepath = "../Spring2017/results_clean.csv"
category="animals"
Xs, items, irtdata, numnodes = rw.readX(subs,category,filepath,ignorePerseverations=True,ignoreIntrusions=True,scheme="categories/troyer_hills_zemla_animals.csv",spellfile="categories/zemla_spellfile.csv")

pairs=[]
while len(pairs) < 200:
    a=b=0
    while a==b:
        a = random.randint(0,len(items)-1)
        b = random.randint(0,len(items)-1)
    pair=[items[a],items[b]]
    
    newedge=1
    with open('humans_2017.csv','r') as fh:
        for line in fh:
            line=line.split(',')
            already_an_edge=[line[1],line[2]]
            if (pair[0] in already_an_edge) and (pair[1] in already_an_edge):
                newedge=0
                #print "ALREADY AN EDGE", already_an_edge
    if newedge==1:
        pairs.append(pair)
        #print "NEW EDGE", pair

for edge in pairs:
    print "S100," + edge[0] + "," + edge[1] + ",0,0,0,0,0,0,0,0"

