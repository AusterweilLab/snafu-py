import rw
import numpy as np
import pickle

category="animals"
filepath="../Spring2017/results_clean.csv"

subs=["S"+str(i) for i in range(101,151)]

Xs, items, irtdata, numnodes = rw.readX(subs,category,filepath,removePerseverations=True,spellfile="spellfiles/zemla_spellfile.csv",flatten=True)

for x in Xs:
    for inum in range(len(x)-1):
        


