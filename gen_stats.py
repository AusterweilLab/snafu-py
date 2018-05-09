### What does this script do?
### 
### 1) Import the USF network from disk
### 2) Generate toy data (censored random walks) from that network for many pseudo-participants
### 3) Estimate the best network from the data using one or several methods, showing how the method
###    improves as more participants are added
### 4) Write data about those graphs to a CSV (cost, SDT measures)


import snafu
import networkx as nx
import numpy as np

subs=["S"+str(i) for i in range(101,151)]
filepath = "fluency/spring2017.csv"
category="animals"

# read data from file, preserving hierarchical structure
Xs, items, irtdata, numnodes, groupitems, groupnumnodes = snafu.readX(subs,filepath,category=category,removePerseverations=False,spellfile="spellfiles/zemla_spellfile.csv")

# for each participant, calculate the average cluster size and average number of perservations across that participant's lists
cluster_sizes = []
perseverations = []
for i in range(len(Xs)):
    cluster_sizes.append(snafu.avgClusterSize(Xs[i]))
    perseverations.append(snafu.avgNumPerseverations(Xs[i]))

