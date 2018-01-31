import rw
import networkx as nx
import numpy as np
import pickle
import sys
import numpy as np
import scipy.stats
import json

filepath = "kendra_fluency.csv"
category="animal"
s=[]
es=[]
m=[]
pm=[]

with open('kendra_SubjGroupKey.csv','r') as subjkey:
    for line in subjkey:
        subj, group = line.rstrip().split(',')
        if group=="S":
            s.append(subj)
        if group=="ES":
            es.append(subj)
        if group=="M":
            m.append(subj)
        if group=="PM":
            pm.append(subj)
    subs = e + es + m + pm
    
fo=open('kendra_networks.csv','w')
fo2=open('kendra_networks2.csv ','w')

for sub in subs:
    Xs, items, irtdata, numnodes = rw.readX(sub,category,filepath,removePerseverations=True,spellfile="schemes/kendra_spellfile.csv")
    graph = rw.chan(Xs, numnodes)
    nx_graph = nx.to_networkx_graph(graph)
    json_graph =  json.dumps(rw.gui.jsonGraph(nx_graph, items))

    fo.write(sub + "," + json_graph + "\n")

fo.close()
fo2.close()
