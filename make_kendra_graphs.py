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
    subs = s + es + m + pm
    
fo=open('kendra_networks_kenett.csv','w')

for sub in subs:
    Xs, items, irtdata, numnodes = rw.readX(sub,category,filepath,removePerseverations=True,spellfile="spellfiles/kendra_spellfile.csv")
    #graph = rw.chan(Xs, numnodes)
    graph = rw.kenett(Xs, numnodes)
    nx_graph = nx.to_networkx_graph(graph)
    json_graph =  json.dumps(rw.gui.jsonGraph(nx_graph, items))

    fo.write(sub + "," + json_graph + "\n")

# M group
Xs, items, irtdata, numnodes = rw.readX(m,category,filepath,removePerseverations=True,spellfile="spellfiles/kendra_spellfile.csv",flatten=True)
#graph = rw.chan(Xs, numnodes)
graph = rw.kenett(Xs, numnodes)
nx_graph = nx.to_networkx_graph(graph)
json_graph =  json.dumps(rw.gui.jsonGraph(nx_graph, items))
fo.write("m," + json_graph + "\n")

# PM group
Xs, items, irtdata, numnodes = rw.readX(pm,category,filepath,removePerseverations=True,spellfile="spellfiles/kendra_spellfile.csv",flatten=True)
#graph = rw.chan(Xs, numnodes)
graph = rw.kenett(Xs, numnodes)
nx_graph = nx.to_networkx_graph(graph)
json_graph =  json.dumps(rw.gui.jsonGraph(nx_graph, items))
fo.write("pm," + json_graph + "\n")

# S group
Xs, items, irtdata, numnodes = rw.readX(s,category,filepath,removePerseverations=True,spellfile="spellfiles/kendra_spellfile.csv",flatten=True)
#graph = rw.chan(Xs, numnodes)
graph = rw.kenett(Xs, numnodes)
nx_graph = nx.to_networkx_graph(graph)
json_graph =  json.dumps(rw.gui.jsonGraph(nx_graph, items))
fo.write("s," + json_graph + "\n")

# ES group
Xs, items, irtdata, numnodes = rw.readX(es,category,filepath,removePerseverations=True,spellfile="spellfiles/kendra_spellfile.csv",flatten=True)
#graph = rw.chan(Xs, numnodes)
graph = rw.kenett(Xs, numnodes)
nx_graph = nx.to_networkx_graph(graph)
json_graph =  json.dumps(rw.gui.jsonGraph(nx_graph, items))
fo.write("es," + json_graph + "\n")

# S-ES group
Xs, items, irtdata, numnodes = rw.readX(s+es,category,filepath,removePerseverations=True,spellfile="spellfiles/kendra_spellfile.csv",flatten=True)
#graph = rw.chan(Xs, numnodes)
graph = rw.kenett(Xs, numnodes)
nx_graph = nx.to_networkx_graph(graph)
json_graph =  json.dumps(rw.gui.jsonGraph(nx_graph, items))
fo.write("ses," + json_graph + "\n")

# M-PM group
Xs, items, irtdata, numnodes = rw.readX(m+pm,category,filepath,removePerseverations=True,spellfile="spellfiles/kendra_spellfile.csv",flatten=True)
#graph = rw.chan(Xs, numnodes)
graph = rw.kenett(Xs, numnodes)
nx_graph = nx.to_networkx_graph(graph)
json_graph =  json.dumps(rw.gui.jsonGraph(nx_graph, items))
fo.write("mpm," + json_graph + "\n")


fo.close()
