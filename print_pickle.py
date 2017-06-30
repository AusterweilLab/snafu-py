import rw
import networkx as nx
import pickle

gs=[]
for filename in ["humans_rw.pickle","humans_fe.pickle","humans_goni.pickle","humans_chan.pickle","humans_kenett.pickle"]:
    fh=open(filename,"r")
    alldata=pickle.load(fh)
    fh.close()
    g=nx.to_networkx_graph(alldata['graph'])
    nx.relabel_nodes(g, alldata['items'], copy=False)
    gs.append(g)

rw.write_csv(gs,"humans2017.csv")
