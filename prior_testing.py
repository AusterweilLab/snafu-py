import networkx as nx
import numpy as np


n=100
k=4
p=.3
tries=1000
recover_p=[]
numtris=[]
tm=[]
tstd=[]

g1=nx.connected_watts_strogatz_graph(n,k,0,tries)
tri=float(sum(nx.triangles(g1).values())/3)
# tri = (k-3)*n # when n is large relative to k

for p in range(0,11,1):
    p=p/10.0
    for i in range(1000):
        g=nx.connected_watts_strogatz_graph(n,k,p,tries)
        numtri=float(sum(nx.triangles(g).values())/3) # number of triangles
        numtris.append(numtri)
        numtri=numtri-(n*k)*(float(k)/(n-k)) # adjust for k/(n-k) prob of rewiring to make new triangle by chance
        newp=(numtri/tri) # recovered w-s p
        recover_p.append(newp)
    tstd.append(np.std(numtris))
    tm.append(np.mean(numtris))

newp=np.mean(recover_p)
