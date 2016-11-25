import rw
import matplotlib.pyplot as plt
import numpy as np

outfile='tmp.csv'
header=1

toygraphs=rw.Toygraphs({
        'numgraphs': 10,
        'graphtype': "steyvers",
        'numnodes': 30,
        'numlinks': 6,
        'prob_rewire': .3})

toydata=rw.Toydata({
        'numx': range(5,25),
        'trim': .7,
        'jump': 0.0,
        'jumptype': "uniform",
        'startX': "uniform"})

irts=rw.Irts({
        'data': [],
        'irttype': "gamma",
        'beta': (1/1.1),
        'irt_weight': 0.9,
        'rcutoff': 20})

fitinfo=rw.Fitinfo({
        'tolerance': 1500,
        'startGraph': "naiverw",
        'prob_multi': 1.0,
        'prob_overlap': 0.5})

x_seed=1
graph_seed=1
td=toydata[0]

g,a=rw.genG(toygraphs,seed=graph_seed)
                                                                   
[Xs,irts.data]=zip(*[rw.genX(g, td, seed=x_seed+i) for i in range(td.numx)])
Xs=list(Xs)
irts.data=list(irts.data)
[Xs,irts.data,alter_graph]=rw.trimX(td.trim,Xs,irts.data)      # trim data when necessary

rw.drawMat(a,cmap=plt.cm.BuGn)
newmat=rw.drawMatChange(Xs, a, td, (0,1), keep=0)  # e.g.
