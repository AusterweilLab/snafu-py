import rw
import numpy as np

outfile='log.csv'
header=1

toygraphs=rw.Toygraphs({
        'numgraphs': 1,
        'graphtype': "steyvers",
        'numnodes': 50,
        'numlinks': 6,
        'prob_rewire': .3})

toydata=rw.Toydata({
        'numx': [25],
        'trim': .7,
        'jump': 0.0,
        'jumptype': "stationary",
        'startX': "stationary"})

irts=rw.Irts({
        'data': [],
        'irttype': "gamma",
        'beta': (1/1.1), 
        'irt_weight': 0.9,
        'rcutoff': 20})

fitinfo=rw.Fitinfo({
        'startGraph': "naiverw",
        'followtype': "avg", 
        'recorddir': "records/",
        'prune_limit': 100,
        'triangle_limit': 100,
        'other_limit': 100})

# optionally, pass a methods argument
# methods=['fe','rw','uinvite','uinvite_irt','uinvite_prior','uinvite_irt_prior','window'] 

for td in toydata:
    rw.toyBatch(toygraphs, td, outfile, irts=irts, fitinfo=fitinfo, start_seed=17, methods=['uinvite'],header=header,debug="T")
    header=0

# with limit=np.inf
# 160 nodes 5 lists (trim .7) 1:05:14.006897
# 160 nodes 3 lists (trim .7) 0:39:54.803221
# 100 nodes 15 lists (t7) 0:54:02.743266
