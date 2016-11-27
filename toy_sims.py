import rw

outfile='tmp.csv'
header=1

toygraphs=rw.Toygraphs({
        'numgraphs': 1,
        'graphtype': "steyvers",
        'numnodes': 30,
        'numlinks': 6,
        'prob_rewire': .3})

toydata=rw.Toydata({
        'numx': [10],
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
        'tolerance': 1500,
        'startGraph': "naiverw",
        'prune_limit': 100,
        'triangle_limit': 100,
        'other_limit': 100})

# optionally, pass a methods argument
# default is methods=['fe','rw','uinvite','uinvite_irt'] 

for td in toydata:
    rw.toyBatch(toygraphs, td, outfile, irts=irts, start_seed=1,methods=['uinvite'],header=header,debug="T")
    header=0


# 160 nodes 5 lists (trim .7) 1:05:14.006897
# 160 nodes 3 lists (trim .7) 0:39:54.803221
