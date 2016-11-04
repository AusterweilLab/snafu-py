import rw

# PARAMETERS OF TOY SMALL-WORLD GRAPH
#numnodes=50                           # number of nodes in graph
#numlinks=4                            # initial number of edges per node (must be even)
#probRewire=.3                         # probability of re-wiring an edge
#numedges=numnodes*(numlinks/2)        # number of edges in graph
#numx=3                                # How many observed lists?
#trim=.7                               # ~ What proportion of graph does each list cover?

# PARAMETERS FOR RECONTRUCTING GRAPH
#jeff=0.9                                # 1-IRT weight
#beta=(1/1.1)                            # for gamma distribution when generating IRTs from hidden nodes

# WRITE DATA
#numgraphs=10                         # number of toy graphs to generate/reconstruct
outfile='tmp2.csv'
header=1

toygraphs=rw.Toygraphs({
        'graphtype': "smallworld",
        'numgraphs': 10,
        'numnodes': 10,
        'numlinks': 4,
        'probRewire': .3})

toydata=rw.Toydata({
        'numx': 3,
        'trim': .7,
        'jump': 0.0,
        'jumptype': "uniform"
        'start': "stationary"})

irtinfo=rw.Irtinfo({
        'irttype': "gamma",
        'irt_weight': 0.9,
        'beta': (1/1.1)})

# optionally, pass a methods argument
# default is methods=['fe','rw','invite','inviteirt'] 
for numx in range(3,15):
    rw.toyBatch(toygraphs, toydata, outfile,  irtinfo=irtinfo, start_seed=1,methods=['rw','invitenoprior','invite'],prior=0,header=header)
    header=0
