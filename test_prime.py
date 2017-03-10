import rw

toydata=rw.Toydata({
        'numx': range(50,51),
        'priming': 0.5,
        'jump': 0.0,
        'trim': 1.0,
        'jumptype': "stationary",
        'startX': "stationary"})

fitinfo=rw.Fitinfo({
        'startGraph': "windowgraph_valid",
        'followtype': "avg", 
        'record': False,
        'recorddir': "records/",
        'prune_limit': 100,
        'triangle_limit': 100,
        'other_limit': 100})

toygraph=rw.Toygraphs({
        'graphtype': "steyvers",
        'numnodes': 50,
        'numlinks': 6})

fh=open('priming_test.csv','w')

seed=12

for td in toydata:
    print "numx: ", td.numx
    # generate data with priming and fit best graph
    g,a=rw.genG(toygraph,seed=seed)
    [Xs,irts,alter_graph]=rw.genX(g, td,seed=seed)
    bestgraph_priming, ll=rw.uinvite(Xs, td, toygraph.numnodes, fitinfo=fitinfo, seed=seed)
    priming_cost=rw.cost(bestgraph_priming,a)
    print priming_cost

    td.priming=0.0
    # fit best graph assuming no priming
    bestgraph_nopriming, ll=rw.uinvite(Xs, td, toygraph.numnodes, fitinfo=fitinfo, seed=seed)
    nopriming_cost=rw.cost(bestgraph_nopriming,a)
    print nopriming_cost
    
    line = str(seed) + "," + str(td.numx) + "," + str(priming_cost) + "," + str(nopriming_cost) + "\n"
    fh.write(line)

fh.close()
