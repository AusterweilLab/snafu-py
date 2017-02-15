import rw

toydata=rw.Toydata({
        'numx': range(3,51),
        'priming': 0.5,
        'startX': "stationary"})

fitinfo=rw.Fitinfo({
        'startGraph': "windowgraph_valid",
        'followtype': "avg", 
        'recorddir': "records/",
        'prune_limit': 100,
        'triangle_limit': 100,
        'other_limit': 100})

toygraph=rw.Toygraphs({
        'numgraphs': 5,
        'graphtype': "steyvers",
        'numnodes': 50,
        'numlinks': 6,
        'prob_rewire': .3})

fh=open('priming_test.csv','w')

seed=1

for td in toydata:
    print "numx: ", td.numx
    # generate data with priming and fit best graph
    g,a=rw.genG(toygraph,seed=seed)
    [Xs,irts,alter_graph]=rw.genX(g, td)
    bestgraph_priming, ll=rw.uinvite(Xs, td, toygraph.numnodes, fitinfo=fitinfo)

    # fit best graph assuming no priming
    td.priming=0.0
    bestgraph_nopriming, ll=rw.uinvite(Xs, td, toygraph.numnodes, fitinfo=fitinfo)

    priming_cost=rw.cost(bestgraph_priming,a)
    nopriming_cost=rw.cost(bestgraph_nopriming,a)
    
    line = str(seed) + "," + str(td.numx) + "," + str(priming_cost) + "," + str(nopriming_cost) + "\n"
    fh.write(line)

fh.close()
