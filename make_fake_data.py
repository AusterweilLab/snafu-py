import networkx as nx
import rw
import numpy as np

# length of fluency lists by subject (Brown, Spring 2015 data)
listlengths=[[28,34,38], [18,22,28], [45,52,61],
             [35,38,39], [14,20,24], [17,20,21],
             [41,47,50], [41,38,47], [26,31,27],
             [41,37,29], [15,19,16], [39,34,43],
             [45,34,40], [39,37,43], [24,26,19], 
             [32,35,41], [41,46,52], [29,33,28],
             [30,29,24], [42,43,36]]

subs=['S101','S102','S103','S104','S105','S106','S107','S108','S109','S110',
      'S111','S112','S113','S114','S115','S116','S117','S118','S119','S120']

toydata=rw.Toydata({
        'numx': 3,
        'trim': 1,
        'jump': 0.0,
        'jumptype': "stationary",
        'startX': "stationary"})

fitinfo=rw.Fitinfo({
        'startGraph': "windowgraph",
        'windowgraph_size': 2,
        'windowgraph_threshold': 2,
        'followtype': "avg", 
        'prior_samplesize': 10000,
        'recorddir': "records/",
        'prune_limit': 100,
        'triangle_limit': 100,
        'other_limit': 100})

irts=rw.Irts({
        'data': [],
        'irttype': "gamma",
        'beta': (1/1.1), 
        'irt_weight': 0.9,
        'rcutoff': 20})

for subj in subs:
    category="animals"
    Xs, items, irts.data, numnodes=rw.readX(subj,category,'pooled.csv')
    best_graph, bestval=rw.findBestGraph(Xs, toydata, numnodes, fitinfo=fitinfo)
    best_rw=rw.noHidden(Xs, numnodes)
    best_wg=rw.windowGraph(Xs, numnodes, td=toydata, valid=1, fitinfo=fitinfo)
    g=nx.to_networkx_graph(best_graph)
    g2=nx.to_networkx_graph(best_rw)
    g3=nx.to_networkx_graph(best_wg)
    nx.relabel_nodes(g, items, copy=False)
    nx.relabel_nodes(g2, items, copy=False)
    nx.relabel_nodes(g3, items, copy=False)
    rw.write_csv([g, g2, g3],subj+".csv",subj) # write multiple graphs
    
    
    
     

## generate fake data and irts yoked to real data
#fakedata=[]
#fakeirts=[]
#for subj in listlengths:
#    fakedata.append([])
#    fakeirts.append([])
#    for trimval in subj:
#        [data,irts]=zip(*[rw.genX(usfg, toydata)])
#        [data,irts,alter_graph]=rw.trimX(trimval,data,irts)      # trim data when necessary
#        fakedata[-1].append(data[0])
#        fakeirts[-1].append(irts[0])
#
#freqdata=[rw.freq_stat(i) for i in fakedata]
#freqs={1: [], 2: [], 3: []}
#for i in freqdata:
#    for numtimes in range(1,4):
#        try:
#            freqs[numtimes].append(i[numtimes])
#        except:
#            freqs[numtimes].append(0)
#
#for i in freqs:
#    freqs[i]=np.mean(freqs[i])

# (53.39, 18.25, 3.75)
# c.f. 
# dat2<-dat[category=="animals",.N,keyby=.(id,item,game)]
# dat3<-dat2[,.N,keyby=.(item,id)]
# dat4<-table(dat3[,N,keyby=id])
# mean(dat4[,1]) # [,2] [,3] --> (23.35, 17.25, 13.85)
