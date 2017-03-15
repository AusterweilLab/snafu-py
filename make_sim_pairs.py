# selects pairs from networks to use in similarity task
#
# selects all edges that the models disagree on, plus a random sample of equal
# number of edges that the models agree on, and disagree on

import rw
import numpy as np

subs2015=['S101','S102','S103','S104','S105','S106','S107','S108','S109','S110',
      'S111','S112','S113','S114','S115','S116','S117','S118','S119','S120']
subs2016=['S1','S2','S3','S4','S5','S7','S8','S9','S10','S11','S12','S13']

fi='s2016.csv'
subs=subs2016

fo=open('pairs.csv','w')
subpairs=open('subpairs.csv','w')
finalpairs=[]

for sub in subs:
    # read in graphs
    g1, items1 = rw.read_csv(fi,cols=('node1','node2'),header=1,filters={'subj': sub, 'u-invite': '1'},undirected=1)
    g2, items2 = rw.read_csv(fi,cols=('node1','node2'),header=1,filters={'subj': sub, 'priming': '1'},undirected=1)
    g3, items3 = rw.read_csv(fi,cols=('node1','node2'),header=1,filters={'subj': sub, 'rw': '1'},undirected=1)
    g4, items4 = rw.read_csv(fi,cols=('node1','node2'),header=1,filters={'subj': sub, 'windowgraph': '1'},undirected=1)
    g5, items5 = rw.read_csv(fi,cols=('node1','node2'),header=1,filters={'subj': sub, 'irt5': '1'},undirected=1)
    g6, items6 = rw.read_csv(fi,cols=('node1','node2'),header=1,filters={'subj': sub, 'irt9': '1'},undirected=1)
    g7, items7 = rw.read_csv(fi,cols=('node1','node2'),header=1,filters={'subj': sub, 'irt95': '1'},undirected=1)
    
    # re-arrange graphs to have same indices
    idx=[items2.values().index(i) for i in items1.values()]
    g2=g2[idx][:,idx]
    idx=[items3.values().index(i) for i in items1.values()]
    g3=g3[idx][:,idx]
    idx=[items4.values().index(i) for i in items1.values()]
    g4=g4[idx][:,idx]
    idx=[items5.values().index(i) for i in items1.values()]
    g5=g5[idx][:,idx]
    idx=[items6.values().index(i) for i in items1.values()]
    g6=g6[idx][:,idx]
    idx=[items7.values().index(i) for i in items1.values()]
    g7=g7[idx][:,idx]
    
    gtotal=(g1+g2+g3+g4+g5+g6+g7)
    
    yes=zip(*np.where(gtotal==7))                # all models agree there is an edge
    no=zip(*np.where(gtotal==0))                 # all models agree there is no edge
    some=zip(*np.where(gtotal>0))                # models disagree on where there is an edge
    some=[i for i in some if i not in yes]
   
    # take only one triangle (and exclude self-edges)
    yes=[i for i in yes if i[0] > i[1]]
    no=[i for i in no if i[0] > i[1]]
    some=[i for i in some if i[0] > i[1]]

    # take a random sample of yes and no arrays (of len(some))
    yesidx=np.random.choice(range(len(yes)),len(some),replace=False)
    noidx=np.random.choice(range(len(no)),len(some),replace=False)
    yes=[yes[i] for i in yesidx]
    no=[no[i] for i in noidx]
    alls=(yes+no+some)
   
        
    # write which edges belong to which participants
    for i in alls:
        pair=list(np.sort([items1[i[0]], items1[i[1]]]))
        towrite=[str(sub), str(pair[0]), str(pair[1]), 
                 str(int(g1[i[0],i[1]])), 
                 str(int(g2[i[0],i[1]])), 
                 str(int(g3[i[0],i[1]])), 
                 str(int(g4[i[0],i[1]])),
                 str(int(g5[i[0],i[1]])),
                 str(int(g6[i[0],i[1]])),
                 str(int(g7[i[0],i[1]]))]
        subpairs.write(','.join(towrite) + '\n')

    for inum, i in enumerate(alls):
        alls[inum]=list(np.sort([items1[i[0]], items1[i[1]]]))    # num-> animal, alphabetized

    # only write each edge once, even if it occurs for multiple participants
    for i in alls:
        if i not in finalpairs:
            finalpairs.append(i)
    
# write list of pairs for experiment-- but not which subjects or networks they belong to
for pair in finalpairs:
    towrite=[str(i) for i in pair]
    fo.write(','.join(towrite) + '\n')

fo.close()
subpairs.close()
