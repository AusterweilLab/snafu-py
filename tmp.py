hs=[]
crs=[]
ms=[]
fas=[]
for sub in range(numsubs):
    graph = uinvite_graphs[sub]
    hit=miss=fa=cr=0
    for rownum, row in enumerate(graph):
        for colnum, val in enumerate(row):
            item1 = items[sub][rownum]
            item2 = items[sub][colnum]
            idx1 = usf_items.keys()[usf_items.values().index(item1)]
            idx2 = usf_items.keys()[usf_items.values().index(item2)]
            if (val==1) and (usf_graph[idx1][idx2] == 1):
                hit += 1
            if (val==1) and (usf_graph[idx1][idx2] == 0):
                fa += 1
            if (val==0) and (usf_graph[idx1][idx2] == 1):
                miss += 1
            if (val==0) and (usf_graph[idx1][idx2] == 0):
                cr += 1
    #hs.append(hit / float(hit + fa)) # ~ .7
    #crs.append(cr / float(cr + miss)) # ~.96
    hs.append(hit)
    crs.append(cr)
    ms.append(miss)
    fas.append(fa)
