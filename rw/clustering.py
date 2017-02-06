import numpy as np

# given list of cluster lengths, compute average cluster size of each list, then return avearge of that
# also works on single list
def avgClusterSize(clist):
    avglist=[]
    for l in clist:
        avglist.append(np.mean(l))
    return np.mean(avglist)

# given list of cluster lengths, compute average number of cluster switches of each list, then return avearge of that
# also works on single list
def avgNumClusterSwitches(clist):
    avgnum=[]
    for l in clist:
        avgnum.append(len(l))
    return np.mean(avgnum)

# report average cluster size for list or nested lists
def clusterSize(l, scheme, clustertype='fluid'):
    # only convert items to labels if list of items, not list of lists
    if isinstance(l[0], list):
        clusters=l
    else:
        clusters=labelClusters(l, scheme)
    
    csize=[]
    curcats=set([])
    runlen=0
    clustList=[]
    firstitem=1
    for inum, item in enumerate(clusters):
        if isinstance(item, list):
            clustList.append(clusterSize(item, scheme, clustertype=clustertype))
        else:
            newcats=set(item.split(';'))
            if 'unknown' in newcats:
                print "Warning: Unknown category for item '", l[inum], "'. Add this item to your category labels or results may be incorrect!"
            if newcats.isdisjoint(curcats) and firstitem != 1:      # end of cluster, append cluster length
                csize.append(runlen)
                runlen = 1
            else:                                                   # shared cluster or start of list
                runlen += 1
            
            if clustertype=="fluid":
                curcats = newcats
            elif clustertype=="rigid":
                curcats = (curcats & newcats)
                if curcats==set([]):
                    curcats = newcats
            else:
                raise ValueError('Invalid cluster type')
        firstitem=0
    csize.append(runlen)
    if sum(csize) > 0:
        clustList += csize
    return clustList

# returns labels in place of items for list or nested lists
# provide list (l) and coding scheme (external file)
def labelClusters(l, scheme):
    cf=open(scheme,'r')
    cats={}
    for line in cf:
        line=line.rstrip()
        cat, items = line.split(',', 1)
        cat=cat.lower().replace(' ','')
        for item in items.split(','):
            if item != '':
                item=item.lower().replace(' ','')
                if item not in cats.keys():
                    cats[item]=cat
                else:
                    if cat not in cats[item]:
                        cats[item]=cats[item] + ';' + cat
    labels=[]
    for inum, item in enumerate(l):
        if isinstance(item, list):
            labels.append(labelClusters(item, scheme))
        else:
            item=item.lower().replace(' ','')
            if item in cats.keys():
                labels.append(cats[item])
            else:
                labels.append("unknown")
    return labels

# ** UNTESTED // modified from usf_cogsci.py
# avg num cluster types per list
def numClusters(data):
    numclusts_real=np.mean([len(set(rw.flatten_list([j.split(';') for j in i]))) for i in labelClusters(data,scheme)])
    return numclusts_real

