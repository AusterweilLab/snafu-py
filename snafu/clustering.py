from . import *

# given list of cluster lengths, compute average cluster size of each list, then return avearge of that
# also works on single list
def clusterSize(l, scheme, clustertype='fluid'):
    clist = findClusters(l, scheme, clustertype)
    
    avglists=[]
    for i in clist:
        avglist=[]
        for l in i:
            avglist.append(np.mean(l))
        avglists.append(np.mean(avglist))
    return avglists

# given list of cluster lengths, compute average number of cluster switches of each list, then return avearge of that
# also works on single list
def clusterSwitch(l, scheme, clustertype='fluid',switchrate=False):
    clist = findClusters(l, scheme, clustertype)
    
    avglists=[]
    for inum, i in enumerate(clist):
        avgnum=[]
        if len(i) > 0:
            if isinstance(i[0], list):
                for lstnum, lst in enumerate(i):
                    switches = len(lst)-1
                    if switchrate:
                        switches = switches / len(l[inum][lstnum])
                    avgnum.append(switches)
                avglists.append(np.mean(avgnum))
            else:
                switches = len(i)-1
                if switchrate:
                    switches = switches / len(l[inum])
                avglists.append(switches)
        else:
            avglists.append(0)
    return avglists

# report average cluster size for list or nested lists
def findClusters(l, scheme, clustertype='fluid'):
    # only convert items to labels if list of items, not list of lists
    if len(l) > 0:
        if isinstance(l[0], list):
            clusters=l
        else:
            clusters=labelClusters(l, scheme)
    else:
        clusters=[]
    
    csize=[]
    curcats=set([])
    runlen=0
    clustList=[]
    firstitem=1
    for inum, item in enumerate(clusters):
        if isinstance(item, list):
            clustList.append(findClusters(item, scheme, clustertype=clustertype))
        else:
            newcats=set(item.split(';'))
            if newcats.isdisjoint(curcats) and firstitem != 1:      # end of cluster, append cluster length
                csize.append(runlen)
                runlen = 1
            else:                                                   # shared cluster or start of list
                runlen += 1
            
            if clustertype=="fluid":
                curcats = newcats
            elif clustertype=="static":
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
def labelClusters(l, scheme, labelIntrusions=False, targetLetter=None):
    if isinstance(scheme,str):
        clustertype = "semantic"    # reads clusters from a fixed file
    elif isinstance(scheme,int):
        clustertype = "letter"      # if an int is given, use the first N letters as a clustering scheme
        maxletters = scheme
        if targetLetter:
            targetLetter = targetLetter.lower()
        
    else:
        raise Exception('Unknown clustering type in labelClusters()')

    if clustertype == "semantic":
        cf=open(scheme,'rt', encoding='utf-8-sig')
        cats={}
        for line in cf:
            line=line.rstrip()
            if line[0] == "#": continue         # skip commented lines
            cat, item = line.split(',')
            cat=cat.lower().replace(' ','').replace("'","").replace("-","") # basic clean-up
            item=item.lower().replace(' ','').replace("'","").replace("-","")
            if item not in list(cats.keys()):
                cats[item]=cat
            else:
                if cat not in cats[item]:
                    cats[item]=cats[item] + ';' + cat
    labels=[]
    for inum, item in enumerate(l):
        if isinstance(item, list):
            labels.append(labelClusters(item, scheme, labelIntrusions=labelIntrusions, targetLetter=targetLetter))
        else:
            item=item.lower().replace(' ','')
            if clustertype == "semantic":
                if item in list(cats.keys()):
                    labels.append(cats[item])
                elif labelIntrusions:               # if item not in dict, either ignore it or label is as category "intrusion"
                    labels.append("intrusion")
            elif clustertype == "letter":
                if (item[0] == targetLetter) or ((targetLetter == None) and (labelIntrusions == False)):
                    labels.append(item[:maxletters])
                elif labelIntrusions:
                    if targetLetter == None:
                        raise Exception('Cant label intrusions without a target letter [labelClusters]')
                    else:
                         labels.append("intrusion")     # if item not in dict, either ignore it or label is as category "intrusion"
    return labels
