from . import *

def clusterSize(fluency_lists, scheme, clustertype='fluid'):
    """
    Calculate average cluster size of a fluency list (or list of fluency lists).

    This function expects a list of lists. If you want to calculate the average
    cluster size of a single list, you can wrap it in another list, e.g.,
    [fluency_list]

    Parameters
    ----------
    fluency_lists : list
        A list of fluency lists, e.g., fluencydata.labeledlists
    scheme : str or int
        For semantic fluency data, specify a path indicating clustering scheme
        (.csv) to use. For letter fluency data, specify an in integer
        indicating the number of initial letters to use as clusters (e.g., 2)
    clustertype : str, optional
        Type of clustering to apply. Default is 'fluid'. The other option is 'static'.

    Returns
    -------
    list of float
        A list containing the average cluster size in each fluency list.
    """
    clist = findClusters(fluency_lists, scheme, clustertype)
    
    avglists=[]
    for i in clist:
        avglist=[]
        for l in i:
            avglist.append(np.mean(l))
        avglists.append(np.mean(avglist))
    return avglists

def clusterSwitch(fluency_lists, scheme, clustertype='fluid', switchrate=False):
    """
    Calculate the number of cluster switches in a fluency list (or list of
    fluency lists. Alternatively, calculate the switch rate (number of switches
    divided by list length).

    This function expects a list of lists. If you want to calculate the number of
    cluster switches in a single list, you can wrap it in another list, e.g.,
    [fluency_list]

    Parameters
    ----------
    fluency_lists : list
        A list of fluency lists, e.g., fluencydata.labeledlists
    scheme : str or int
        For semantic fluency data, specify a path indicating clustering scheme
        (.csv) to use. For letter fluency data, specify an in integer
        indicating the number of initial letters to use as clusters (e.g., 2)
    clustertype : str, optional
        Type of clustering to apply. Default is 'fluid'. The other option is 'static'.
    switchrate : bool, optional
        If True, returns the switch rate instead of switch count. Default is False.

    Returns
    -------
    list of float
        A list containing the number of switches in each fluency list.
    """    
    clist = findClusters(fluency_lists, scheme, clustertype)
    avglists=[]
    for inum, i in enumerate(clist):
        avgnum=[]
        if len(i) > 0:
            if isinstance(i[0], list):
                for lstnum, lst in enumerate(i):
                    switches = len(lst)-1
                    if switchrate:
                        switches = switches / len(fluency_lists[inum][lstnum])
                    avgnum.append(switches)
                avglists.append(np.mean(avgnum))
            else:
                switches = len(i)-1
                if switchrate:
                    switches = switches / len(fluency_lists[inum])
                avglists.append(switches)
        else:
            avglists.append(0)
    return avglists

def findClusters(fluency_lists, scheme, clustertype='fluid'):
    """
    Calculate the size of each cluster in a fluency list (or list of fluency
    lists) and return these cluster sizes as a list. For example, ['dog',
    'cat', 'whale', 'shark'] might return [2, 2], as there are two clusters of
    size 2.

    This function is used internally by snafu.clusterSize and snafu.clusterSwitch.

    Parameters
    ----------
    fluency_lists : list
        A list of fluency lists, e.g., fluencydata.labeledlists
    scheme : str or int
        For semantic fluency data, specify a path indicating clustering scheme
        (.csv) to use. For letter fluency data, specify an in integer
        indicating the number of initial letters to use as clusters (e.g., 2)
    clustertype : str, optional
        Type of clustering to apply. Default is 'fluid'. The other option is 'static'.

    Returns
    -------
    list
        A list of cluster sizes (or nested list of cluster sizes).
    """

    if len(fluency_lists) > 0:
        if isinstance(fluency_lists[0], list):
            clusters=fluency_lists
        else:
            clusters=labelClusters(fluency_lists, scheme)
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

def labelClusters(fluency_lists, scheme, labelIntrusions=False, targetLetter=None):
    """
    Replace each item in a fluency list (or list of fluency lists) with its
    category or categories. For example, ['dog', 'cat', 'whale', 'shark'] might
    return ['canine;pets', 'pets', 'fish;water', 'fish;water'].
    
    This function is used internally by snafu.findClusters.
   
    Parameters
    ----------
    fluency_lists : list
        A list of fluency lists, e.g., fluencydata.labeledlists
    scheme : str or int
        For semantic fluency data, specify a path indicating clustering scheme
        (.csv) to use. For letter fluency data, specify an in integer
        indicating the number of initial letters to use as clusters (e.g., 2)
    labelIntrusions : bool, optional
        When False, intrusions are silently omitted (as if they do not exist).
        When True, intrusions are replaced with the pseudo-category label 'intrusion'.
        Default is False.
    targetLetter : str, optional
        For letter fluency data, identifies the target letter. This is
        necessary only to identify intrusions (when labelIntrusions is set to
        True), otherwise it has no effect. Default is None.

    Returns
    -------
    list
        A list (or nested list) of categoriesed corresponding to each item.
    """
    ...

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
    for inum, item in enumerate(fluency_lists):
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
