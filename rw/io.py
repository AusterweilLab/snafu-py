# IO functions:
#
# * Read graph from file
# * Write graph to file
# * Read fluency data from file

import textwrap
import numpy as np

# sibling functions
from helper import *

# ** DEPRECATED
# ** use nx.generate_sparse6(nx.to_networkx_graph(graph),header=False) instead
# helper function converts binary adjacency matrix to base 36 string for easy storage in CSV
# binary -> int -> base62
def graphToHash(a,numnodes):
    def baseN(num,b,numerals="0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ"):
        return ((num == 0) and numerals[0]) or (baseN(num // b, b, numerals).lstrip(numerals[0]) + numerals[num % b])
    return str(numnodes) + '!' + baseN(int(''.join([str(i) for i in flatten_list(a)]),2), 62)

# ** DEPRECATED
# see graphToHash function
def hashToGraph(graphhash):
    numnodes, graphhash = graphhash.split('!')
    numnodes=int(numnodes)
    graphstring=bin(int(graphhash, 36))[2:]
    zeropad=numnodes**2-len(graphstring)
    graphstring=''.join(['0' for i in range(zeropad)]) + graphstring
    arrs=textwrap.wrap(graphstring, numnodes)
    mat=np.array([map(int, s) for s in arrs])
    return mat

# reads in graph from CSV
# row order not preserved; could be optimized more
def read_csv(fh,cols=(0,1),header=False,filters={},undirected=True,sparse=False):
    fh=open(fh,'r')
    idx=0
    bigdict={}

    if header:
        headerrow=fh.readline().split('\n')[0].split(',')
        cols=(headerrow.index(cols[0]),headerrow.index(cols[1]))
        filterrows={}
        for i in filters.keys():
            filterrows[headerrow.index(i)]=filters[i]
    else:
        filterrows={}

    done=dict()
    for line in fh:
        line=line.rstrip()
        linesplit=line.split(',')
        twoitems=[linesplit[cols[0]],linesplit[cols[1]]]
       
        skiprow=0
        for i in filterrows:
            if linesplit[i]!=filterrows[i]:
                skiprow=1
        if skiprow==1:
            continue
        
        try:
            if twoitems[1] not in bigdict[twoitems[0]]:
                bigdict[twoitems[0]].append(twoitems[1])
        except:
            bigdict[twoitems[0]] = [twoitems[1]]
        if twoitems[1] not in bigdict:              # doesn't this scale with dictionary size-- something i was trying to avoid by rewriting this function?
            bigdict[twoitems[1]] = []


    
    items_rev = dict(zip(bigdict.keys(),range(len(bigdict.keys()))))
    items = dict(zip(range(len(bigdict.keys())),bigdict.keys()))
    
    if sparse:
        from scipy.sparse import csr_matrix
        rows=[]
        cols=[]
        numedges=0
        for i in bigdict:
            for j in bigdict[i]:
                rows.append(items_rev[i])
                cols.append(items_rev[j])
                numedges += 1
                if undirected:
                    rows.append(items_rev[j])
                    cols.append(items_rev[i])
                    numedges += 1
        data=np.array([1]*numedges)
        rows=np.array(rows)
        cols=np.array(cols)
        graph = csr_matrix((data, (rows, cols)), shape=(len(items),len(items)))
    else:
        graph = np.zeros((len(items),len(items)))
        
        for item1 in bigdict:
            for item2 in bigdict[item1]:
                idx1=items_rev[item1]
                idx2=items_rev[item2]
                graph[idx1,idx2]=1
                if undirected:
                    graph[idx2,idx1]=1

    return graph, items

#def oldreadX(subj,category,filepath,removePerseverations=False,removeIntrusions=False,spellfile=None,scheme=None,flatten=True):
#    if type(subj) == str:
#        subj=[subj]
#    game=-1
#    cursubj=-1
#    Xs=[]
#    irts=[]
#    items={}
#    idx=0
#    spellingdict={}
#    validitems=[]
#    
#    if removeIntrusions:
#        if not scheme:
#            raise ValueError('You need to provide a category scheme if you want to ignore intrusions!')
#        else:
#            # this should really go somewhere else, like io.py
#            with open(scheme,'r') as fh:
#                for line in fh:
#                    validitems.append(line.rstrip().split(',')[1].lower())
#
#    if spellfile:
#        with open(spellfile,'r') as spellfile:
#            for line in spellfile:
#                correct, incorrect = line.rstrip().split(',')
#                spellingdict[incorrect] = correct
#   
#    with open(filepath) as f:
#        for line in f:
#            row=line.strip('\n').split(',')
#            if (row[0] in subj) & (row[2] == category):
#                if (row[1] != game) or (row[0] != cursubj):
#                    Xs.append([])
#                    irts.append([])
#                    game=row[1]
#                    cursubj=row[0]
#                # basic clean-up
#                item=row[3].lower()
#                badchars=" '-\"\\;"
#                for char in badchars:
#                    item=item.replace(char,"")
#                if item in spellingdict.keys():
#                    item = spellingdict[item]
#                try:
#                    irt=row[4]
#                except:
#                    pass
#                if item not in items.values():
#                    items[idx]=item
#                    idx += 1
#                itemval=items.values().index(item)
#                if (not removePerseverations) or (itemval not in Xs[-1]):   # ignore any duplicates in same list resulting from spelling corrections
#                    if (not removeIntrusions) or (item in validitems):
#                        Xs[-1].append(itemval)
#                        try: 
#                            irts[-1].append(int(irt)/1000.0)
#                        except:
#                            pass
#    numnodes = len(items)
#    return Xs, items, irts, numnodes


# read Xs in from user files
# flatten == treat all subjects as identical; when False, keep hierarchical structure and dictionaries
def readX(subj,category,filepath,removePerseverations=False,removeIntrusions=False,spellfile=None,scheme=None,flatten=False):
    if (type(subj) == list) and not flatten:
        subj_data=[]
        for sub in subj:
            subj_data.append(readX(sub,category,filepath,removePerseverations,removeIntrusions,spellfile,scheme,flatten))
        Xs = [i[0] for i in subj_data]
        items = [i[1] for i in subj_data]
        irts = [i[2] for i in subj_data]
        numnodes = [i[3] for i in subj_data]
        
        groupitems={}
        idx=0
        for subitems in items:
            for item in subitems.values():
                if item not in groupitems.values():
                    groupitems[idx] = item
                    idx += 1
        
        groupnumnodes = len(groupitems)

        return Xs, items, irts, numnodes, groupitems, groupnumnodes
    else:
        if type(subj) == "str":
            subj=[subj]
        game=-1
        Xs=[]
        irts=[]
        items={}
        idx=0
        spellingdict={}
        validitems=[]
        
        if removeIntrusions:
            if not scheme:
                raise ValueError('You need to provide a category scheme if you want to ignore intrusions!')
            else:
                with open(scheme,'r') as fh:
                    for line in fh:
                        validitems.append(line.rstrip().split(',')[1].lower())

        if spellfile:
            with open(spellfile,'r') as spellfile:
                for line in spellfile:
                    correct, incorrect = line.rstrip().split(',')
                    spellingdict[incorrect] = correct
       
        with open(filepath) as f:
            for line in f:
                row=line.strip('\n').split(',')
                if (row[0] in subj) & (row[2] == category):
                    if (row[1] != game):
                        Xs.append([])
                        irts.append([])
                        game=row[1]
                    # basic clean-up
                    item=row[3].lower()
                    badchars=" '-\"\\;"
                    for char in badchars:
                        item=item.replace(char,"")
                    if item in spellingdict.keys():
                        item = spellingdict[item]
                    try:
                        irt=row[4]
                    except:
                        pass
                    if item not in items.values():
                        if (item in validitems) or (not removeIntrusions):
                            items[idx]=item
                            idx += 1
                    try:
                        itemval=items.values().index(item)
                        if (not removePerseverations) or (itemval not in Xs[-1]):   # ignore any duplicates in same list resulting from spelling corrections
                            if (not removeIntrusions) or (item in validitems):
                                Xs[-1].append(itemval)
                                try: 
                                    irts[-1].append(int(irt)/1000.0)
                                except:
                                    pass    # bad practice?
                    except:
                        pass                # bad practice?
        numnodes = len(items)
    return Xs, items, irts, numnodes

# some sloppy code in here; methods are different depending on whether you pass an nx or array (but should be the same)
# needs to be re-written
def write_csv(gs, fh, subj="NA", directed=False, extra_data={}):
    onezero={True: '1', False: '0'}        
    import networkx as nx
    fh=open(fh,'w',0)
    if isinstance(gs,nx.classes.graph.Graph):       # write nx graph
        edges=set(flatten_list([gs.edges()]))
        for edge in edges:
            isdirected=""
            if directed:
                isdirected="," + (onezero[(g.has_edge(edge[1],edge[0]) and g.has_edge(edge[1],edge[0]))])
            extrainfo=""
            if edge[0] in extra_data.keys():
                if edge[1] in extra_data[edge[0]].keys():
                    extrainfo=","+",".join([str(i) for i in extra_data[edge[0]][edge[1]]])
            fh.write(subj    + "," +
                    edge[0]  + "," +
                    edge[1]  + 
                    isdirected + 
                    extrainfo + "\n")
    else:                                           # write matrix
        edges=set(flatten_list([gs[i].edges() for i in range(len(gs))]))
        for edge in edges:
            edgelist=""
            for g in gs:
                edgelist=edgelist+"," + onezero[g.has_edge(edge[0],edge[1])]
            if directed:
                for g in gs:
                    edgelist=edgelist + "," + (onezero[(g.has_edge(edge[1],edge[0]) and g.has_edge(edge[1],edge[0]))]) # this doesn't look right...
            extrainfo=""
            sortededge=np.sort([edge[0],edge[1]])
            if sortededge[0] in extra_data.keys():
                if sortededge[1] in extra_data[sortededge[0]].keys():
                    if isinstance(extra_data[sortededge[0]][sortededge[1]],list):
                        extrainfo=","+",".join([str(i) for i in extra_data[sortededge[0]][sortededge[1]]])
                    else:
                        extrainfo=","+","+str(extra_data[sortededge[0]][sortededge[1]])
            fh.write(subj    + "," +
                    edge[0]  + "," +
                    edge[1]  + 
                    edgelist + "," +
                    extrainfo + "\n")
    return
