# IO functions:
#
# * Read graph from file
# * Write graph to file
# * Read fluency data from file

import numpy as np
import csv
from  more_itertools import unique_everseen

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
    import textwrap
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
def read_graph(fh,cols=(0,1),header=False,filters={},undirected=True,sparse=False):
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

# read Xs in from user files
# flatten == treat all subjects as identical; when False, keep hierarchical structure and dictionaries
def readX(subj,category,filepath,removePerseverations=False,removeIntrusions=False,spellfile=None,scheme=None,flatten=False,factor_label="id",group=None):
   
    mycsv = csv.reader(open(filepath))
    headers = next(mycsv, None)
    subj_row = headers.index('id')
    category_row = headers.index('category')
    listnum_row = headers.index('listnum')
    item_row = headers.index('item')

    try:
        group_row = headers.index('group')
        has_group = True
    except:
        has_group = False
    try:
        rt_row = headers.index('rt')
        has_rts = True
    except:
        has_rts = False

    # if group is specified, replace subj with list of subjects who have that grouping; "all" takes all subjects
    # loops through file twice (once to grab subject ids), inefficient for large files
    if group != None:
        subj=[]
        if group=="all":
            for row in mycsv:
                subj.append(row[subj_row])
            subj = list(unique_everseen(subj)) # reduce to unique values, convert back to list
        else:
            if not has_group:
                raise ValueError('Data file does not have grouping column, but you asked for a specific group.')
            for row in mycsv:
                if row[group_row] == group:
                    subj.append(row[subj_row])
            subj = list(unique_everseen(subj))
    
    # for hierarchical
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
    else:                           # non-hierarchical
        if isinstance(subj,str):
            subj=[subj]
        listnum=-1
        cursubj="-1"    # hacky; hopefully no one has a subject labeled -1...
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
                row=line.rstrip().split(',')
                if (row[subj_row] in subj) and (row[category_row] == category):
                    if (row[listnum_row] != listnum) or ((row[listnum_row] == listnum) and (row[subj_row] != cursubj)):
                        Xs.append([])
                        irts.append([])
                        listnum=row[listnum_row]
                        cursubj=row[subj_row]
                    # basic clean-up
                    item=row[item_row].lower()
                    badchars=" '-\"\\;?"
                    for char in badchars:
                        item=item.replace(char,"")
                    if item in spellingdict.keys():
                        item = spellingdict[item]
                    if has_rts:
                        irt=row[rt_row]
                    if item not in items.values():
                        if (item in validitems) or (not removeIntrusions):
                            items[idx]=item
                            idx += 1
                    try:
                        itemval=items.values().index(item)
                        if (not removePerseverations) or (itemval not in Xs[-1]):   # ignore any duplicates in same list resulting from spelling corrections
                            if (not removeIntrusions) or (item in validitems):
                                Xs[-1].append(itemval)
                                if has_rts:
                                    irts[-1].append(int(irt)/1000.0)
                    except:
                        pass                # bad practice?
        numnodes = len(items)
    return Xs, items, irts, numnodes

def write_graph(gs, fh, subj="NA", directed=False, extra_data={}, header=False):
    onezero={True: '1', False: '0'}        
    import networkx as nx
    fh=open(fh,'w',0)

    nodes = list(set(flatten_list([gs[i].nodes() for i in range(len(gs))])))

    if header != False:
        fh.write('subj,item1,item2,'+ header+'\n')

    for node1 in nodes:
        for node2 in nodes:
            if (node1 < node2) or ((directed) and (node1 != node2)):   # write edges in alphabetical order unless directed graph
                edge = (node1,node2)
                edgelist=""
                for g in gs:
                    edgelist=edgelist+"," + onezero[g.has_edge(edge[0],edge[1])]    # assumes graph is symmetrical if directed=True !!
                extrainfo=""
                if edge[0] in extra_data.keys():
                    if edge[1] in extra_data[edge[0]].keys():
                        if isinstance(extra_data[edge[0]][edge[1]],list):
                            extrainfo=","+",".join([str(i) for i in extra_data[sortededge[0]][sortededge[1]]])
                        else:
                            extrainfo=","+str(extra_data[sortededge[0]][sortededge[1]])
                fh.write(subj    + "," +
                        edge[0]  + "," +
                        edge[1]  + 
                        edgelist + "," +
                        extrainfo + "\n")
    return
