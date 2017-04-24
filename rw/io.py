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
# row order not preserved; not optimized
def read_csv(fh,cols=(0,1),header=False,filters={},undirected=True):
    fh=open(fh,'r')
    items={}
    idx=0
    biglist=[]

    if header:
        headerrow=fh.readline().split('\n')[0].split(',')
        cols=(headerrow.index(cols[0]),headerrow.index(cols[1]))
        filterrows={}
        for i in filters.keys():
            filterrows[headerrow.index(i)]=filters[i]
    else:
        filterrows={}

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
        
        biglist.append(twoitems)
        for item in twoitems:
            if item not in items.values():
                items[idx]=item
                idx += 1

    graph=np.zeros((len(items),len(items)))

    for twoitems in biglist:
        idx1=items.values().index(twoitems[0])
        idx2=items.values().index(twoitems[1])
        graph[idx1,idx2]=1
        if undirected:
            graph[idx2,idx1]=1

    return graph, items

# read Xs in from user files
def readX(subj,category,filepath,ignorePerseverations=False,spellfile=None):
    if type(subj) == str:
        subj=[subj]
    game=-1
    cursubj=-1
    Xs=[]
    irts=[]
    items={}
    idx=0
    spellingdict={}
    
    if spellfile:
        with open(spellfile,'r') as spellfile:
            for line in spellfile:
                correct, incorrect = line.strip('\n').split(',')
                spellingdict[incorrect] = correct
   
    with open(filepath) as f:
        for line in f:
            row=line.strip('\n').split(',')
            if (row[0] in subj) & (row[2] == category):
                if (row[1] != game) or (row[0] != cursubj):
                    Xs.append([])
                    irts.append([])
                    game=row[1]
                    cursubj=row[0]
                item=row[3].lower().replace(" ","").replace("'","").replace("-","") # basic clean-up
                if item in spellingdict.keys():
                    item = spellingdict[item]
                try:
                    irt=row[4]
                except:
                    pass
                if item not in items.values():
                    items[idx]=item
                    idx += 1
                itemval=items.values().index(item)
                if (itemval not in Xs[-1]) or (not ignorePerseverations):   # ignore any duplicates in same list resulting from spelling corrections
                    Xs[-1].append(itemval)
                    try: 
                        irts[-1].append(int(irt)/1000.0)
                    except:
                        pass
    numnodes = len(items)
    return Xs, items, irts, numnodes

# some sloppy code in here + extra_data doesn't work when passed list
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
                    edgelist=edgelist + "," + (onezero[(g.has_edge(edge[1],edge[0]) and g.has_edge(edge[1],edge[0]))])
            fh.write(subj    + "," +
                    edge[0]  + "," +
                    edge[1]  + 
                    edgelist + "\n")
    return
