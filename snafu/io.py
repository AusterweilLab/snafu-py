# IO functions:
#
# * Read graph from file
# * Write graph to file
# * Read fluency data from file

import numpy as np
import csv
from  more_itertools import unique_everseen
from .structs import *

# sibling functions
from .helper import *

# wrapper
def graphToHash(a):
    return nx.generate_sparse6(nx.to_networkx_graph(a),header=False)

# wrapper
def hashToGraph(graphhash):
    return nx.to_numpy_matrix(nx.parse_graph6(graphhash))

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
        for i in list(filters.keys()):
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


    
    items_rev = dict(list(zip(list(bigdict.keys()),list(range(len(list(bigdict.keys())))))))
    items = dict(list(zip(list(range(len(list(bigdict.keys())))),list(bigdict.keys()))))
    
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
# this should be re-written with pandas or something more managable
def readX(ids,filepath,category=None,removePerseverations=False,removeIntrusions=False,spellfile=None,scheme=None,group=False):
   
    # grab header col indices
    mycsv = csv.reader(open(filepath,'rbU'))
    headers = next(mycsv, None)

    subj_col = headers.index('id')
    listnum_col = headers.index('listnum')
    item_col = headers.index('item')
    
    try:
        category_col = headers.index('category')
        has_category_col = True
    except:
        has_category_col = False
    try:
        group_col = headers.index('group')
        has_group_col = True
    except:
        has_group_col = False
        if group and (ids != "all"):
            raise ValueError('Data file does not have grouping column, but you asked for a specific group.')
    try:
        rt_col = headers.index('rt')
        has_rt_col = True
    except:
        has_rt_col = False

    # if ids is string wrap it in a list
    if isinstance(ids,str):
        ids=[ids]
        
    Xs=dict()
    irts=dict()
    items=dict()
    spellingdict=dict()
    validitems=[]
    
    # read in list of valid items when removeIntrusions = True
    if removeIntrusions:
        if not scheme:
            raise ValueError('You need to provide a category scheme if you want to ignore intrusions!')
        else:
            with open(scheme,'r') as fh:
                for line in fh:
                    if line[0] == "#": pass         # skip commented lines
                    validitems.append(line.rstrip().split(',')[1].lower())

    # read in spelling correction dictionary when spellfile is specified
    if spellfile:
        with open(spellfile,'r') as spellfile:
            for line in spellfile:
                if line[0] == "#": pass         # skip commented lines
                correct, incorrect = line.rstrip().split(',')
                spellingdict[incorrect] = correct
   
    with open(filepath,'rbU') as f:
        f.readline()    # discard header row
        for line in f:
            row=line.rstrip().split(',')

            listnum_int = int(row[listnum_col])
            
            storerow=False
            if ((group==False) and (row[subj_col] in ids)) or ((group==True) and (ids!=["all"]) and (row[group_col] in ids)) or ((group==True) and (ids==["all"])):
                if category == None:
                    storerow = True 
                elif (has_category_col == True) and (row[category_col] == category):
                    storerow = True
                else:
                    storerow = False
            else:
                storerow = False
            
            if storerow == True:

                # make sure dict keys exist
                if row[subj_col] not in Xs:
                    Xs[row[subj_col]] = dict()
                    if has_rt_col:
                        irts[row[subj_col]] = dict()
                if listnum_int not in Xs[row[subj_col]]:
                    Xs[row[subj_col]][listnum_int] = []
                    if has_rt_col:
                        irts[row[subj_col]][listnum_int] = []
                if row[subj_col] not in items:
                    items[row[subj_col]] = dict()
                    
                # basic clean-up
                item=row[item_col].lower()
                badchars=" '-\"\\;?"
                for char in badchars:
                    item=item.replace(char,"")
                if item in list(spellingdict.keys()):
                    item = spellingdict[item]
                if has_rt_col:
                    irt=row[rt_col]
                if item not in list(items[row[subj_col]].values()):
                    if (item in validitems) or (not removeIntrusions):
                        next_idx = len(items[row[subj_col]])
                        items[row[subj_col]][next_idx]=item
                        
                try:
                    itemval=list(items[row[subj_col]].values()).index(item)
                    if (not removePerseverations) or (itemval not in Xs[row[subj_col]][listnum_int]):   # ignore any duplicates in same list resulting from spelling corrections
                        if (item in validitems) or (not removeIntrusions):
                            Xs[row[subj_col]][listnum_int].append(itemval)
                            if has_rt_col:
                                irts[row[subj_col]][listnum_int].append(int(irt)/1000.0)
                except:
                    pass                # bad practice
   
    return Data({'Xs': Xs, 'items': items, 'irts': irts})

def write_graph(gs, fh, subj="NA", directed=False, extra_data={}, header=False):
    onezero={True: '1', False: '0'}        
    import networkx as nx
    fh=open(fh,'w',0)
    
    nodes = list(set(flatten_list([list(gs[i].nodes()) for i in range(len(gs))])))
    
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
                if edge[0] in list(extra_data.keys()):
                    if edge[1] in list(extra_data[edge[0]].keys()):
                        if isinstance(extra_data[edge[0]][edge[1]],list):
                            extrainfo=","+",".join([str(i) for i in extra_data[sortededge[0]][sortededge[1]]])
                        else:
                            extrainfo=","+str(extra_data[sortededge[0]][sortededge[1]])
                fh.write(subj    + "," +
                        str(edge[0])  + "," +
                        str(edge[1])  + 
                        edgelist + "," +
                        extrainfo + "\n")
    return
