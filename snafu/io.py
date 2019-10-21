# IO functions:
#
# * Read graph from file
# * Write graph to file
# * Read fluency data from file

from . import *

# wrapper
def graphToHash(a):
    """One line description here.
    
        Detailed description here. Detailed description here.  Detailed 
        description here.  
    
        Args:
            arg1 (type): Description here.
            arg2 (type): Description here.
        Returns:
            Detailed description here. Detailed description here.  Detailed 
            description here. 
    """
    return nx.generate_sparse6(nx.to_networkx_graph(a),header=False)

# wrapper
def hashToGraph(graphhash):
    """One line description here.
    
        Detailed description here. Detailed description here.  Detailed 
        description here.  
    
        Args:
            arg1 (type): Description here.
            arg2 (type): Description here.
        Returns:
            Detailed description here. Detailed description here.  Detailed 
            description here. 
    """
    return nx.to_numpy_matrix(nx.parse_graph6(graphhash))

# reads in graph from CSV
# row order not preserved; could be optimized more
def read_graph(fh,cols=(0,1),header=False,filters={},undirected=True,sparse=False):
    """One line description here.
    
        Detailed description here. Detailed description here.  Detailed 
        description here.  
    
        Args:
            arg1 (type): Description here.
            arg2 (type): Description here.
        Returns:
            Detailed description here. Detailed description here.  Detailed 
            description here. 
    """
    fh=open(fh,'rt', encoding='utf-8-sig')
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

# deprecated function name
def readX(*args, **kwargs):
    """One line description here.
    
        Detailed description here. Detailed description here.  Detailed 
        description here.  
    
        Args:
            arg1 (type): Description here.
            arg2 (type): Description here.
        Returns:
            Detailed description here. Detailed description here.  Detailed 
            description here. 
    """
    return load_fluency_data(*args, **kwargs)

def load_graph(*args, **kwargs):
    """One line description here.
    
        Detailed description here. Detailed description here.  Detailed 
        description here.  
    
        Args:
            arg1 (type): Description here.
            arg2 (type): Description here.
        Returns:
            Detailed description here. Detailed description here.  Detailed 
            description here. 
    """
    return read_graph(*args, **kwargs)

# read Xs in from user files
# this should be re-written with pandas or something more managable
def load_fluency_data(filepath,category=None,removePerseverations=False,removeIntrusions=False,spell=None,scheme=None,group=None,subject=None,cleanBadChars=False,hierarchical=False,targetletter=None):
    """One line description here.
    
        Detailed description here. Detailed description here.  Detailed 
        description here.  
    
        Args:
            arg1 (type): Description here.
            arg2 (type): Description here.
        Returns:
            Detailed description here. Detailed description here.  Detailed 
            description here. 
    """
   
    if targetletter:
        targetletter = targetletter.lower()
    if type(group) is str:
        group = [group]
    if type(subject) is str:
        subject = [subject]
    if type(category) is str:
        category = [category]

    # grab header col indices
    mycsv = csv.reader(open(filepath,'rt', encoding='utf-8-sig'))
    headers = next(mycsv, None)
    subj_col = headers.index('id')
    listnum_col = headers.index('listnum')
    item_col = headers.index('item')
    
    # check for optional columns
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
        if group:
            raise ValueError('Data file does not have grouping column, but you asked for a specific group.')
    try:
        rt_col = headers.index('rt')
        has_rt_col = True
    except:
        has_rt_col = False

    Xs=dict()
    irts=dict()
    items=dict()
    spellingdict=dict()
    spell_corrected = dict()
    perseverations = dict()
    intrusions = dict()
    validitems=[]
    
    # read in list of valid items when removeIntrusions = True
    if removeIntrusions:
        if (not scheme) and (not targetletter):
            raise ValueError('You need to provide a scheme or targetletter if you want to ignore intrusions!')
        elif scheme:
            with open(scheme,'rt', encoding='utf-8-sig') as fh:
                for line in fh:
                    if line[0] == "#": continue         # skip commented lines
                    try:
                        validitems.append(line.rstrip().split(',')[1].lower())
                    except:
                        pass    # fail silently on wrong format

    # read in spelling correction dictionary when spell is specified
    if spell:
        with open(spell,'rt', encoding='utf-8-sig') as spellfile:
            for line in spellfile:
                if line[0] == "#": continue         # skip commented lines
                try:
                    correct, incorrect = line.rstrip().split(',')
                    spellingdict[incorrect] = correct
                except:
                    pass    # fail silently on wrong format
   
    with open(filepath,'rt', encoding='utf-8-sig') as f:
        f.readline()    # discard header row
        for line in f:
            if line[0] == "#": continue         # skip commented lines
            row = line.rstrip().split(',')

            storerow = True  # if the row meets the filters specified then load it, else skip it
            if (subject != None) and (row[subj_col] not in subject):
                storerow = False
            if (group != None) and (row[group_col] not in group):
                storerow = False
            if (category != None) and (row[category_col] not in category):
                storerow = False
            
            if storerow == True:

                idx = row[subj_col]
                listnum_int = int(row[listnum_col])
                
                # make sure dict keys exist
                if idx not in Xs:
                    Xs[idx] = dict()
                    spell_corrected[idx] = dict()
                    perseverations[idx] = dict()
                    intrusions[idx] = dict()
                    if has_rt_col:
                        irts[idx] = dict()
                if listnum_int not in Xs[idx]:
                    Xs[idx][listnum_int] = []
                    spell_corrected[idx][listnum_int] = []
                    perseverations[idx][listnum_int] = []
                    intrusions[idx][listnum_int] = []
                    if has_rt_col:
                        irts[idx][listnum_int] = []
                if idx not in items:
                    items[idx] = dict()
                    
                # basic clean-up
                item=row[item_col].lower()
                if cleanBadChars:
                    badchars=" '-\"\\;?"
                    for char in badchars:
                        item=item.replace(char,"")
                if item in list(spellingdict.keys()):
                    newitem = spellingdict[item]
                    spell_corrected[idx][listnum_int].append((item, newitem))
                    item = newitem
                if has_rt_col:
                    irt=row[rt_col]
                if item not in list(items[idx].values()):
                    if (item in validitems) or (not removeIntrusions) or (item[0] == targetletter):
                        item_count = len(items[idx])
                        items[idx][item_count]=item
                    else:
                        intrusions[idx][listnum_int].append(item) # record as intrusion

                # add item to list
                try:
                    itemval=list(items[idx].values()).index(item)
                    if (not removePerseverations) or (itemval not in Xs[idx][listnum_int]):   # ignore any duplicates in same list resulting from spelling corrections
                        if (item in validitems) or (not removeIntrusions) or (item[0] == targetletter):
                            Xs[idx][listnum_int].append(itemval)
                            if has_rt_col:
                                irts[idx][listnum_int].append(int(irt))
                    else:
                        perseverations[idx][listnum_int].append(item) # record as perseveration
                except:
                    pass                # bad practice to have empty except
   
    return Data({'Xs': Xs, 'items': items, 'irts': irts, 'structure': hierarchical, 
                 'spell_corrected': spell_corrected, 'perseverations': perseverations, 'intrusions': intrusions})

def write_graph(gs, fh, subj="NA", directed=False, extra_data={}, header=True, labels=None, sparse=False):
    """One line description here.
    
        Detailed description here. Detailed description here.  Detailed 
        description here.  
    
        Args:
            arg1 (type): Description here.
            arg2 (type): Description here.
        Returns:
            Detailed description here. Detailed description here.  Detailed 
            description here. 
    """
    onezero={True: '1', False: '0'}        
    import networkx as nx
    fh=open(fh,'w')
    
    # if gs is not a list of graphs, then it should be a single graph
    if not isinstance(gs, list):
        gs = [gs]

    # turn them all into networkx graphs if they aren't already
    gs = [g if type(g) == nx.classes.graph.Graph else nx.to_networkx_graph(g) for g in gs]

    # label nodes if labels are provided
    if labels != None:
        if not isinstance(labels, list):
            labels = [labels]
        gs = [nx.relabel_nodes(i[0], i[1], copy=False) for i in zip(gs, labels)]
    
    nodes = list(set(flatten_list([list(gs[i].nodes()) for i in range(len(gs))])))
    
    if header == True:
        fh.write('subj,item1,item2,edge,\n')            # default header
    elif type(header) is str:
        fh.write('subj,item1,item2,'+ header+'\n')      # manual header if string is specified
    
    for node1 in nodes:
        for node2 in nodes:
            if (node1 < node2) or ((directed) and (node1 != node2)):   # write edges in alphabetical order unless directed graph
                edge = (node1,node2)
                edgelist=""
                if sparse:
                    write_edge = 0
                else:
                    write_edge = 1
                for g in gs:
                    hasedge = onezero[g.has_edge(edge[0],edge[1])]
                    edgelist=edgelist+"," + hasedge    # assumes graph is symmetrical if directed=True !!
                    write_edge += int(hasedge)
                
                if write_edge > 0:
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
