from . import *

# stochastic search through graph by node degree (proxy for frequency)
def nodeDegreeSearch(g, td):
    numnodes=nx.number_of_nodes(g)
    if td.trim <= 1:
        numtrim=int(round(numnodes*td.trim))       # if <=1, paramater is proportion of a list
    else:
        numtrim=td.trim                            # else, parameter is length of a list
    
    # make list of nodes by frequency
    l=[]
    for i, j in g.degree().items():
        l=l+[i]*j
    
    # simulate search
    walk=[]
    
    if td.start_node[0]=="specific":
        newnode=td.start_node[1]
        walk.append(newnode)
        l[:] = [j for j in l if j != newnode]
    
    while len(walk) < numtrim:
        newnode=np.random.choice(l)
        walk.append(newnode)
        l[:] = [j for j in l if j != newnode]
    
    return walk

# cluster-based depth first search: output all neighbors of starting node (random order), then all neighbors of most recently
# outputted node, etc; when you reach a dead end, back through the list until a new node with neighbors is usable
def cbdfs(g, td):
    import scipy    
    numnodes=nx.number_of_nodes(g)
    if td.trim <= 1:
        numtrim=int(round(numnodes*td.trim))       # if <=1, paramater is proportion of a list# make list of nodes by frequency
    else:
        numtrim=td.trim                            # else, parameter is length of a list
    # simulate search
    walk=[]
    
    if (td.start_node=="stationary") or (td.jumptype=="stationary"):
        a=nx.to_numpy_array(g)
        t=a/sum(a).astype(float)
        statdist=stationary(t)
        statdist=scipy.stats.rv_discrete(values=(list(range(len(t))),statdist))
    
    if td.start_node=="stationary":
        start=statdist.rvs(random_state=seed)      # choose starting point from stationary distribution #TODO: no definition of seed
    elif td.start_node=="uniform":
        start=np.random.choice(nx.nodes(g))        # choose starting point uniformly
    elif td.start_node[0]=="specific":
        start=td.start_node[1]
    
    unused_nodes=g.nodes()
    walk.append(start)
    unused_nodes.remove(start)
    currentnode=start
    
    # will have problems with disconnected graphs if numtrim is too high!
    while len(walk) < numtrim:
        next_nodes=g.neighbors(currentnode)
        next_nodes[:]=[i for i in next_nodes if i in unused_nodes]
        np.random.shuffle(next_nodes)
        if len(next_nodes) > 0:
            walk = walk + next_nodes
            unused_nodes[:] = [i for i in unused_nodes if i not in next_nodes]
            currentnode=walk[-1]
        else:
            currentnode = walk[walk.index(currentnode)-1]
    if len(walk) > numtrim:
        walk = walk[:numtrim]
    return walk

def spreadingActivationSearch(g, td, decay):
    import scipy    
    numnodes=nx.number_of_nodes(g)
    if td.trim <= 1:
        numtrim=int(round(numnodes*td.trim))       # if <=1, paramater is proportion of a list# make list of nodes by frequency
    else:
        numtrim=td.trim                            # else, parameter is length of a list
    
    if (td.start_node=="stationary") or (td.jumptype=="stationary"):
        a=nx.to_numpy_array(g)
        t=a/sum(a).astype(float)
        statdist=stationary(t)
        statdist=scipy.stats.rv_discrete(values=(list(range(len(t))),statdist))
    
    if td.start_node=="stationary":
        start=statdist.rvs(random_state=seed)      # choose starting point from stationary distribution #TODO: no definition of seed
    elif td.start_node=="uniform":
        start=np.random.choice(nx.nodes(g))        # choose starting point uniformly
    elif td.start_node[0]=="specific":
        start=td.start_node[1]
    
    activations=dict.fromkeys(list(range(numnodes)), 0)

    activations[start]=1.0
    walk=[start]
    
    while len(walk) < numtrim:
        newacts=dict.fromkeys(list(range(numnodes)), 0)
        walknodes=[]
        for node in range(numnodes):
            nn=g.neighbors(node)
            newact=activations[node]*decay
            for i in nn:
                newact += activations[i]
            if newact > 1.0:
                newact = 1.0
            newacts[node] = newact
        
        for i in activations:            # batch update
            activations[i]=newacts[i]
         
        denom = float(sum([activations[i] for i in activations if i not in walk]))
        probs=[activations[i]/denom for i in activations]
        for i in walk:
            probs[i]=0.0

        newnode=np.random.choice(list(range(numnodes)),p=probs)
        walk.append(newnode)
        activations[newnode]=1.0
        
    return walk
