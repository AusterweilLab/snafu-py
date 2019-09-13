# these functions need to be made more extensible

from . import *

def commonNode(graph, items, node1, node2, numsims=100, jumpval=0.0):
    import scipy.sparse as sp
    
    if sp.issparse(graph):
        sparse=True
    else:
        sparse=False
    
    def chooseRandomNeighbor(pos):
        if sparse:
            return np.random.choice(sp.find(graph[pos])[1])
        else:
            return np.random.choice(np.where(graph[pos]==1)[0])

    inv_items = {v: k for k, v in items.items()}
    common_items={}
    for sim in range(numsims):
        a_idx = a_cur = inv_items[node1]
        b_idx = b_cur = inv_items[node2]
        idx=[a_idx, b_idx]
        a_hits=[]
        b_hits=[]
        intersect=[]
        while len(intersect) == 0:
            if np.random.random() < jumpval:
                a_cur = a_idx
            else:
                a_cur = chooseRandomNeighbor(a_cur)
            if np.random.random() < jumpval:
                b_cur = b_idx
            else:
                b_cur = chooseRandomNeighbor(b_cur)
            
            if a_cur not in a_hits and a_cur not in idx:
                a_hits.append(a_cur)
            if b_cur not in b_hits and b_cur not in idx:
                b_hits.append(b_cur)
            
            intersect = [val for val in a_hits if val in b_hits]

        incr_val = 1.0/len(intersect)
        for i in intersect:
            if items[i] not in common_items:
                common_items[items[i]] = incr_val
            else:
                common_items[items[i]] += incr_val
    
    for i in common_items:
        common_items[i] = common_items[i] / float(numsims)

    common_items = [(i,common_items[i]) for i in common_items]
    common_items = sorted(common_items, key=lambda tup: tup[1]) 

    return common_items

# first attempt to find similar items between node1 and node2 -- i think something is conceptually wrong here
#def commonNode(graph, items, node1, node2):
#    inv_items = {v: k for k, v in items.iteritems()}
#    t=graph/sum(graph.astype(float))                                                          # link to transition matrix
#    node_idx=[inv_items[node1], inv_items[node2]]                                             # item labels to ids
#    x=[i for i in range(len(graph)) if i not in node_idx]                                     # item ids to transition across freely
#    x1 = x + [node_idx[0], node_idx[1]]
#    x2 = x + [node_idx[1], node_idx[0]]
#    x1=np.array(x1)
#    x2=np.array(x2)
#    t1=t[x1[:,None],x1]                                                                       # re-arrange transition matrix to be in this order
#    t2=t[x2[:,None],x2]
#    
#    Q1=t1[:-1,:-1]
#    Q2=t2[:-1,:-1]
#    
#    # this can probably be made more efficient by using linalg instead of inv -- model after snafu.probX()
#    reg=(1+1e-10)                        
#    identmat=np.identity(len(graph)) * reg
#    I=identmat[:len(Q1),:len(Q1)]
#    N1=inv(I-Q1)
#    N2=inv(I-Q2)
#    #prob1=N1[-1,:-1]/sum(N1[-1,:-1])    # probability of starting at node 2 passing through node i before being absorbed by node 2
#    #prob2=N2[-1,:-1]/sum(N2[-1,:-1])    # vice versa
#    prob1=N1[-1,:-1]    # probability of starting at node 2 passing through node i before being absorbed by node 2
#    prob2=N2[-1,:-1]    # vice versa
#    #probs = np.multiply(prob1, prob2)
#    probs = np.add(prob1, prob2)
#    probs = probs/sum(probs)
#    similarityscore = sum(probs)
#    labels = [items[i] for i in x]
#    common = zip(probs, labels)
#    common = sorted(common, key=lambda tup: tup[0]) 
#
#    return zip(probs, labels), similarityscore


# monte carlo implementation of triadic comparison task, for when network is too large (works with sparse matrices)
# sloppy code put together quickly...
def triadicMonteCarlo(graph, items, triad, numsims=100,jumpval=0.0):
    import scipy.sparse as sp

    if sp.issparse(graph):
        sparse=True
    else:
        sparse=False
    
    def chooseRandomNeighbor(pos):
        if sparse:
            return np.random.choice(sp.find(graph[pos])[1])
        else:
            return np.random.choice(np.where(graph[pos]==1)[0])

    inv_items = {v: k for k, v in items.items()}
    ab = ac = bc = 0
    for sim in range(numsims):
        a_idx = a_cur = inv_items[triad[0]]
        b_idx = b_cur = inv_items[triad[1]]
        c_idx = c_cur = inv_items[triad[2]]
        fin=0
        while fin==0:
            if np.random.random() < jumpval:
                a_cur = a_idx
            else:
                a_cur = chooseRandomNeighbor(a_cur)
            if np.random.random() < jumpval:
                b_cur = b_idx
            else:
                b_cur = chooseRandomNeighbor(b_cur)
            if np.random.random() < jumpval:
                c_cur = c_idx
            else:
                c_cur = chooseRandomNeighbor(c_cur)
            
            # split wins
            numwins = 0
            if a_cur in [b_idx, c_idx]:
                numwins += 1
            if b_cur in [a_idx, c_idx]:
                numwins += 1
            if c_cur in [a_idx, b_idx]:
                numwins += 1
            if numwins > 1:
                # reset walk if there's a tie (could split wins, but then results are slightly different than analytic version)
                a_cur = a_idx
                b_cur = b_idx
                c_cur = c_idx

            
            if a_cur == b_idx:
                ab += 1.0
                fin = 1
            if a_cur == c_idx:
                ac += 1.0
                fin = 1
            if b_cur == a_idx:
                ab += 1.0
                fin = 1
            if b_cur == c_idx:
                bc += 1.0
                fin = 1
            if c_cur == a_idx:
                ac += 1.0
                fin = 1
            if c_cur == b_idx:
                bc += 1.0
                fin = 1
    numsims=float(numsims)
    return [ab/numsims, ac/numsims, bc/numsims]

def similarity(a, items, start, choices, steps="inf", jumpval=0.0):
    inv_items = {v: k for k, v in items.items()}
    t=a/sum(a.astype(float))                                                                # link to transition matrix
    choices_idx=[inv_items[i] for i in choices]                                             # item labels to ids
    x=[i for i in range(len(a)) if ((i not in choices_idx) and (i != inv_items[start]))]    # item ids to transition across freely
    x = x + [inv_items[start]] + choices_idx                                                # tack start state and absorbging states on the end
    x2=np.array(x)
    t2=t[x2[:,None],x2]                                                                       # re-arrange transition matrix to be in this order
   
    # indices/length for convenience later
    numchoices=len(choices)
    nonchoicelength = len(a)-numchoices
    start_idx = nonchoicelength-1
    choice_idx = list(range(start_idx+1,len(x)))
  
    # add jumps
    if jumpval > 0.0:
        t2 = t2 * (1.0-jumpval)
        t2[start_idx,:] = t2[start_idx,:] + jumpval        # + jumpval probability of transitioning to start node from any node
  
    # separate into Q (btw non-absorbing) and R (non-absorbing to absorbing)
    Q=t2[:nonchoicelength,:nonchoicelength]
    
    # compute absorbing probabilities
    if steps=="inf":
        # this can probably be made more efficient by using linalg instead of inv -- model after snafu.probX()
        reg=(1+1e-10)                        
        identmat=np.identity(len(a)) * reg
        I=identmat[:len(Q),:len(Q)]
        R=t2[nonchoicelength:,:nonchoicelength]
        N=inv(I-Q)
        B=np.dot(R,N)
        probs = B[:,nonchoicelength-1]
    elif steps >= 1:
        probs=[]
        newQ=np.zeros(nonchoicelength)
        newQ[start_idx]=1.0
        for stepnum in range(steps):
            ptmp=np.zeros(numchoices)
            for k in range(nonchoicelength):
                num1=newQ[k]
                for choicenum, choice in enumerate(choice_idx):
                    num2=t2[choice,k]                          # probability transitioning from k to absorbing node
                    ptmp[choicenum] += (num1*num2)
            newQ=np.inner(newQ,Q)
            probs.append(ptmp)
    else:
        print("similarity() must take steps >=1 or 'inf'")
    
    return probs

def triadicComparison(graph, items, triad, steplimit=200, jumpval=0.0):
    # should simplify this code / extend to work with arbitrary number of items
    starta = similarity(graph, items, start=triad[0], choices=[triad[1],triad[2]],steps=steplimit, jumpval=jumpval)
    startb = similarity(graph, items, start=triad[1], choices=[triad[0],triad[2]],steps=steplimit, jumpval=jumpval)
    startc = similarity(graph, items, start=triad[2], choices=[triad[0],triad[1]],steps=steplimit, jumpval=jumpval)
    
    a_term = [sum(starta[0])]
    b_term = [sum(startb[0])]
    c_term = [sum(startc[0])]
    for step in range(1,steplimit):
        a_term.append(a_term[-1] + sum(starta[step]))   # probability that chain a has terminated by step X
        b_term.append(b_term[-1] + sum(startb[step]))   # " chain b
        c_term.append(c_term[-1] + sum(startc[step]))   # " chain c
    
    # e.g., startab = probability that chain starting at a ends at b before chains starting from b or c terminate
    startab = startac = startba = startbc = startca = startcb = 0
    for step in range(steplimit):
        startab += starta[step][0] * (1-b_term[step]) * (1-c_term[step])
        startac += starta[step][1] * (1-b_term[step]) * (1-c_term[step])
        startba += startb[step][0] * (1-a_term[step]) * (1-c_term[step])
        startbc += startb[step][1] * (1-a_term[step]) * (1-c_term[step])
        startca += startc[step][0] * (1-a_term[step]) * (1-b_term[step])
        startcb += startc[step][1] * (1-a_term[step]) * (1-b_term[step])
    
    denom = (startab+startac+startba+startbc+startca+startcb)
    choiceab = (startab + startba) / denom
    choiceac = (startac + startca) / denom
    choicebc = (startbc + startcb) / denom

    return [choiceab, choiceac, choicebc]
