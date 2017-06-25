from numpy.linalg import inv
import numpy as np

# monte carlo implementation of triadic comparison task, for when network is too large
# sloppy code put together quickly...
def triadicMonteCarlo(graph, items, triad, numsims=100):
    inv_items = {v: k for k, v in items.iteritems()}
    ab = ac = bc = 0
    for simnum, sim in enumerate(range(numsims)):
        print simnum
        a_idx = a_cur = inv_items[triad[0]]
        b_idx = b_cur = inv_items[triad[1]]
        c_idx = c_cur = inv_items[triad[2]]
        fin=0
        while fin==0:
            a_cur = np.random.choice(np.where(graph[a_cur]==1)[0])
            b_cur = np.random.choice(np.where(graph[b_cur]==1)[0])
            c_cur = np.random.choice(np.where(graph[c_cur]==1)[0])
            
            # split wins
            numwins = 0
            if a_cur in [b_idx, c_idx]:
                numwins += 1
            if b_cur in [a_idx, c_idx]:
                numwins += 1
            if c_cur in [a_idx, b_idx]:
                numwins += 1
            if numwins > 0:
                winval = 1/float(numwins)
            
            if a_cur == b_idx:
                ab += winval
                fin = 1
            if a_cur == c_idx:
                ac += winval
                fin = 1
            if b_cur == a_idx:
                ab += winval
                fin = 1
            if b_cur == c_idx:
                bc += winval
                fin = 1
            if c_cur == a_idx:
                ac += winval
                fin = 1
            if c_cur == b_idx:
                bc += winval
                fin = 1
    numsims=float(numsims)
    return [ab/numsims, ac/numsims, bc/numsims]

def similarity(a, items, start, choices, steps="inf"):
    inv_items = {v: k for k, v in items.iteritems()}
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
    choice_idx = range(start_idx+1,len(x))
   
    # separate into Q (btw non-absorbing) and R (non-absorbing to absorbing)
    Q=t2[:nonchoicelength,:nonchoicelength]
    
    # compute absorbing probabilities
    if steps=="inf":
        # this can probably be made more efficient by using linalg instead of inv -- model after rw.probX()
        reg=(1+1e-10)                        
        identmat=np.identity(len(a)) * reg
        I=identmat[:len(Q),:len(Q)]
        R=t2[nonchoicelength:,:nonchoicelength]
        N=inv(I-Q)
        B=np.dot(R,N)
        probs = B[:,nonchoicelength-1]
    else:
        if steps >= 1:
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
            print "similarity() must take steps >=1 or 'inf'"
    
    return probs

def triadicComparison(graph, items, triad, steplimit=200):
    # should simplify this code / extend to work with arbitrary number of items
    starta = similarity(graph, items, start=triad[0], choices=[triad[1],triad[2]],steps=steplimit)
    startb = similarity(graph, items, start=triad[1], choices=[triad[0],triad[2]],steps=steplimit)
    startc = similarity(graph, items, start=triad[2], choices=[triad[0],triad[1]],steps=steplimit)
    
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
