#!/usr/bin/env python
# coding: utf-8

# ## Transition Model
# Kesong Cao
# $$P = P(G_1,...,G_m|X_{11},...,X_{mk}) = \prod_1^m \prod_1^k P(X_{tk}|G_t)  \times P(G_1) \prod_2^m P(G_t|G_{t-1}) $$
# $$ \log P = \sum_1^m \sum_1^k \log P(X_{tk}|G_t) + \log P(G_1) + \sum_2^m \log P(G_t|G_{t-1}) \quad (*)$$

# In[1]:


from __future__ import print_function
import numpy as np
import pickle
import random
import copy
import os
import snafu
import networkx as nx
import matplotlib.pyplot as plt


# In[2]:


def P(G2, G1, alpha=0.8, logMode=True):
    """Calculate P(G2|G1)
    param: G2: Graph 2
    param: G1: Graph 1
    param: logMode: True if using log
    raise: ValueError if G1 and G2 have invalid shapes.
    return: P(G2|G1)
    """
    if G2.shape!=G1.shape or G1.shape[0]!=G1.shape[1]:
        raise ValueError("Two matrices must have the same, valid (n by n) shape!")
    alpha = alpha
    beta = 1-alpha
    n = G1.shape[0]
    ans = 0.0 if logMode else 1.0
    for i in range(n):
        for j in range(i+1, n):
            if G1[i][j]==G2[i][j]:
                ans = (ans + np.log(alpha)) if logMode else (ans * alpha)
            else:
                ans = (ans + np.log(beta)) if logMode else (ans * beta)
    return ans

def random_transition(G1, alpha):
    """Generate G2 randomly based on G1
    param: G1: Graph 1
    param: alpha: [0, 1), the probability of keeping the same edge
    raise: ValueError if G1 has an invalid shape.
    return: G2
    """
    G2 = copy.deepcopy(G1)
    if G1.shape[0]!=G1.shape[1]:
        raise ValueError("The graph must have an 'n by n' shape!")
    n = G1.shape[0]
    for i in range(n):
        for j in range(i+1, n):
            if random.random() > alpha:
                G2[i][j] = 0 if G2[i][j]==1 else 1
                G2[j][i] = 0 if G2[j][i]==1 else 1
    return G2


# In[4]:


# GLOBAL VAR
graphs = ""

with open('sub_graphs.pickle','rbU') as f: # JZ br to rbU
    graphs = pickle.load(f)

# GLOBAL VAR
ids = ["S101", "S102", "S103", "S104", "S105", "S106", "S107", "S108", "S109", "S110", "S111", "S112", "S113", "S114", "S115", "S116", "S117", "S118", "S119", "S120"]

# GLOBAL VAR
toydata=snafu.DataModel({
        'jump': 0.0,
        'jumptype': "stationary",
        'priming': 0.0,
        'jumponcensored': None,
        'censor_fault': 0.0,
        'emission_fault': 0.0,
        'startX': "stationary",     # stationary, uniform, or a specific node?
        'numx': 3,                  # number of lists per subject
        'trim': 1.0 })

# GLOBAL VAR
filedata = snafu.readX(
    ids = ids,
    filepath = os.path.join(os.getcwd(), 'fluency/spring2017.csv'),
    category = "animals",
    spellfile = os.path.join(os.getcwd(), 'spellfiles/animals_snafu_spellfile.csv'),
)

Xs = filedata.rawdata['Xs']


# In[5]:


P(random_transition(graphs[0], 0.8),graphs[0],0.8,True)
# P(random_transition(graphs[0], 0.8),graphs[0],0.8,False)


# In[6]:


# helper method, to convert data to the format used in snafu
def process_Xs(Xs):
    ans = []
    for xnum, x in enumerate(Xs):
        ans.append(Xs[x])
    return ans

def calculate_Xs(ids):
    for i, subj in enumerate(ids):
        print(snafu.probX(process_Xs(Xs[subj]), graphs[i], toydata)[0])
# ISSUES: some probX() return -inf


# In[7]:


snafu.probX(process_Xs(Xs['S101']), graphs[0], toydata)[0] # gives us SUM log P(Xik|Gk)


# In[8]:


gen_nx_graph = nx.convert_matrix.from_numpy_matrix # an alias
log_prob_G1 = 0.00 # to be determined

def lprob(G1, Xs1, m):
    """Calculate the probability given in equation (*) above
    param: G1: The initial graph
    param: Xs1: Lists for G1
    param: m: number of iterations
    return: lprob: probability (in logarithm)
    return: Gs: Graphs
    return: Xss: Lists
    """
    Gs = [copy.deepcopy(G1)]
    Xss = [process_Xs(copy.deepcopy(Xs1))]
    lprob = 0.0 + log_prob_G1 + snafu.probX(Xss[0], Gs[0], toydata)[0]
    for i in range(m-1):
        Gs.append(random_transition(Gs[i], 0.8))
        lprob += P(Gs[i+1], Gs[i], 0.8, True)
        Xss.append(snafu.genX(gen_nx_graph(Gs[i+1]), toydata)[0]) # [0] because genX generates 3*3 lists
        lprob += snafu.probX(Xss[i+1], Gs[i+1], toydata)[0]
    return lprob, Gs, Xss


# In[9]:


# GLOBAL VARS
p, origGs, origXss = lprob(G1 = graphs[0], Xs1 = Xs['S101'], m = 5)
p


# In[10]:


def maxlprob(origGs, Xss, times, output_info = True):
    """Maximize lprob by randomly switching edges locally
    param: Gs: Input graphs
    param: Xss: Input lists
    return: Gs: Output graphs
    return: maxlprob: probability (in logarithm)
    """
    Gs = copy.deepcopy(origGs)
    m = len(Gs)
    n = len(Gs[0])
    maxlprobs = []
    Gss = []
    for k in range(times):    
        maxlprob = 0.0 + log_prob_G1 + snafu.probX(Xss[0], Gs[0], toydata)[0]
    
        # one iteration of graphs
        it = [i for i in range(1, m-1)] 
        random.shuffle(it)
        for i in it:
            if output_info:
                print("Graph",i)
            localp = snafu.probX(Xss[i], Gs[i], toydata)[0] + P(Gs[i], Gs[i-1], 0.8, True) + P(Gs[i+1], Gs[i], 0.8, True)
            if output_info:
                print("  Initial", localp)

            rc = []
            for r in range(n):
                for c in range(r+1,n):
                    rc.append((r,c))
            random.shuffle(rc)
            numpos = 0
            numtotal = 0
            for r, c in rc:
                numtotal += 1
                if(output_info and numtotal%100==0):
                    print('|', end='')
                Gs[i][r][c] = 1 - Gs[i][r][c]
                newp = snafu.probX(Xss[i], Gs[i], toydata)[0] +  P(Gs[i], Gs[i-1], 0.8, True) +  P(Gs[i+1], Gs[i], 0.8, True)
                if newp>localp:
                    numpos +=1
                    localp=newp
                else:
                    Gs[i][r][c] = 1 - Gs[i][r][c]

                if output_info:
                    print(" => positive swapping:", numpos/numtotal*100, "%\n  Final", localp)

        for i in range(1, m):    
            maxlprob += P(Gs[i], Gs[i-1], 0.8, True)
            maxlprob += snafu.probX(Xss[i], Gs[i], toydata)[0]
        if output_info:
            print("\nmaxlprob at iteration",k+1,"=", maxlprob)
        maxlprobs.append(maxlprob)
        Gss.append(copy.deepcopy(Gs))
    return Gs, maxlprob, Gss, maxlprobs


# In[ ]:


# TESTING: takes a LONG time to run!
Gs = copy.deepcopy(origGs)
Xss = copy.deepcopy(origXss)
import cProfile
cProfile.run("outputGs, maxp, _, __ = maxlprob(Gs, Xss, 5)")
maxp


# In[11]:


def distance(Gs1, Gs2):
    m = len(Gs1)
    n = len(Gs1[0])
    res = 0
    for k in range(m):
        for i in range(n):
            for j in range(n):
                if Gs1[k][i][j] != Gs2[k][i][j]:
                    res += 1
    return res


# ## PLOTTING part
# work in progess

# In[16]:


# GLOBAL VAR
m = 5 # number of time series 
# GLOBAL VAR
it = 5 # number of maxlprob iterations

# GLOBAL VAR; generate a naive random walk (theta=0) from our lists Xs using S120
nrwG = snafu.nrw(Xs= process_Xs(Xs['S120']), numnodes= len(graphs[19]))

# GLOBAL VAR
nrwlprob, Gs, Xss = lprob(nrwG, Xs['S120'], m)
# GLOBAL VAR
numnodes = len(graphs[19])
print(numnodes, nrwlprob)
# GLOBAL VAR
baseline = [nrwlprob for i in range(it+1)]

# GLOBAL VAR
startingGs = []
for i in range(m):
    startingGs.append(snafu.nrw(Xss[i], numnodes = numnodes))
#     print(startingGs[i])


# In[17]:


# GLOBAL VARS
outputG, maxp, outputGs, maxps = maxlprob(startingGs, Xss, it)


# In[18]:


iterations = np.arange(it+1)
fig, ax = plt.subplots(figsize=(5, 3))
ax.plot(iterations, baseline)
#ax.plot(iterations, [-2740.6, *maxps]) # manually added maxlprob "Initial" values at iteration 1 


# In[19]:


distances = []
for i in range(it):
    distances.append(distance(Gs, outputGs[i]))


# In[20]:


fig, ax = plt.subplots(figsize=(5, 3))
ax.plot(iterations[1:], distances)


# In[ ]:




